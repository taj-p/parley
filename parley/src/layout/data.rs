// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::inline_box::InlineBox;
use crate::layout::{ContentWidths, LineMetrics, RunMetrics, Style};
use crate::style::Brush;
use crate::util::nearly_zero;
use crate::{FontData, IndentOptions, InlineBoxKind, LineHeight, OverflowWrap, TextWrapMode};
use core::ops::Range;

use alloc::vec::Vec;
use parley_core::shape::{ClusterData, ClusterInfo, Whitespace, to_whitespace};
use parley_core::{Boundary, CharInfo, Glyph};

/// `HarfRust`-based run data
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RunData {
    /// Index of the font for the run.
    pub(crate) font_index: usize,
    /// Font size.
    pub(crate) font_size: f32,
    /// Font attributes, needed for accessibility.
    pub(crate) font_attrs: fontique::Attributes,
    /// Synthesis for rendering (contains variation settings)
    pub(crate) synthesis: fontique::Synthesis,
    /// Range of normalized coordinates in the layout data.
    pub(crate) coords_range: Range<usize>,
    /// Range of the source text.
    pub(crate) text_range: Range<usize>,
    /// Bidi level for the run.
    pub(crate) bidi_level: u8,
    /// Range of clusters.
    pub(crate) cluster_range: Range<usize>,
    /// Base for glyph indices.
    pub(crate) glyph_start: usize,
    /// Metrics for the run.
    pub(crate) metrics: RunMetrics,
    /// Additional word spacing.
    pub(crate) word_spacing: f32,
    /// Additional letter spacing.
    pub(crate) letter_spacing: f32,
    /// Total advance of the run.
    pub(crate) advance: f32,
}

#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub enum BreakReason {
    #[default]
    None,
    Regular,
    Explicit,
    Emergency,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub(crate) struct LineData {
    /// Range of the source text.
    pub(crate) text_range: Range<usize>,
    /// Range of line items.
    pub(crate) item_range: Range<usize>,
    /// Metrics for the line.
    pub(crate) metrics: LineMetrics,
    /// The cause of the line break.
    pub(crate) break_reason: BreakReason,
    /// Maximum advance for the line.
    pub(crate) max_advance: f32,
    /// Number of justified clusters on the line.
    pub(crate) num_spaces: usize,
    /// Text indent applied to this line.
    pub(crate) indent: f32,
}

impl LineData {
    pub(crate) fn size(&self) -> f32 {
        self.metrics.line_height
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LineItemData {
    /// Whether the item is a run or an inline box
    pub(crate) kind: LayoutItemKind,
    /// The index of the run or inline box in the runs or `inline_boxes` vec
    pub(crate) index: usize,
    /// Bidi level for the item (used for reordering)
    pub(crate) bidi_level: u8,
    /// Advance (size in direction of text flow) for the run.
    pub(crate) advance: f32,

    // Fields that only apply to text runs (Ignored for boxes)
    // TODO: factor this out?
    /// True if the run is composed entirely of whitespace.
    pub(crate) is_whitespace: bool,
    /// True if the run ends in whitespace.
    pub(crate) has_trailing_whitespace: bool,
    /// Range of the source text.
    pub(crate) text_range: Range<usize>,
    /// Range of clusters.
    pub(crate) cluster_range: Range<usize>,
}

impl LineItemData {
    pub(crate) fn is_text_run(&self) -> bool {
        self.kind == LayoutItemKind::TextRun
    }

    #[inline(always)]
    pub(crate) fn is_rtl(&self) -> bool {
        self.bidi_level & 1 != 0
    }

    /// If the item is a text run
    ///   - Determine if it consists entirely of whitespace (`is_whitespace` property)
    ///   - Determine if it has trailing whitespace (`has_trailing_whitespace` property)
    pub(crate) fn compute_whitespace_properties<B: Brush>(&mut self, layout_data: &LayoutData<B>) {
        // Skip items which are not text runs
        if self.kind != LayoutItemKind::TextRun {
            return;
        }

        self.is_whitespace = true;
        if self.is_rtl() {
            // RTL runs check for "trailing" whitespace at the front.
            for cluster in layout_data.clusters[self.cluster_range.clone()].iter() {
                if cluster.info.is_whitespace() {
                    self.has_trailing_whitespace = true;
                } else {
                    self.is_whitespace = false;
                    break;
                }
            }
        } else {
            for cluster in layout_data.clusters[self.cluster_range.clone()]
                .iter()
                .rev()
            {
                if cluster.info.is_whitespace() {
                    self.has_trailing_whitespace = true;
                } else {
                    self.is_whitespace = false;
                    break;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutItemKind {
    TextRun,
    InlineBox,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayoutItem {
    /// Whether the item is a run or an inline box
    pub(crate) kind: LayoutItemKind,
    /// The index of the run or inline box in the runs or `inline_boxes` vec
    pub(crate) index: usize,
    /// Bidi level for the item (used for reordering)
    pub(crate) bidi_level: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayoutData<B: Brush> {
    // General settings (directly from the "builder")
    /// The display scale factor
    pub(crate) scale: f32,
    /// Whether metrics should be quantized to pixel boundaries
    pub(crate) quantize: bool,
    /// The `BiDi` base level
    pub(crate) base_level: u8,
    /// The length of the text in the layout
    pub(crate) text_len: usize,

    // Output of style resolution (input to line breaking)
    pub(crate) styles: Vec<Style<B>>,
    pub(crate) inline_boxes: Vec<InlineBox>,

    // Output of shaping (input to line breaking)
    pub(crate) fonts: Vec<FontData>,
    pub(crate) coords: Vec<i16>,
    pub(crate) runs: Vec<RunData>,
    pub(crate) items: Vec<LayoutItem>,
    pub(crate) clusters: Vec<ClusterData>,
    pub(crate) glyphs: Vec<Glyph>,

    // Output of line breaking
    /// The lines in the
    pub(crate) lines: Vec<LineData>,
    /// Items within each line
    pub(crate) line_items: Vec<LineItemData>,
    /// The width constraint that was used to line break the layout
    pub(crate) layout_max_advance: f32,
    /// The computed width of the layout excluding trailing whitespace
    pub(crate) width: f32,
    /// The computed width of the layout including trailing whitespace
    pub(crate) full_width: f32,
    /// The computed height of the layout
    pub(crate) height: f32,

    // Output of alignment
    #[cfg(feature = "accesskit")]
    /// Directly store the alignment if accessibility is enabled so we can
    /// set the corresponding AccessKit property.
    pub(crate) alignment: Option<super::Alignment>,
    /// Whether the layout is aligned with [`crate::Alignment::Justify`].
    pub(crate) is_aligned_justified: bool,
    /// The text-indent amount in layout units.
    pub(crate) indent_amount: f32,
    /// Options controlling text-indent behavior (each-line, hanging).
    pub(crate) indent_options: IndentOptions,
}

impl<B: Brush> Default for LayoutData<B> {
    fn default() -> Self {
        Self {
            scale: 1.,
            quantize: true,
            base_level: 0,
            text_len: 0,
            width: 0.,
            full_width: 0.,
            height: 0.,
            fonts: Vec::new(),
            coords: Vec::new(),
            styles: Vec::new(),
            inline_boxes: Vec::new(),
            runs: Vec::new(),
            items: Vec::new(),
            clusters: Vec::new(),
            glyphs: Vec::new(),
            lines: Vec::new(),
            line_items: Vec::new(),
            #[cfg(feature = "accesskit")]
            alignment: None,
            is_aligned_justified: false,
            layout_max_advance: 0.0,
            indent_amount: 0.0,
            indent_options: IndentOptions::default(),
        }
    }
}

impl<B: Brush> LayoutData<B> {
    pub(crate) fn clear(&mut self) {
        self.scale = 1.;
        self.quantize = true;
        self.base_level = 0;
        self.text_len = 0;
        self.width = 0.;
        self.full_width = 0.;
        self.height = 0.;
        self.fonts.clear();
        self.coords.clear();
        self.styles.clear();
        self.inline_boxes.clear();
        self.runs.clear();
        self.items.clear();
        self.clusters.clear();
        self.glyphs.clear();
        self.lines.clear();
        self.line_items.clear();
    }

    /// Push an inline box to the list of items
    pub(crate) fn push_inline_box(&mut self, index: usize) {
        // Give the box the same bidi level as the preceding text run
        // (or else default to 0 if there is not yet a text run)
        let bidi_level = self.runs.last().map(|r| r.bidi_level).unwrap_or(0);

        self.items.push(LayoutItem {
            kind: LayoutItemKind::InlineBox,
            index,
            bidi_level,
        });
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_run(
        &mut self,
        font: FontData,
        font_size: f32,
        font_attrs: fontique::Attributes,
        synthesis: fontique::Synthesis,
        glyph_buffer: &harfrust::GlyphBuffer,
        bidi_level: u8,
        style_index: u16,
        word_spacing: f32,
        letter_spacing: f32,
        source_text: &str,
        char_infos: &[CharInfo], // From text analysis
        char_style_indices: &[u16],
        text_range: Range<usize>, // The text range this run covers
        coords: &[harfrust::NormalizedCoord],
    ) {
        let coords_start = self.coords.len();
        self.coords.extend(coords.iter().map(|c| c.to_bits()));
        let coords_end = self.coords.len();

        let font_index = self
            .fonts
            .iter()
            .position(|f| *f == font)
            .unwrap_or_else(|| {
                let index = self.fonts.len();
                self.fonts.push(font);
                index
            });

        let metrics = {
            let font = &self.fonts[font_index];
            let font_ref = skrifa::FontRef::from_index(font.data.as_ref(), font.index).unwrap();
            skrifa::metrics::Metrics::new(&font_ref, skrifa::prelude::Size::new(font_size), coords)
        };
        let units_per_em = metrics.units_per_em as f32;

        let metrics = {
            let (underline_offset, underline_size) = if let Some(underline) = metrics.underline {
                (underline.offset, underline.thickness)
            } else {
                // Default values from Harfbuzz: https://github.com/harfbuzz/harfbuzz/blob/00492ec7df0038f41f78d43d477c183e4e4c506e/src/hb-ot-metrics.cc#L334
                let default = units_per_em / 18.0;
                (default, default)
            };
            let (strikethrough_offset, strikethrough_size) =
                if let Some(strikeout) = metrics.strikeout {
                    (strikeout.offset, strikeout.thickness)
                } else {
                    // Default values from HarfBuzz: https://github.com/harfbuzz/harfbuzz/blob/00492ec7df0038f41f78d43d477c183e4e4c506e/src/hb-ot-metrics.cc#L334-L347
                    (metrics.ascent / 2.0, units_per_em / 18.0)
                };

            // Compute line height
            let style = &self.styles[style_index as usize];
            let line_height = match style.line_height {
                LineHeight::Absolute(value) => value,
                LineHeight::FontSizeRelative(value) => value * font_size,
                LineHeight::MetricsRelative(value) => {
                    (metrics.ascent - metrics.descent + metrics.leading) * value
                }
            };

            RunMetrics {
                ascent: metrics.ascent,
                descent: -metrics.descent,
                leading: metrics.leading,
                underline_offset,
                underline_size,
                strikethrough_offset,
                strikethrough_size,
                line_height,
                x_height: metrics.x_height,
                cap_height: metrics.cap_height,
            }
        };

        let cluster_range = self.clusters.len()..self.clusters.len();

        let mut run = RunData {
            font_index,
            font_size,
            font_attrs,
            synthesis,
            coords_range: coords_start..coords_end,
            text_range,
            bidi_level,
            cluster_range,
            glyph_start: self.glyphs.len(),
            metrics,
            word_spacing,
            letter_spacing,
            advance: 0.,
        };

        // `HarfRust` returns glyphs in visual order, so we need to process them as such while
        // maintaining logical ordering of clusters.

        let glyph_infos = glyph_buffer.glyph_infos();
        if glyph_infos.is_empty() {
            return;
        }
        let glyph_positions = glyph_buffer.glyph_positions();
        let scale_factor = font_size / units_per_em;
        let cluster_range_start = self.clusters.len();
        let is_rtl = bidi_level & 1 == 1;
        if !is_rtl {
            run.advance = process_clusters(
                Direction::Ltr,
                &mut self.clusters,
                &mut self.glyphs,
                scale_factor,
                glyph_infos,
                glyph_positions,
                char_infos,
                char_style_indices,
                source_text.char_indices(),
            );
        } else {
            run.advance = process_clusters(
                Direction::Rtl,
                &mut self.clusters,
                &mut self.glyphs,
                scale_factor,
                glyph_infos,
                glyph_positions,
                char_infos,
                char_style_indices,
                source_text.char_indices().rev(),
            );
            // Reverse clusters into logical order for RTL
            let clusters_len = self.clusters.len();
            self.clusters[cluster_range_start..clusters_len].reverse();
        }

        run.cluster_range = cluster_range_start..self.clusters.len();
        if !run.cluster_range.is_empty() {
            self.runs.push(run);
            self.items.push(LayoutItem {
                kind: LayoutItemKind::TextRun,
                index: self.runs.len() - 1,
                bidi_level,
            });
        }
    }

    pub(crate) fn finish(&mut self) {
        for run in &self.runs {
            let word = run.word_spacing;
            let letter = run.letter_spacing;
            if nearly_zero(word) && nearly_zero(letter) {
                continue;
            }
            let clusters = &mut self.clusters[run.cluster_range.clone()];
            for cluster in clusters {
                let mut spacing = letter;
                if !nearly_zero(word) && cluster.info.whitespace().is_space_or_nbsp() {
                    spacing += word;
                }
                if !nearly_zero(spacing) {
                    cluster.advance += spacing;
                    if cluster.glyph_len != 0xFF {
                        let start = run.glyph_start + cluster.glyph_offset as usize;
                        let end = start + cluster.glyph_len as usize;
                        let glyphs = &mut self.glyphs[start..end];
                        if let Some(last) = glyphs.last_mut() {
                            last.advance += spacing;
                        }
                    }
                }
            }
        }
    }

    // TODO: this method does not handle mixed direction text at all.
    pub(crate) fn calculate_content_widths(&self) -> ContentWidths {
        fn whitespace_advance(cluster: Option<&ClusterData>) -> f32 {
            cluster
                .filter(|cluster| cluster.info.whitespace().is_space_or_nbsp())
                .map_or(0.0, |cluster| cluster.advance)
        }

        let mut min_width = 0.0_f32;
        let mut max_width = 0.0_f32;

        let mut running_min_width = 0.0;
        let mut running_max_width = 0.0;
        let mut text_wrap_mode = TextWrapMode::Wrap;
        let mut prev_cluster: Option<&ClusterData> = None;
        let is_rtl = self.base_level & 1 == 1;
        for item in &self.items {
            match item.kind {
                LayoutItemKind::TextRun => {
                    let run = &self.runs[item.index];
                    let clusters = &self.clusters[run.cluster_range.clone()];
                    if is_rtl {
                        prev_cluster = clusters.first();
                    }
                    for cluster in clusters {
                        let boundary = cluster.info.boundary();
                        let style = &self.styles[cluster.style_index as usize];
                        let prev_text_wrap_mode = text_wrap_mode;
                        text_wrap_mode = style.text_wrap_mode;
                        if boundary == Boundary::Mandatory
                            || (prev_text_wrap_mode == TextWrapMode::Wrap
                                && (boundary == Boundary::Line
                                    || style.overflow_wrap == OverflowWrap::Anywhere))
                        {
                            let trailing_whitespace = whitespace_advance(prev_cluster);
                            min_width = min_width.max(running_min_width - trailing_whitespace);
                            running_min_width = 0.0;
                            if boundary == Boundary::Mandatory {
                                max_width = max_width.max(running_max_width - trailing_whitespace);
                                running_max_width = 0.0;
                            }
                        }
                        running_min_width += cluster.advance;
                        running_max_width += cluster.advance;
                        if !is_rtl {
                            prev_cluster = Some(cluster);
                        }
                    }
                    let trailing_whitespace = whitespace_advance(prev_cluster);
                    min_width = min_width.max(running_min_width - trailing_whitespace);
                }
                LayoutItemKind::InlineBox => {
                    let ibox = &self.inline_boxes[item.index];
                    if ibox.kind == InlineBoxKind::InFlow {
                        running_max_width += ibox.width;
                        if text_wrap_mode == TextWrapMode::Wrap {
                            let trailing_whitespace = whitespace_advance(prev_cluster);
                            min_width = min_width.max(running_min_width - trailing_whitespace);
                            min_width = min_width.max(ibox.width);
                            running_min_width = 0.0;
                        } else {
                            running_min_width += ibox.width;
                        }
                    }
                    prev_cluster = None;
                }
            }
            let trailing_whitespace = whitespace_advance(prev_cluster);
            max_width = max_width.max(running_max_width - trailing_whitespace);
        }

        let trailing_whitespace = whitespace_advance(prev_cluster);
        min_width = min_width.max(running_min_width - trailing_whitespace);

        ContentWidths {
            min: min_width,
            max: max_width,
        }
    }
}

/// Maximum number of glyphs a single `ClusterData` can reference.
///
/// `ClusterData::glyph_len` is a `u8` and `0xFF` is the inline-glyph sentinel.
const MAX_CLUSTER_GLYPHS: u32 = 0xFE;

/// Processes shaped glyphs from `HarfRust` and converts them into `ClusterData` and `Glyph`,
/// emitting one `ClusterData` per extended grapheme cluster.
///
/// # Parameters
///
/// ## Output Parameters (mutated by this function):
/// * `clusters` - Vector where new `ClusterData` entries will be pushed.
/// * `glyphs` - Vector where new `Glyph` entries will be pushed. Note: single-glyph clusters
///   with zero offsets may be inlined directly into `ClusterData`.
///
/// ## Input Parameters:
/// * `direction` - Direction of the text.
/// * `scale_factor` - Scaling factor used to convert font units to the target size.
/// * `glyph_infos` - `HarfRust` glyph information in visual order.
/// * `glyph_positions` - `HarfRust` glyph positioning data in visual order.
/// * `char_infos` - Character information from text analysis, indexed by cluster value.
/// * `char_indices_iter` - Iterator over (`byte_offset`, `char`) pairs from the source text.
///   Should be in logical order (forward for LTR, reverse for RTL).
fn process_clusters<I: Iterator<Item = (usize, char)>>(
    direction: Direction,
    clusters: &mut Vec<ClusterData>,
    glyphs: &mut Vec<Glyph>,
    scale_factor: f32,
    glyph_infos: &[harfrust::GlyphInfo],
    glyph_positions: &[harfrust::GlyphPosition],
    char_infos: &[CharInfo],
    char_style_indices: &[u16],
    char_indices_iter: I,
) -> f32 {
    let mut char_indices_iter = char_indices_iter;
    let mut group_start_char = char_indices_iter.next().unwrap();
    let mut total_glyphs: u32 = 0;
    let mut cluster_glyph_offset: u32 = 0;
    let mut cluster_id = glyph_infos.first().unwrap().cluster;
    let mut run_advance = 0.0;
    let mut cluster_advance = 0.0;
    // If the current cluster might be a single-glyph, zero-offset cluster, we defer
    // pushing the first glyph to `glyphs` because it might be inlined into `ClusterData`.
    let mut pending_inline_glyph: Option<Glyph> = None;

    // The mental model for understanding this function is best grasped by first reading
    // the HarfBuzz docs on [clusters](https://harfbuzz.github.io/working-with-harfbuzz-clusters.html).
    //
    // Each source character was inserted into `HarfRust`'s buffer with the logical (char) index
    // of the extended grapheme cluster it belongs to as its cluster value (see `shape_item`).
    // `HarfRust` assigns the minimum value to merged clusters because the minimum
    // ID is selected for [merging](https://github.com/harfbuzz/harfrust/blob/a38025fb336230b492366740c86021bb406bcd0d/src/hb/buffer.rs#L920-L924),
    // so every output cluster value is the logical index of a shaped cluster's first character,
    // and a shaped cluster always covers whole graphemes.
    //
    // `char_span` is the number of characters in the current shaped cluster, derived from
    // cluster value deltas. Which neighboring cluster to compare against depends on `direction`:
    //   - In LTR, `char_span` is the difference between the next cluster and the current cluster.
    //   - In RTL, `char_span` is the difference between the last cluster and the current cluster.
    // This is because we must compare the current cluster to its next larger ID (in other words, the next
    // logical index, which is visually downstream in LTR and visually upstream in RTL).
    //
    // For example, consider the LTR text for "afi" where "fi" form a ligature.
    //   Initial cluster values: 0, 1, 2 (logical + visual order)
    //   `HarfRust` assignation: 0, 1, 1
    //   Cluster count:          2
    //   `char_span`:            (1 - 0 =) 1, (3 - 1 =) 2
    //
    // Now consider the RTL text for "حداً".
    //   Initial cluster values:  0, 1, 2, 3 (logical, or in-memory, order)
    //   Reversed cluster values: 3, 2, 1, 0 (visual order - the return order of `HarfRust` for RTL)
    //   `HarfRust` assignation:  3, 2, 0, 0
    //   Cluster count:           3
    //   `char_span`:             (4 - 3 =) 1, (3 - 2 =) 1, (2 - 0 =) 2
    //
    // (In this example `ا` and `ً` are one grapheme, so the last shaped cluster is emitted as a
    // single `ClusterData`; only shaped clusters spanning *multiple* graphemes are split.)
    let char_span = |next_cluster: u32, current_cluster: u32, last_cluster: u32| match direction {
        Direction::Ltr => next_cluster - current_cluster,
        Direction::Rtl => last_cluster - current_cluster,
    };
    let mut last_cluster_id: u32 = match direction {
        Direction::Ltr => 0,
        Direction::Rtl => char_infos.len() as u32,
    };

    for (glyph_info, glyph_pos) in glyph_infos.iter().zip(glyph_positions.iter()) {
        // Flush previous cluster if we've reached a new cluster
        if cluster_id != glyph_info.cluster {
            run_advance += cluster_advance;
            let span = char_span(glyph_info.cluster, cluster_id, last_cluster_id);
            flush_shaped_cluster(
                direction,
                clusters,
                glyphs,
                char_infos,
                char_style_indices,
                &mut char_indices_iter,
                group_start_char,
                cluster_id,
                span,
                cluster_advance,
                &mut cluster_glyph_offset,
                &mut total_glyphs,
                &mut pending_inline_glyph,
            );
            group_start_char = char_indices_iter.next().unwrap();

            cluster_advance = 0.0;
            last_cluster_id = cluster_id;
            cluster_id = glyph_info.cluster;
        }

        let glyph = Glyph {
            id: glyph_info.glyph_id,
            x: (glyph_pos.x_offset as f32) * scale_factor,
            // Convert from font space (Y-up) to layout space (Y-down)
            y: -(glyph_pos.y_offset as f32) * scale_factor,
            advance: (glyph_pos.x_advance as f32) * scale_factor,
        };
        cluster_advance += glyph.advance;
        // Push any pending glyph. If it was a zero-offset, single glyph cluster, it would
        // have been pushed in the first `if` block.
        if let Some(pending) = pending_inline_glyph.take() {
            glyphs.push(pending);
            total_glyphs += 1;
        }
        if total_glyphs == cluster_glyph_offset && glyph.x == 0.0 && glyph.y == 0.0 {
            // Defer this potential zero-offset, single glyph cluster
            pending_inline_glyph = Some(glyph);
        } else {
            glyphs.push(glyph);
            total_glyphs += 1;
        }
    }

    // Flush the last cluster. See the comment above `char_span` for why the "next" cluster
    // value is the char count for LTR and 0 for RTL.
    run_advance += cluster_advance;
    let next_cluster_id = match direction {
        Direction::Ltr => char_infos.len() as u32,
        Direction::Rtl => 0,
    };
    let span = char_span(next_cluster_id, cluster_id, last_cluster_id);
    flush_shaped_cluster(
        direction,
        clusters,
        glyphs,
        char_infos,
        char_style_indices,
        &mut char_indices_iter,
        group_start_char,
        cluster_id,
        span,
        cluster_advance,
        &mut cluster_glyph_offset,
        &mut total_glyphs,
        &mut pending_inline_glyph,
    );
    if cfg!(debug_assertions) {
        assert!(
            char_indices_iter.next().is_none(),
            "cluster values should tile the source text exactly"
        );
    }

    run_advance
}

#[derive(Copy, Clone, PartialEq)]
enum Direction {
    Ltr,
    Rtl,
}

/// Flushes one shaped cluster (a maximal run of glyphs sharing a cluster value, covering `span`
/// chars) into one `ClusterData` per extended grapheme cluster.
///
/// `group_start_char` is the shaped cluster's first `(byte_offset, char)` in iteration order
/// (logical order for LTR, reverse-logical for RTL); the remaining `span - 1` chars are consumed
/// from `char_indices_iter`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors `process_clusters`'s loop state"
)]
fn flush_shaped_cluster<I: Iterator<Item = (usize, char)>>(
    direction: Direction,
    clusters: &mut Vec<ClusterData>,
    glyphs: &mut Vec<Glyph>,
    char_infos: &[CharInfo],
    char_style_indices: &[u16],
    char_indices_iter: &mut I,
    group_start_char: (usize, char),
    cluster_id: u32,
    span: u32,
    cluster_advance: f32,
    cluster_glyph_offset: &mut u32,
    total_glyphs: &mut u32,
    pending_inline_glyph: &mut Option<Glyph>,
) {
    let logical_start = cluster_id as usize;
    let logical_end = logical_start + span as usize;
    // Input cluster values are grapheme-aligned and `HarfBuzz` only ever merges them (keeping
    // the minimum), so every shaped cluster begins on a grapheme start — except the segment's
    // first char, which is treated as a grapheme start even when itemization split a grapheme
    // across runs (e.g. a style boundary between '\r' and '\n').
    debug_assert!(logical_start == 0 || char_infos[logical_start].is_grapheme_start());
    let grapheme_count = 1 + char_infos[logical_start + 1..logical_end]
        .iter()
        .filter(|info| info.is_grapheme_start())
        .count() as u32;

    if grapheme_count == 1 {
        // The shaped cluster covers exactly one grapheme; emit a single cluster.
        let mut last_char = group_start_char;
        for _ in 1..span {
            last_char = char_indices_iter.next().unwrap();
        }
        // Chars arrive in logical order for LTR and reverse-logical order for RTL.
        let (first_char, last_char) = match direction {
            Direction::Ltr => (group_start_char, last_char),
            Direction::Rtl => (last_char, group_start_char),
        };
        let text_offset = first_char.0;
        let text_len = last_char.0 + last_char.1.len_utf8() - text_offset;
        let source_char = first_char.1;
        let is_emoji = is_emoji(&char_infos[logical_start..logical_end]);
        let boundary = char_infos[logical_start].boundary;
        let style_index = char_style_indices[logical_start];

        if to_whitespace(source_char) == Whitespace::Newline {
            // Newline clusters are stripped of their glyph contribution.
            if let Some(pending) = pending_inline_glyph.take() {
                glyphs.push(pending);
                *total_glyphs += 1;
            }
            debug_assert!(
                matches!(*total_glyphs - *cluster_glyph_offset, 1 | 2),
                "expected a newline to shape to one glyph (or two for CRLF)"
            );
            push_cluster(
                clusters,
                boundary,
                style_index,
                source_char,
                is_emoji,
                text_offset,
                text_len,
                0,   // flags
                0,   // glyph_len
                0,   // glyph_offset
                0.0, // advance
            );
        } else if let Some(pending) = pending_inline_glyph.take() {
            // A single zero-offset glyph is stored inline within `ClusterData`.
            debug_assert_eq!(*total_glyphs, *cluster_glyph_offset);
            push_cluster(
                clusters,
                boundary,
                style_index,
                source_char,
                is_emoji,
                text_offset,
                text_len,
                0,    // flags
                0xFF, // glyph_len: inline sentinel
                pending.id,
                cluster_advance,
            );
        } else {
            let glyph_len = *total_glyphs - *cluster_glyph_offset;
            debug_assert_ne!(glyph_len, 0);
            if glyph_len <= MAX_CLUSTER_GLYPHS {
                push_cluster(
                    clusters,
                    boundary,
                    style_index,
                    source_char,
                    is_emoji,
                    text_offset,
                    text_len,
                    0, // flags
                    glyph_len as u8,
                    *cluster_glyph_offset,
                    cluster_advance,
                );
            } else {
                // A degenerate grapheme with more glyphs than `glyph_len` can express (e.g.
                // hundreds of combining marks on one base). Spill the excess into extra
                // glyph-carrying `LIGATURE_COMPONENT` pieces with an empty text range at the
                // grapheme's end, so that every glyph stays attributed to a cluster.
                //
                // The shaped cluster's glyphs are the last `glyph_len` entries of `glyphs`
                // (`pending_inline_glyph` is `None` here as the shaped cluster has more than
                // one glyph).
                let group_base = glyphs.len() - glyph_len as usize;
                let piece_advance = |piece_offset: u32, piece_len: u32| -> f32 {
                    let base = group_base + piece_offset as usize;
                    glyphs[base..base + piece_len as usize]
                        .iter()
                        .map(|g| g.advance)
                        .sum()
                };
                let main_advance = piece_advance(0, MAX_CLUSTER_GLYPHS);
                push_cluster(
                    clusters,
                    boundary,
                    style_index,
                    source_char,
                    is_emoji,
                    text_offset,
                    text_len,
                    ClusterData::LIGATURE_START,
                    MAX_CLUSTER_GLYPHS as u8,
                    *cluster_glyph_offset,
                    main_advance,
                );
                let mut piece_offset = MAX_CLUSTER_GLYPHS;
                while piece_offset < glyph_len {
                    let piece_len = (glyph_len - piece_offset).min(MAX_CLUSTER_GLYPHS);
                    push_cluster(
                        clusters,
                        Boundary::None,
                        style_index,
                        source_char,
                        is_emoji,
                        text_offset + text_len, // empty text range at the grapheme's end
                        0,
                        ClusterData::LIGATURE_COMPONENT,
                        piece_len as u8,
                        *cluster_glyph_offset + piece_offset,
                        piece_advance(piece_offset, piece_len),
                    );
                    piece_offset += piece_len;
                }
            }
        }
    } else {
        // The shaped cluster covers several graphemes (e.g. an "fi" ligature or a conjunct):
        // split it into one cluster per grapheme with the advance divided evenly. The
        // logically-first grapheme keeps all the glyphs (`LIGATURE_START`); the others carry
        // none (`LIGATURE_COMPONENT`).
        if let Some(pending) = pending_inline_glyph.take() {
            glyphs.push(pending);
            *total_glyphs += 1;
        }
        let piece_advance = cluster_advance / grapheme_count as f32;
        // `glyph_len` can't express more than `MAX_CLUSTER_GLYPHS` glyphs; a *ligature* with
        // that many glyphs is not reachable in practice (unlike a degenerate single grapheme,
        // which is handled by the spill path above).
        debug_assert!(*total_glyphs - *cluster_glyph_offset <= MAX_CLUSTER_GLYPHS);
        let glyph_len = (*total_glyphs - *cluster_glyph_offset).min(MAX_CLUSTER_GLYPHS) as u8;
        debug_assert_ne!(glyph_len, 0);

        match direction {
            Direction::Ltr => {
                // Chars arrive in logical order; a flagged char closes the previous grapheme.
                let mut piece_first_char = group_start_char;
                let mut piece_last_char = group_start_char;
                let mut piece_logical_start = logical_start;
                for i in 1..span as usize {
                    let ch = char_indices_iter.next().unwrap();
                    let logical_i = logical_start + i;
                    if char_infos[logical_i].is_grapheme_start() {
                        push_ligature_piece(
                            clusters,
                            char_infos,
                            char_style_indices,
                            piece_logical_start,
                            logical_i,
                            piece_first_char,
                            piece_last_char,
                            piece_advance,
                            (piece_logical_start == logical_start)
                                .then_some((glyph_len, *cluster_glyph_offset)),
                        );
                        piece_first_char = ch;
                        piece_logical_start = logical_i;
                    }
                    piece_last_char = ch;
                }
                push_ligature_piece(
                    clusters,
                    char_infos,
                    char_style_indices,
                    piece_logical_start,
                    logical_end,
                    piece_first_char,
                    piece_last_char,
                    piece_advance,
                    (piece_logical_start == logical_start)
                        .then_some((glyph_len, *cluster_glyph_offset)),
                );
            }
            Direction::Rtl => {
                // Chars arrive in reverse-logical order; a grapheme is complete once its first
                // char (flagged as a grapheme start) is consumed. Pieces are therefore pushed
                // in reverse-logical order, and `push_run`'s slice reversal puts them back in
                // logical order — leaving the glyph-holding `LIGATURE_START` (pushed last)
                // logically first, mirroring LTR.
                let mut piece_last_char = group_start_char;
                let mut piece_logical_end = logical_end;
                let mut piece_complete = false;
                for j in 0..span as usize {
                    let current = if j == 0 {
                        group_start_char
                    } else {
                        char_indices_iter.next().unwrap()
                    };
                    if piece_complete {
                        piece_last_char = current;
                        piece_complete = false;
                    }
                    let logical_i = logical_end - 1 - j;
                    if char_infos[logical_i].is_grapheme_start() {
                        push_ligature_piece(
                            clusters,
                            char_infos,
                            char_style_indices,
                            logical_i,
                            piece_logical_end,
                            current,
                            piece_last_char,
                            piece_advance,
                            (logical_i == logical_start)
                                .then_some((glyph_len, *cluster_glyph_offset)),
                        );
                        piece_logical_end = logical_i;
                        piece_complete = true;
                    }
                }
                debug_assert!(piece_complete, "every char should belong to a grapheme");
            }
        }
    }

    *cluster_glyph_offset = *total_glyphs;
}

/// Pushes one grapheme piece of a shaped cluster that spans multiple graphemes.
///
/// `glyph_range` is `Some((glyph_len, glyph_offset))` for the logically-first piece (which keeps
/// all of the shaped cluster's glyphs) and `None` for the zero-glyph components.
#[expect(clippy::too_many_arguments, reason = "plain data plumbing")]
fn push_ligature_piece(
    clusters: &mut Vec<ClusterData>,
    char_infos: &[CharInfo],
    char_style_indices: &[u16],
    logical_start: usize,
    logical_end: usize,
    first_char: (usize, char),
    last_char: (usize, char),
    advance: f32,
    glyph_range: Option<(u8, u32)>,
) {
    debug_assert!(
        to_whitespace(first_char.1) != Whitespace::Newline,
        "a shaped cluster spanning multiple graphemes should not contain a newline"
    );
    let text_offset = first_char.0;
    let text_len = last_char.0 + last_char.1.len_utf8() - text_offset;
    let (flags, glyph_len, glyph_offset) = match glyph_range {
        Some((glyph_len, glyph_offset)) => (ClusterData::LIGATURE_START, glyph_len, glyph_offset),
        None => (ClusterData::LIGATURE_COMPONENT, 0, 0),
    };
    push_cluster(
        clusters,
        char_infos[logical_start].boundary,
        char_style_indices[logical_start],
        first_char.1,
        is_emoji(&char_infos[logical_start..logical_end]),
        text_offset,
        text_len,
        flags,
        glyph_len,
        glyph_offset,
        advance,
    );
}

/// Whether any char in the range is an emoji, pictograph, or regional indicator.
fn is_emoji(char_infos: &[CharInfo]) -> bool {
    char_infos
        .iter()
        .any(|info| info.is_emoji_or_pictograph() || info.is_region_indicator())
}

#[expect(clippy::too_many_arguments, reason = "plain data plumbing")]
fn push_cluster(
    clusters: &mut Vec<ClusterData>,
    boundary: Boundary,
    style_index: u16,
    source_char: char,
    is_emoji: bool,
    text_offset: usize,
    text_len: usize,
    flags: u8,
    glyph_len: u8,
    glyph_offset: u32,
    advance: f32,
) {
    clusters.push(ClusterData {
        info: ClusterInfo::new(boundary, source_char, is_emoji),
        flags,
        style_index,
        glyph_len,
        text_len: text_len as u16,
        glyph_offset,
        text_offset: text_offset as u16,
        advance,
    });
}
