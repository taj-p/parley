// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::inline_box::InlineBox;
use crate::layout::{ContentWidths, Glyph, LineMetrics, RunMetrics, Style};
use crate::style::Brush;
use crate::util::nearly_zero;
use crate::{Font, GlyphFlags, OverflowWrap};
use core::ops::Range;

use swash::text::cluster::{Boundary, Whitespace};

use alloc::vec::Vec;

#[cfg(feature = "libm")]
#[allow(unused_imports)]
use core_maths::CoreFloat;

use skrifa::raw::TableProvider;

/// Default units per em for font scaling.
///
/// This is used as a fallback when the actual font units per em cannot be determined.
/// Most TrueType fonts use 2048, while PostScript fonts typically use 1000.
const DEFAULT_UNITS_PER_EM: f32 = 2048.0;

/// Simple synthesis specification for HarfBuzz compatibility
///
/// TODO: This is a minimal implementation that covers basic font synthesis.
/// For full compatibility with swash::Synthesis and fontique::Synthesis, we should add:
///
/// 1. **Variation settings**: `Vec<(Tag, f32)>` - for precise weight/width/slant adjustments
///    instead of just boolean bold/italic. Modern variable fonts use continuous axes
///    (e.g., weight 100-900) rather than discrete on/off switches.
///
/// 2. **Precise skew angle**: `Option<f32>` - the exact italic/oblique angle in degrees
///    instead of just a boolean. Different fonts may need different skew amounts.
///
/// 3. **Small caps synthesis**: `bool` - for synthesizing small capitals when the font
///    doesn't have native small caps support.
///
/// 4. **Advanced synthesis options**:
///    - Condensed/expanded synthesis for width adjustments
///    - Optical sizing hints
///    - Any other synthesis features that HarfBuzz supports
///
/// The current implementation satisfies basic use cases but should be expanded when:
/// - Variable font support becomes critical
/// - More sophisticated font matching is needed  
/// - Full swash feature parity is required
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct HarfSynthesis {
    /// Fake bold by emboldening glyphs
    pub bold: bool,
    /// Fake italic by skewing glyphs  
    pub italic: bool,
}

impl Default for HarfSynthesis {
    fn default() -> Self {
        Self {
            bold: false,
            italic: false,
        }
    }
}

/// Simple cluster info for HarfBuzz compatibility
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) struct HarfClusterInfo {
    boundary: Option<Boundary>,
    source_char: char,
}

impl HarfClusterInfo {
    pub(crate) fn new(boundary: Option<Boundary>, source_char: char) -> Self {
        Self {
            boundary,
            source_char,
        }
    }

    /// Get boundary type (critical for line breaking)
    pub(crate) fn boundary(&self) -> Option<Boundary> {
        self.boundary
    }

    /// Get whitespace type
    pub(crate) fn whitespace(&self) -> Whitespace {
        match self.source_char {
            ' ' => Whitespace::Space,
            '\t' => Whitespace::Tab,
            '\n' => Whitespace::Newline,
            '\r' => Whitespace::Newline,
            '\u{00A0}' => Whitespace::Space, // Non-breaking space
            _ => Whitespace::None,
        }
    }

    /// Check if this is a word boundary
    pub(crate) fn is_boundary(&self) -> bool {
        self.boundary.is_some()
    }

    /// Check if this is an emoji
    pub(crate) fn is_emoji(&self) -> bool {
        // TODO: Simple emoji detection - could be enhanced
        matches!(self.source_char as u32, 0x1F600..=0x1F64F | 0x1F300..=0x1F5FF | 0x1F680..=0x1F6FF | 0x2600..=0x26FF | 0x2700..=0x27BF)
    }

    /// Check if this is any kind of whitespace
    pub(crate) fn is_whitespace(&self) -> bool {
        self.source_char.is_whitespace()
    }
}

impl Default for HarfClusterInfo {
    fn default() -> Self {
        Self {
            boundary: None,
            source_char: ' ',
        }
    }
}

/// Cluster data - uses swash analysis with harfrust shaping
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ClusterData {
    /// Cluster information from swash text analysis (using our own type)
    pub(crate) info: HarfClusterInfo,
    /// Cluster flags (ligature info, style divergence, etc.)
    pub(crate) flags: u16,
    /// Style index for this cluster
    pub(crate) style_index: u16,
    /// Number of glyphs in this cluster (0xFF = single glyph stored inline)
    pub(crate) glyph_len: u8,
    /// Number of text bytes in this cluster
    pub(crate) text_len: u8,
    /// If `glyph_len == 0xFF`, then `glyph_offset` is a glyph identifier,
    /// otherwise, it's an offset into the glyph array with the base
    /// taken from the owning run.
    pub(crate) glyph_offset: u16,
    /// Offset into the text for this cluster
    pub(crate) text_offset: u16,
    /// Advance width for this cluster
    pub(crate) advance: f32,
}

impl ClusterData {
    pub(crate) const LIGATURE_START: u16 = 1;
    pub(crate) const LIGATURE_COMPONENT: u16 = 2;
    pub(crate) const DIVERGENT_STYLES: u16 = 4;

    pub(crate) fn is_ligature_start(self) -> bool {
        self.flags & Self::LIGATURE_START != 0
    }

    pub(crate) fn is_ligature_component(self) -> bool {
        self.flags & Self::LIGATURE_COMPONENT != 0
    }

    pub(crate) fn has_divergent_styles(self) -> bool {
        self.flags & Self::DIVERGENT_STYLES != 0
    }

    pub(crate) fn text_range(self, run: &RunData) -> Range<usize> {
        let start = run.text_range.start + self.text_offset as usize;
        start..start + self.text_len as usize
    }
}

/// Harfrust-based run data (updated to use harfrust types)
#[derive(Clone, Debug)]
pub(crate) struct RunData {
    /// Index of the font for the run.
    pub(crate) font_index: usize,
    /// Font size.
    pub(crate) font_size: f32,
    /// Harfrust-based synthesis information for the font.
    pub(crate) synthesis: HarfSynthesis,
    /// Original fontique synthesis for renderer (contains variation settings)
    pub(crate) fontique_synthesis: Option<fontique::Synthesis>,
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
}

impl LineData {
    pub(crate) fn size(&self) -> f32 {
        self.metrics.ascent + self.metrics.descent + self.metrics.leading
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

    pub(crate) fn compute_line_height<B: Brush>(&self, layout: &LayoutData<B>) -> f32 {
        match self.kind {
            LayoutItemKind::TextRun => {
                let mut line_height = 0_f32;
                let run = &layout.runs[self.index];
                let glyph_start = run.glyph_start;
                for cluster in &layout.clusters[run.cluster_range.clone()] {
                    if cluster.glyph_len != 0xFF && cluster.has_divergent_styles() {
                        let start = glyph_start + cluster.glyph_offset as usize;
                        let end = start + cluster.glyph_len as usize;
                        for glyph in &layout.glyphs[start..end] {
                            line_height = line_height
                                .max(layout.styles[glyph.style_index()].line_height.resolve(run));
                        }
                    } else {
                        line_height = line_height.max(
                            layout.styles[cluster.style_index as usize]
                                .line_height
                                .resolve(run),
                        );
                    }
                }
                line_height
            }
            LayoutItemKind::InlineBox => {
                // TODO: account for vertical alignment (e.g. baseline alignment)
                layout.inline_boxes[self.index].height
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

#[derive(Clone, Debug)]
pub(crate) struct LayoutData<B: Brush> {
    pub(crate) scale: f32,
    pub(crate) quantize: bool,
    pub(crate) has_bidi: bool,
    pub(crate) base_level: u8,
    pub(crate) text_len: usize,
    pub(crate) width: f32,
    pub(crate) full_width: f32,
    pub(crate) height: f32,
    pub(crate) fonts: Vec<Font>,
    pub(crate) coords: Vec<i16>,

    // Input (/ output of style resolution)
    pub(crate) styles: Vec<Style<B>>,
    pub(crate) inline_boxes: Vec<InlineBox>,

    // Output of shaping
    pub(crate) runs: Vec<RunData>,
    pub(crate) items: Vec<LayoutItem>,
    pub(crate) clusters: Vec<ClusterData>,
    pub(crate) glyphs: Vec<Glyph>,

    // Output of line breaking
    pub(crate) lines: Vec<LineData>,
    pub(crate) line_items: Vec<LineItemData>,

    // Output of alignment
    /// Whether the layout is aligned with [`crate::Alignment::Justified`].
    pub(crate) is_aligned_justified: bool,
    /// The width the layout was aligned to.
    pub(crate) alignment_width: f32,
}

impl<B: Brush> Default for LayoutData<B> {
    fn default() -> Self {
        Self {
            scale: 1.,
            quantize: true,
            has_bidi: false,
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
            is_aligned_justified: false,
            alignment_width: 0.0,
        }
    }
}

impl<B: Brush> LayoutData<B> {
    pub(crate) fn clear(&mut self) {
        self.scale = 1.;
        self.quantize = true;
        self.has_bidi = false;
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

    /// Push data for a new run using HarfBuzz-shaped glyph data.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_run_from_harfrust(
        &mut self,
        font: Font,
        font_size: f32,
        synthesis: HarfSynthesis,
        fontique_synthesis: fontique::Synthesis,
        glyph_buffer: &harfrust::GlyphBuffer,
        bidi_level: u8,
        word_spacing: f32,
        letter_spacing: f32,
        // NEW: Add text analysis data needed for proper clustering
        source_text: &str,
        infos: &[(swash::text::cluster::CharInfo, u16)], // From text analysis
        text_range: Range<usize>,                        // The text range this run covers
        _char_range: Range<usize>,                       // Range into infos array
        // NEW: Add actual font variations used during shaping
        variations: &[harfrust::Variation],
    ) {
        // Store font variations as normalized coordinates FIRST (before font moves)
        // Proper solution: Read font's fvar table and map variations to correct axis positions
        let coords_start = self.coords.len();

        if !variations.is_empty() {
            self.store_variations_properly(&font, variations);
        }

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

        // TODO(taj): Why do we need self.fonts? And do we need to do this here? And should I store the font ref alongside the font?
        let font = &self.fonts[font_index];
        let font = skrifa::FontRef::from_index(font.data.as_ref(), font.index).unwrap(); // TODO(taj): Handle unwrap.

        let (ascent, descent, leading, strikethrough_offset, strikethrough_size) =
            // This copies harfrust's behaviour (https://github.com/harfbuzz/harfrust/blob/a38025fb336230b492366740c86021bb406bcd0d/src/hb/glyph_metrics.rs#L55-L60)
            // but it's unclear to me whether we should use os2 only if the appropriate fs selection "use os2 metrics" bit is set.
            if let Ok(os2) = font.os2() {
                (
                    os2.s_typo_ascender(),
                    os2.s_typo_descender(),
                    os2.s_typo_line_gap(),
                    os2.y_strikeout_position(),
                    os2.y_strikeout_size(),
                )
            } else if let Ok(hhea) = font.hhea() {
                (
                    hhea.ascender().to_i16(),
                    hhea.descender().to_i16(),
                    hhea.line_gap().to_i16(),
                    i16::default(),
                    i16::default(),
                )
            } else {
                todo!()
            };

        let (underline_offset, underline_size) = if let Ok(post) = font.post() {
            // TODO: What to do if these tables don't exist? Should we actually err?
            (
                post.underline_position().to_i16(),
                post.underline_thickness().to_i16(),
            )
        } else {
            (i16::default(), i16::default())
        };

        // For now, create default metrics since we don't have them from harfrust
        let units_per_em = font
            .head()
            .map(|h| h.units_per_em() as f32)
            .unwrap_or(DEFAULT_UNITS_PER_EM);
        let metrics = RunMetrics {
            ascent: font_size * ascent as f32 / units_per_em,
            descent: -font_size * descent as f32 / units_per_em,
            leading: font_size * leading as f32 / units_per_em,
            underline_offset: font_size * underline_offset as f32 / units_per_em,
            underline_size: font_size * underline_size as f32 / units_per_em,
            strikethrough_offset: font_size * strikethrough_offset as f32 / units_per_em,
            strikethrough_size: font_size * strikethrough_size as f32 / units_per_em,
        };

        let cluster_range = self.clusters.len()..self.clusters.len();

        let mut run = RunData {
            font_index,
            font_size,
            synthesis,
            fontique_synthesis: Some(fontique_synthesis), // Store original fontique synthesis
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

        // Get harfrust glyph data
        let glyph_infos = glyph_buffer.glyph_infos();
        let glyph_positions = glyph_buffer.glyph_positions();

        if glyph_infos.is_empty() {
            return;
        }

        // Process glyphs in visual order (HarfBuzz output) but store clusters in logical order
        // Step 1: Add all glyphs to maintain visual order for rendering
        let scale_factor = font_size / units_per_em;
        let glyph_start_idx = self.glyphs.len();

        self.glyphs
            .extend(glyph_infos.iter().zip(glyph_positions.iter()).map(
                |(glyph_info, glyph_pos)| {
                    Glyph {
                        id: glyph_info.glyph_id,
                        style_index: 0, // Will be set when processing clusters
                        x: (glyph_pos.x_offset as f32) * scale_factor,
                        y: (glyph_pos.y_offset as f32) * scale_factor,
                        advance: (glyph_pos.x_advance as f32) * scale_factor,
                        cluster_index: glyph_info.cluster,
                        flags: GlyphFlags::from(glyph_info),
                    }
                },
            ));

        let is_rtl = bidi_level & 1 != 0;
        let mut visual_idx = if is_rtl { glyph_infos.len() - 1 } else { 0 };
        let mut char_idx = if is_rtl { infos.len() - 1 } else { 0 };

        let mut current_cluster_id: Option<u32> = None;
        let mut current_cluster_glyph_idx_start = glyph_start_idx + visual_idx;
        let mut current_cluster_glyph_count = 0;
        let mut current_cluster_advance = 0.0;

        let mut run_advance = 0.0;

        let mut char_indices_iter = source_text.char_indices();
        let mut start_byte_idx = char_indices_iter.next().unwrap();

        let cluster_range_start = self.clusters.len();

        assert_eq!(glyph_infos.len(), glyph_positions.len());

        let mut current_cluster_char_info: Option<&(swash::text::cluster::CharInfo, u16)> = None;

        // Cluster processing loop
        for _logical_idx in 0..glyph_infos.len() {
            let glyph_info = &glyph_infos[visual_idx];
            let char_info = &infos[char_idx];

            let global_glyph_idx = glyph_start_idx + visual_idx;
            self.glyphs[global_glyph_idx].style_index = char_info.1;

            visual_idx = if is_rtl {
                visual_idx.saturating_sub(1)
            } else {
                visual_idx + 1
            };

            // Check if we're starting a new cluster
            if current_cluster_id != Some(glyph_info.cluster) {
                // Finalize the previous cluster if it exists
                if let Some(_) = current_cluster_id {
                    let char_byte_end = char_indices_iter.next().unwrap();
                    let cluster_char_info = current_cluster_char_info.unwrap();
                    let cluster_data = self.create_cluster_data(
                        start_byte_idx.1,
                        cluster_char_info.0.boundary(),
                        start_byte_idx.0,
                        char_byte_end.0,
                        cluster_char_info.1,
                        current_cluster_glyph_count,
                        if is_rtl {
                            global_glyph_idx + 1
                        } else {
                            current_cluster_glyph_idx_start
                        },
                        current_cluster_advance,
                        run.glyph_start,
                    );
                    self.clusters.push(cluster_data);
                    start_byte_idx = char_byte_end;
                }

                current_cluster_id = Some(glyph_info.cluster);
                current_cluster_char_info = Some(char_info);
                char_idx = if is_rtl {
                    char_idx.saturating_sub(1)
                } else {
                    (char_idx + 1).min(infos.len() - 1)
                };
                run_advance += current_cluster_advance;
                current_cluster_advance = 0.0;
                current_cluster_glyph_count = 0;
                current_cluster_glyph_idx_start = global_glyph_idx;
            }

            current_cluster_advance += self.glyphs[global_glyph_idx].advance;
            current_cluster_glyph_count += 1;
        }

        // Finalize the last cluster
        if let Some(_) = current_cluster_id {
            let cluster_char_info = current_cluster_char_info.unwrap();
            let cluster_data = self.create_cluster_data(
                start_byte_idx.1,
                cluster_char_info.0.boundary(),
                start_byte_idx.0,
                source_text.len(),
                cluster_char_info.1,
                current_cluster_glyph_count,
                if is_rtl {
                    glyph_start_idx
                } else {
                    current_cluster_glyph_idx_start
                },
                current_cluster_advance,
                run.glyph_start,
            );
            self.clusters.push(cluster_data);
            run_advance += current_cluster_advance;
        }

        run.cluster_range = cluster_range_start..self.clusters.len();
        run.advance = run_advance;
        run.synthesis = synthesis;

        if !run.cluster_range.is_empty() {
            self.runs.push(run);
            self.items.push(LayoutItem {
                kind: LayoutItemKind::TextRun,
                index: self.runs.len() - 1,
                bidi_level,
            });
        }
    }

    // Helper method to create cluster data from cluster information
    fn create_cluster_data(
        &mut self,
        source_char: char,
        boundary: Boundary,
        char_byte_start: usize,
        char_byte_end: usize,
        style_index: u16,
        glyph_count: usize,
        visual_start: usize,
        cluster_advance: f32,
        run_glyph_start: usize,
    ) -> ClusterData {
        let cluster_info = HarfClusterInfo::new(Some(boundary), source_char);
        ClusterData {
            info: cluster_info,
            flags: 0, // TODO
            style_index,
            glyph_len: glyph_count as u8,
            text_len: (char_byte_end - char_byte_start) as u8,
            advance: cluster_advance,
            text_offset: char_byte_start as u16,
            glyph_offset: (visual_start - run_glyph_start) as u16,
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

        let mut running_max_width = 0.0;
        let mut prev_cluster: Option<&ClusterData> = None;
        let is_rtl = self.base_level & 1 == 1;
        for item in &self.items {
            match item.kind {
                LayoutItemKind::TextRun => {
                    let run = &self.runs[item.index];
                    let mut running_min_width = 0.0;
                    let clusters = &self.clusters[run.cluster_range.clone()];
                    if is_rtl {
                        prev_cluster = clusters.first();
                    }
                    for cluster in clusters {
                        let boundary = cluster.info.boundary();
                        let style = &self.styles[cluster.style_index as usize];
                        if matches!(boundary, Some(Boundary::Line) | Some(Boundary::Mandatory))
                            || style.overflow_wrap == OverflowWrap::Anywhere
                        {
                            let trailing_whitespace = whitespace_advance(prev_cluster);
                            min_width = min_width.max(running_min_width - trailing_whitespace);
                            running_min_width = 0.0;
                            if boundary == Some(Boundary::Mandatory) {
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
                    min_width = min_width.max(ibox.width);
                    running_max_width += ibox.width;
                    prev_cluster = None;
                }
            }
            let trailing_whitespace = whitespace_advance(prev_cluster);
            max_width = max_width.max(running_max_width - trailing_whitespace);
        }

        ContentWidths {
            min: min_width,
            max: max_width,
        }
    }

    /// Store font variations as normalized coordinates using proper axis mapping
    /// This replicates what swash did internally: read fvar table, map variations to correct positions
    fn store_variations_properly(&mut self, font: &Font, variations: &[harfrust::Variation]) {
        // Try to read font's axis layout from fvar table
        if let Ok(font_ref) = skrifa::FontRef::from_index(font.data.as_ref(), font.index) {
            if let Ok(fvar) = font_ref.fvar() {
                if let Ok(axes) = fvar.axes() {
                    let axis_count = fvar.axis_count() as usize;
                    let mut coords = vec![0i16; axis_count];

                    // Map each fontique variation to its correct axis position
                    for variation in variations {
                        let variation_tag = skrifa::Tag::from_be_bytes(variation.tag.to_be_bytes());

                        // Find which axis this variation belongs to
                        for (axis_index, axis_record) in axes.iter().enumerate() {
                            if axis_record.axis_tag() == variation_tag {
                                // Use this axis's actual range for normalization
                                let min_val = axis_record.min_value().to_f32();
                                let default_val = axis_record.default_value().to_f32();
                                let max_val = axis_record.max_value().to_f32();

                                // Generic normalization (same formula for all axes)
                                let normalized_f32 = if variation.value >= default_val {
                                    (variation.value - default_val) / (max_val - default_val)
                                } else {
                                    (variation.value - default_val) / (default_val - min_val)
                                };

                                let clamped = normalized_f32.clamp(-1.0, 1.0);
                                let normalized_coord = (clamped * 16384.0) as i16;

                                coords[axis_index] = normalized_coord;
                                break;
                            }
                        }
                    }

                    // Store all coordinates (including zeros for unused axes)
                    self.coords.extend(coords);
                    return;
                }
            }
        }

        // Fallback: simple storage if fvar reading fails
        for variation in variations {
            let normalized_f32 = (variation.value - 400.0) / (1000.0 - 400.0);
            let clamped = normalized_f32.clamp(-1.0, 1.0);
            let normalized_coord = (clamped * 16384.0) as i16;
            self.coords.push(normalized_coord);
        }
    }
}
