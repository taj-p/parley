// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::inline_box::InlineBox;
use crate::layout::{ContentWidths, Glyph, LineMetrics, RunMetrics, Style};
use crate::style::Brush;
use crate::util::nearly_zero;
use crate::{Font, OverflowWrap};
use core::ops::Range;

use swash::text::cluster::{Boundary, Whitespace};

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

#[cfg(feature = "libm")]
#[allow(unused_imports)]
use core_maths::CoreFloat;

use skrifa::raw::TableProvider;

/// Default units per em for font scaling.
///
/// This is used as a fallback when the actual font units per em cannot be determined.
/// Most TrueType fonts use 2048, while PostScript fonts typically use 1000.
///
/// TODO: Extract the actual units_per_em from the font header for more accurate scaling.
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
        // Detect whitespace from actual character content
        let _whitespace = match source_char {
            ' ' => Whitespace::Space,
            '\t' => Whitespace::Tab,
            '\n' => Whitespace::Newline,
            '\r' => Whitespace::Newline,
            '\u{00A0}' => Whitespace::Space, // Non-breaking space treated as regular space
            _ => Whitespace::None,
        };

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
        if self.source_char.is_whitespace() {
            Whitespace::Space
        } else {
            Whitespace::None
        }
    }

    /// Check if this is a word boundary
    pub(crate) fn is_boundary(&self) -> bool {
        self.boundary.is_some()
    }

    /// Check if this is an emoji
    pub(crate) fn is_emoji(&self) -> bool {
        // Simple emoji detection - could be enhanced
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
    // changed: Commenting out harfrust-specific fields for compilation
    // /// Text direction for this run
    // pub(crate) direction: harfrust::Direction,
    // /// Script for this run
    // pub(crate) script: harfrust::Script,
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
    /// Create new layout data.
    pub(crate) fn new() -> Self {
        Self {
            scale: 1.0,
            quantize: true,
            text_len: 0,
            base_level: 0,
            width: 0.0,
            full_width: 0.0,
            height: 0.0,
            fonts: Vec::new(),
            coords: Vec::new(),
            styles: Vec::new(),
            inline_boxes: Vec::new(),
            has_bidi: false,
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
        char_range: Range<usize>,                        // Range into infos array
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

        let units_per_em = font.head().unwrap().units_per_em();

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
        let metrics = RunMetrics {
            ascent: font_size * ascent as f32 / units_per_em as f32,
            descent: font_size * descent as f32 / units_per_em as f32,
            leading: font_size * leading as f32 / units_per_em as f32,
            underline_offset: font_size * underline_offset as f32 / units_per_em as f32,
            underline_size: font_size * underline_size as f32 / units_per_em as f32,
            strikethrough_offset: font_size * strikethrough_offset as f32 / units_per_em as f32,
            strikethrough_size: font_size * strikethrough_size as f32 / units_per_em as f32,
        };

        let cluster_range = self.clusters.len()..self.clusters.len();

        let mut run = RunData {
            font_index,
            font_size,
            synthesis,
            fontique_synthesis: Some(fontique_synthesis), // Store original fontique synthesis
            coords_range: coords_start..coords_end,
            text_range: text_range.clone(), // ✅ Use correct text range from parameter
            bidi_level,
            // changed: Commenting out harfrust-specific fields for compilation
            // direction: Direction::LTR, // Default to LTR for now
            // script: Script::LATIN,   // Default to LATIN for now
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

        // Map harfrust clusters to source text and create proper cluster data
        let cluster_mappings = self.map_harfrust_clusters_to_text(
            glyph_buffer,
            source_text,
            infos,
            &text_range,
            &char_range,
            bidi_level,
        );

        // Group glyphs by harfrust cluster ID
        let mut cluster_groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, info) in glyph_infos.iter().enumerate() {
            cluster_groups.entry(info.cluster).or_default().push(i);
        }

        // Process clusters and add their glyphs to the run
        let mut all_run_glyphs = Vec::new();
        let mut cluster_data_list = Vec::new();
        let mut run_advance = 0.0;

        for (cluster_id, cluster_text_range, cluster_info, style_index) in &cluster_mappings {
            if let Some(glyph_indices) = cluster_groups.get(cluster_id) {
                let cluster_glyphs: Vec<_> = glyph_indices
                    .iter()
                    .map(|&i| (&glyph_infos[i], &glyph_positions[i]))
                    .collect();

                // Store cluster data for later processing
                cluster_data_list.push((
                    *cluster_id,
                    cluster_glyphs,
                    cluster_text_range.clone(),
                    cluster_info.clone(),
                    *style_index,
                ));
            }
        }

        // For RTL text, we need to reverse the glyph order within the run to match visual order
        let is_rtl = bidi_level & 1 != 0;
        if is_rtl {
            cluster_data_list.reverse();
        }

        // Now process clusters in the correct visual order
        for (cluster_id, cluster_glyphs, cluster_text_range, cluster_info, style_index) in
            cluster_data_list
        {
            // Add glyphs to the run in visual order
            let mut cluster_advance = 0.0;
            let glyph_start = all_run_glyphs.len();

            for (info, pos) in &cluster_glyphs {
                let scale_factor = font_size / DEFAULT_UNITS_PER_EM;

                let glyph = Glyph {
                    id: info.glyph_id,
                    style_index,
                    x: (pos.x_offset as f32) * scale_factor,
                    y: (pos.y_offset as f32) * scale_factor,
                    advance: (pos.x_advance as f32) * scale_factor,
                    cluster_index: cluster_id,
                    flags: 0,
                };
                cluster_advance += glyph.advance;
                all_run_glyphs.push(glyph);
            }

            // Create cluster data
            let cluster_data = ClusterData {
                info: cluster_info,
                flags: 0,
                style_index,
                glyph_len: cluster_glyphs.len() as u8,
                text_len: cluster_text_range.len() as u8,
                advance: cluster_advance,
                text_offset: cluster_text_range.start.saturating_sub(text_range.start) as u16,
                glyph_offset: glyph_start as u16,
            };

            self.clusters.push(cluster_data);
            run.cluster_range.end += 1;
            run_advance += cluster_advance;
        }

        // Add all glyphs to the global glyph list in correct order
        self.glyphs.extend(all_run_glyphs);

        run.advance = run_advance;

        // Store final run data with harfrust synthesis
        run.synthesis = synthesis;

        // Push the run
        if !run.cluster_range.is_empty() {
            self.runs.push(run);
            self.items.push(LayoutItem {
                kind: LayoutItemKind::TextRun,
                index: self.runs.len() - 1,
                bidi_level,
            });
        }
    }

    // Helper method to map harfrust clusters back to source text
    fn map_harfrust_clusters_to_text(
        &self,
        glyph_buffer: &harfrust::GlyphBuffer,
        source_text: &str,
        infos: &[(swash::text::cluster::CharInfo, u16)],
        text_range: &Range<usize>,
        char_range: &Range<usize>,
        bidi_level: u8, // Added to handle RTL cluster ordering
    ) -> Vec<(u32, Range<usize>, HarfClusterInfo, u16)> {
        // Returns: (harfrust_cluster_id, text_byte_range, cluster_info, style_index)

        let mut clusters = Vec::new();
        let glyph_infos = glyph_buffer.glyph_infos();

        // Group glyphs by harfrust cluster ID
        let mut cluster_groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, info) in glyph_infos.iter().enumerate() {
            cluster_groups.entry(info.cluster).or_default().push(i);
        }

        // Map each harfrust cluster back to source text
        // Sort cluster IDs to process them in order
        let mut sorted_cluster_ids: Vec<u32> = cluster_groups.keys().copied().collect();
        sorted_cluster_ids.sort();

        // ✅ IMPORTANT: Reverse cluster order for RTL text to match swash behavior
        let is_rtl = bidi_level & 1 != 0;
        if is_rtl {
            sorted_cluster_ids.reverse();
        }

        for &cluster_id in sorted_cluster_ids.iter() {
            // For each cluster, map it to the corresponding character using the cluster ID
            // NOTE: cluster IDs are relative to the current text segment, not global
            let char_idx_in_range = cluster_id as usize;

            if char_idx_in_range < char_range.len() {
                let _absolute_char_idx = char_range.start + char_idx_in_range;

                // Get cluster info from swash text analysis
                // Use char_idx_in_range (relative index) instead of absolute_char_idx
                if let Some((char_info, style_index)) = infos.get(char_idx_in_range) {
                    // ✅ Extract boundary from CharInfo and create our own cluster info!
                    let boundary = char_info.boundary();
                    // Use segment-relative index since source_text is only the current segment
                    let segment_relative_char_idx = char_idx_in_range; // This is already relative to the segment
                    let source_char = source_text
                        .chars()
                        .nth(segment_relative_char_idx)
                        .unwrap_or(' ');
                    let cluster_info = HarfClusterInfo::new(Some(boundary), source_char);

                    // Calculate BYTE range for this cluster from character positions
                    // Convert character index to byte index within the segment
                    let char_byte_start = source_text
                        .char_indices()
                        .nth(segment_relative_char_idx)
                        .map(|(byte_idx, _)| byte_idx)
                        .unwrap_or(0);

                    let char_byte_end =
                        if segment_relative_char_idx + 1 < source_text.chars().count() {
                            source_text
                                .char_indices()
                                .nth(segment_relative_char_idx + 1)
                                .map(|(byte_idx, _)| byte_idx)
                                .unwrap_or(source_text.len())
                        } else {
                            source_text.len()
                        };

                    // Convert to absolute byte positions
                    let cluster_text_range =
                        (text_range.start + char_byte_start)..(text_range.start + char_byte_end);

                    clusters.push((cluster_id, cluster_text_range, cluster_info, *style_index));
                }
            }
        }

        clusters
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
