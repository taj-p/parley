// Copyright 2021 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text shaping implementation using HarfBuzz (via harfrust) for shaping
//! and swash for text analysis and font selection.

use alloc::vec::Vec;

// Parley imports
use super::layout::Layout;
use super::resolve::{RangedStyle, ResolveContext, Resolved};
use super::style::{Brush, FontFeature, FontVariation};
use crate::Font;
use crate::inline_box::InlineBox;
use crate::layout::data::HarfSynthesis;
use crate::util::nearly_eq;

// External crate imports
use fontique::{self, Query, QueryFamily, QueryFont};
use harfrust;
use swash::text::cluster::{CharCluster, CharInfo, Token};
use swash::text::{Language, Script};
use swash::{FontRef, Synthesis};

/// Capacity hint for deferred inline boxes to avoid repeated allocations
const DEFERRED_BOXES_CAPACITY: usize = 16;

/// Convert swash synthesis information to our HarfSynthesis format.
/// This extracts the bold and italic adjustments for use with harfrust.
///
/// TODO: This conversion is lossy and discards important synthesis information:
/// - Variation settings (weight/width/slant axes) are lost
/// - Precise skew angle is reduced to boolean (any non-zero skew becomes true)
/// - Small caps and other synthesis options are ignored
///
/// For full fidelity, HarfSynthesis should be expanded to preserve all
/// swash::Synthesis fields, or we should pass the original synthesis through
/// and convert at render time instead of shaping time.
fn synthesis_to_harf_simple(synthesis: Synthesis) -> HarfSynthesis {
    HarfSynthesis {
        bold: synthesis.embolden(),
        italic: synthesis.skew().unwrap_or(0.0) != 0.0,
    }
}

/// Convert a swash Tag (u32) to a harfrust Tag for OpenType feature/script handling.
fn convert_swash_tag_to_harfrust(swash_tag: u32) -> harfrust::Tag {
    harfrust::Tag::from_be_bytes(swash_tag.to_be_bytes())
}

/// Convert swash Script enum to harfrust Script for proper text shaping.
/// Maps Unicode script codes to their corresponding OpenType script tags.
fn convert_script_to_harfrust(swash_script: Script) -> harfrust::Script {
    let tag = match swash_script {
        Script::Arabic => harfrust::script::ARABIC,
        Script::Latin => harfrust::script::LATIN,
        Script::Common => harfrust::script::COMMON,
        Script::Unknown => harfrust::script::UNKNOWN,
        Script::Inherited => harfrust::script::INHERITED,
        Script::Cyrillic => harfrust::script::CYRILLIC,
        Script::Greek => harfrust::script::GREEK,
        Script::Hebrew => harfrust::script::HEBREW,
        Script::Han => harfrust::script::HAN,
        Script::Hiragana => harfrust::script::HIRAGANA,
        Script::Katakana => harfrust::script::KATAKANA,
        Script::Devanagari => harfrust::script::DEVANAGARI,
        Script::Thai => harfrust::script::THAI,
        // For unmapped scripts, default to Latin
        _ => todo!("Unmapped script: {:?}", swash_script),
    };

    tag
}

struct Item {
    style_index: u16,
    size: f32,
    script: Script,
    level: u8,
    locale: Option<Language>,
    variations: Resolved<FontVariation>,
    features: Resolved<FontFeature>,
    word_spacing: f32,
    letter_spacing: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn shape_text<'a, B: Brush>(
    rcx: &'a ResolveContext,
    mut fq: Query<'a>,
    styles: &'a [RangedStyle<B>],
    inline_boxes: &[InlineBox],
    infos: &[(CharInfo, u16)],
    levels: &[u8],
    _scx: &mut swash::shape::ShapeContext, // Not used in harfrust approach
    mut text: &str,
    layout: &mut Layout<B>,
) {
    // If we have both empty text and no inline boxes, shape with a fake space
    // to generate metrics that can be used to size a cursor.
    if text.is_empty() && inline_boxes.is_empty() {
        text = " ";
    }
    // Do nothing if there is no text or styles (there should always be a default style)
    if text.is_empty() || styles.is_empty() {
        // Process any remaining inline boxes whose index is greater than the length of the text
        for box_idx in 0..inline_boxes.len() {
            // Push the box to the list of items
            layout.data.push_inline_box(box_idx);
        }
        return;
    }

    // Setup mutable state for iteration
    let mut style = &styles[0].style;
    let mut item = Item {
        style_index: 0,
        size: style.font_size,
        level: levels.first().copied().unwrap_or(0),
        script: infos
            .iter()
            .map(|x| x.0.script())
            .find(|&script| real_script(script))
            .unwrap_or(Script::Latin),
        locale: style.locale,
        variations: style.font_variations,
        features: style.font_features,
        word_spacing: style.word_spacing,
        letter_spacing: style.letter_spacing,
    };
    let mut char_range = 0..0;
    let mut text_range = 0..0;

    let mut inline_box_iter = inline_boxes.iter().enumerate();
    let mut current_box = inline_box_iter.next();
    let mut deferred_boxes: Vec<usize> = Vec::with_capacity(DEFERRED_BOXES_CAPACITY);

    // Iterate over characters in the text (same as original)
    for ((char_index, (byte_index, ch)), (info, style_index)) in
        text.char_indices().enumerate().zip(infos)
    {
        let mut break_run = false;
        let mut script = info.script();
        if !real_script(script) {
            script = item.script;
        }
        let level = levels.get(char_index).copied().unwrap_or(0);
        if item.style_index != *style_index {
            item.style_index = *style_index;
            style = &styles[*style_index as usize].style;
            if !nearly_eq(style.font_size, item.size)
                || style.locale != item.locale
                || style.font_variations != item.variations
                || style.font_features != item.features
                || !nearly_eq(style.letter_spacing, item.letter_spacing)
                || !nearly_eq(style.word_spacing, item.word_spacing)
            {
                break_run = true;
            }
        }

        if level != item.level || script != item.script {
            break_run = true;
        }

        // Check if there is an inline box at this index
        while let Some((box_idx, inline_box)) = current_box {
            if inline_box.index == byte_index {
                break_run = true;
                deferred_boxes.push(box_idx);
                current_box = inline_box_iter.next();
            } else {
                break;
            }
        }

        if break_run && !text_range.is_empty() {
            shape_item(
                &mut fq,
                rcx,
                styles,
                &item,
                text,
                &text_range,
                &char_range,
                infos,
                layout,
            );
            item.size = style.font_size;
            item.level = level;
            item.script = script;
            item.locale = style.locale;
            item.variations = style.font_variations;
            item.features = style.font_features;
            text_range.start = text_range.end;
            char_range.start = char_range.end;
        }

        for box_idx in deferred_boxes.drain(0..) {
            layout.data.push_inline_box(box_idx);
        }

        text_range.end += ch.len_utf8();
        char_range.end += 1;
    }

    if !text_range.is_empty() {
        shape_item(
            &mut fq,
            rcx,
            styles,
            &item,
            text,
            &text_range,
            &char_range,
            infos,
            layout,
        );
    }

    // Process any remaining inline boxes
    if let Some((box_idx, _inline_box)) = current_box {
        layout.data.push_inline_box(box_idx);
    }
    for (box_idx, _inline_box) in inline_box_iter {
        layout.data.push_inline_box(box_idx);
    }
}

fn shape_item<'a, B: Brush>(
    fq: &mut Query<'a>,
    rcx: &'a ResolveContext,
    styles: &'a [RangedStyle<B>],
    item: &Item,
    text: &str,
    text_range: &std::ops::Range<usize>,
    char_range: &std::ops::Range<usize>,
    infos: &[(CharInfo, u16)],
    layout: &mut Layout<B>,
) {
    let item_text = &text[text_range.clone()];
    let item_infos = &infos[char_range.start..char_range.end]; // Only process current item
    let first_style_index = item_infos[0].1;
    let mut font_selector =
        FontSelector::new(fq, rcx, styles, first_style_index, item.script, item.locale);

    // Parse text into clusters (exactly like swash does) - but only for current item
    let tokens =
        item_text
            .char_indices()
            .zip(item_infos)
            .map(|((offset, ch), (info, style_index))| Token {
                ch,
                offset: (text_range.start + offset) as u32,
                len: ch.len_utf8() as u8,
                info: *info,
                data: *style_index as u32,
            });

    let mut parser = swash::text::cluster::Parser::new(item.script, tokens);
    let mut cluster = CharCluster::new();

    // Reimplement swash's shape_clusters algorithm - but only for current item
    if !parser.next(&mut cluster) {
        return; // No clusters to process
    }

    let mut current_font = font_selector.select_font(&mut cluster);

    // Main segmentation loop (based on swash shape_clusters) - only within current item
    // TODO: Replace with ICU4X
    while let Some(font) = current_font.take() {
        // Collect all clusters for this font segment
        let mut segment_clusters = vec![cluster.clone()];
        let segment_start_offset = cluster.range().start as usize - text_range.start;
        let mut segment_end_offset = cluster.range().end as usize - text_range.start;

        loop {
            if !parser.next(&mut cluster) {
                // End of current item - process final segment
                break;
            }

            if let Some(next_font) = font_selector.select_font(&mut cluster) {
                if next_font != font {
                    current_font = Some(next_font);
                    break;
                } else {
                    // Same font - add to current segment
                    segment_clusters.push(cluster.clone());
                    segment_end_offset = cluster.range().end as usize - text_range.start;
                }
            } else {
                // No font found - skip this cluster
                if !parser.next(&mut cluster) {
                    break;
                }
            }
        }

        // Shape this font segment with harfrust
        let segment_text = &item_text[segment_start_offset..segment_end_offset];
        // Shape the entire segment text including newlines
        // The line breaking algorithm will handle newlines automatically

        let harf_font =
            harfrust::FontRef::from_index(font.font.blob.as_ref(), font.font.index).unwrap(); // TODO: Propagate error

        // Create harfrust shaper
        // TODO: cache this upstream?
        let shaper_data = harfrust::ShaperData::new(&harf_font);
        let mut variations: Vec<harfrust::Variation> = vec![];

        // Extract variations from swash synthesis
        for setting in font.synthesis.variations() {
            variations.push(harfrust::Variation {
                tag: convert_swash_tag_to_harfrust(setting.tag),
                value: setting.value,
            });
        }

        let instance = harfrust::ShaperInstance::from_variations(&harf_font, &variations);
        // TODO: Don't create a new shaper for each segment.
        let harf_shaper = shaper_data
            .shaper(&harf_font)
            .instance(Some(&instance))
            .point_size(Some(item.size))
            .build();

        // Prepare harfrust buffer
        // TODO: Reuse this buffer for all segments.
        let mut buffer = harfrust::UnicodeBuffer::new();

        // Use the entire segment text including newlines
        buffer.push_str(segment_text);

        let direction = if item.level & 1 != 0 {
            harfrust::Direction::RightToLeft
        } else {
            harfrust::Direction::LeftToRight
        };
        buffer.set_direction(direction);

        let script = convert_script_to_harfrust(item.script);
        buffer.set_script(script);

        if let Some(lang) = item.locale {
            let lang_tag = lang.language();
            if let Ok(harf_lang) = lang_tag.parse::<harfrust::Language>() {
                buffer.set_language(harf_lang);
            }
        }

        let glyph_buffer = harf_shaper.shape(buffer, &[]);

        // Extract relevant CharInfo slice for this segment
        let char_start = char_range.start + item_text[..segment_start_offset].chars().count();
        let segment_char_start = char_start - char_range.start;
        let segment_char_count = segment_text.chars().count();
        let segment_infos =
            &item_infos[segment_char_start..(segment_char_start + segment_char_count)];

        // Push harfrust-shaped run for the entire segment
        layout.data.push_run_from_harfrust(
            Font::new(font.font.blob.clone(), font.font.index),
            item.size,
            synthesis_to_harf_simple(font.synthesis),
            font.font.synthesis, // Use the original fontique synthesis from QueryFont
            &glyph_buffer,
            item.level,
            item.word_spacing,
            item.letter_spacing,
            segment_text,
            segment_infos,
            (text_range.start + segment_start_offset)..(text_range.start + segment_end_offset),
            &variations,
        );
    }
}

fn real_script(script: Script) -> bool {
    script != Script::Common && script != Script::Unknown && script != Script::Inherited
}

struct FontSelector<'a, 'b, B: Brush> {
    query: &'b mut Query<'a>,
    fonts_id: Option<usize>,
    rcx: &'a ResolveContext,
    styles: &'a [RangedStyle<B>],
    style_index: u16,
    attrs: fontique::Attributes,
    variations: &'a [FontVariation],
    features: &'a [FontFeature],
}

impl<'a, 'b, B: Brush> FontSelector<'a, 'b, B> {
    fn new(
        query: &'b mut Query<'a>,
        rcx: &'a ResolveContext,
        styles: &'a [RangedStyle<B>],
        style_index: u16,
        script: Script,
        locale: Option<Language>,
    ) -> Self {
        let style = &styles[style_index as usize].style;
        let fonts_id = style.font_stack.id();
        let fonts = rcx.stack(style.font_stack).unwrap_or(&[]);
        let attrs = fontique::Attributes {
            width: style.font_width,
            weight: style.font_weight,
            style: style.font_style,
        };
        let variations = rcx.variations(style.font_variations).unwrap_or(&[]);
        let features = rcx.features(style.font_features).unwrap_or(&[]);
        query.set_families(fonts.iter().copied());

        let fb_script = crate::swash_convert::script_to_fontique(script);
        let fb_language = locale.and_then(crate::swash_convert::locale_to_fontique);
        query.set_fallbacks(fontique::FallbackKey::new(fb_script, fb_language.as_ref()));
        query.set_attributes(attrs);

        Self {
            query,
            fonts_id: Some(fonts_id),
            rcx,
            styles,
            style_index,
            attrs,
            variations,
            features,
        }
    }

    fn select_font(&mut self, cluster: &mut CharCluster) -> Option<SelectedFont> {
        let style_index = cluster.user_data() as u16;
        let is_emoji = cluster.info().is_emoji();
        if style_index != self.style_index || is_emoji || self.fonts_id.is_none() {
            self.style_index = style_index;
            let style = &self.styles[style_index as usize].style;

            let fonts_id = style.font_stack.id();
            let fonts = self.rcx.stack(style.font_stack).unwrap_or(&[]);
            let fonts = fonts.iter().copied().map(QueryFamily::Id);
            if is_emoji {
                use core::iter::once;
                let emoji_family = QueryFamily::Generic(fontique::GenericFamily::Emoji);
                self.query.set_families(fonts.chain(once(emoji_family)));
                self.fonts_id = None;
            } else if self.fonts_id != Some(fonts_id) {
                self.query.set_families(fonts);
                self.fonts_id = Some(fonts_id);
            }

            let attrs = fontique::Attributes {
                width: style.font_width,
                weight: style.font_weight,
                style: style.font_style,
            };
            if self.attrs != attrs {
                self.query.set_attributes(attrs);
                self.attrs = attrs;
            }
            self.variations = self.rcx.variations(style.font_variations).unwrap_or(&[]);
            self.features = self.rcx.features(style.font_features).unwrap_or(&[]);
        }
        let mut selected_font = None;
        self.query.matches_with(|font| {
            use skrifa::MetadataProvider;
            use swash::text::cluster::Status as MapStatus;

            let Ok(font_ref) = skrifa::FontRef::from_index(font.blob.as_ref(), font.index) else {
                return fontique::QueryStatus::Continue;
            };

            let charmap = font_ref.charmap();
            let map_status = cluster.map(|ch| {
                charmap
                    .map(ch)
                    .map(|g| {
                        g.to_u32()
                            .try_into()
                            .expect("Swash requires u16 glyph, so we hope that the glyph fits")
                    })
                    .unwrap_or_default()
            });

            match map_status {
                MapStatus::Complete => {
                    selected_font = Some(font.into());
                    fontique::QueryStatus::Stop
                }
                MapStatus::Keep => {
                    selected_font = Some(font.into());
                    fontique::QueryStatus::Continue
                }
                MapStatus::Discard => {
                    if selected_font.is_none() {
                        selected_font = Some(font.into());
                    }
                    fontique::QueryStatus::Continue
                }
            }
        });
        selected_font
    }
}

struct SelectedFont {
    font: QueryFont,
    synthesis: Synthesis,
}

impl From<&QueryFont> for SelectedFont {
    fn from(font: &QueryFont) -> Self {
        use crate::swash_convert::synthesis_to_swash;
        Self {
            font: font.clone(),
            synthesis: synthesis_to_swash(font.synthesis),
        }
    }
}

impl PartialEq for SelectedFont {
    fn eq(&self, other: &Self) -> bool {
        self.font.family == other.font.family && self.synthesis == other.synthesis
    }
}

impl swash::shape::partition::SelectedFont for SelectedFont {
    fn font(&self) -> FontRef<'_> {
        FontRef::from_index(self.font.blob.as_ref(), self.font.index as _).unwrap()
    }

    fn id_override(&self) -> Option<[u64; 2]> {
        Some([self.font.blob.id(), self.font.index as _])
    }

    fn synthesis(&self) -> Option<Synthesis> {
        Some(self.synthesis)
    }
}
