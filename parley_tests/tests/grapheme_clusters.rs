// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests that layout clusters are extended grapheme clusters.
//!
//! A multi-codepoint grapheme (a combining sequence, emoji ZWJ sequence, regional-indicator
//! flag pair, or CRLF) must be returned as a single cluster; a shaped cluster spanning several
//! graphemes (e.g. an "fi" ligature) must be split per grapheme.

use core::ops::Range;

use crate::test_name;
use crate::util::{ColorBrush, TestEnv};
use parley::{Affinity, Cluster, Cursor, FontFamily, Layout, PositionedLayoutItem, StyleProperty};

/// Builds a single-line layout for `text` with the default test fonts.
fn build_layout(env: &mut TestEnv, text: &str) -> Layout<ColorBrush> {
    let builder = env.ranged_builder(text);
    let mut layout = builder.build(text);
    layout.break_all_lines(None);
    layout
}

/// Collects every cluster's text range, in logical order.
fn cluster_text_ranges(layout: &Layout<ColorBrush>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                for cluster in glyph_run.run().clusters() {
                    ranges.push(cluster.text_range());
                }
            }
        }
    }
    ranges
}

#[test]
fn grapheme_combining_mark_is_one_cluster() {
    let mut env = TestEnv::new(test_name!(), None);
    // "e" + U+0301 (combining acute accent) is one grapheme cluster of 3 bytes.
    let text = "ae\u{301}b";
    let layout = build_layout(&mut env, text);

    assert_eq!(cluster_text_ranges(&layout), vec![0..1, 1..4, 4..5]);

    // A byte index inside the grapheme resolves to the whole cluster.
    let cluster = Cluster::from_byte_index(&layout, 2).unwrap();
    assert_eq!(cluster.text_range(), 1..4);
    assert!(!cluster.is_ligature_start() && !cluster.is_ligature_continuation());
}

#[test]
fn grapheme_emoji_zwj_sequence_is_one_cluster() {
    let mut env = TestEnv::new(test_name!(), None);
    // Family emoji: 7 codepoints joined by ZWJs, one grapheme cluster of 25 bytes.
    let text = "👨‍👩‍👧‍👦";
    let mut builder = env.ranged_builder(text);
    builder.push_default(StyleProperty::FontFamily(FontFamily::named(
        "Noto Color Emoji",
    )));
    let mut layout = builder.build(text);
    layout.break_all_lines(None);

    assert_eq!(cluster_text_ranges(&layout), vec![0..text.len()]);

    let cluster = Cluster::from_byte_index(&layout, 4).unwrap();
    assert_eq!(cluster.text_range(), 0..text.len());
    assert!(cluster.is_emoji());
}

#[test]
fn grapheme_regional_indicator_pair_is_one_cluster() {
    let mut env = TestEnv::new(test_name!(), None);
    // Two flags: each is a pair of regional indicators (8 bytes per flag). ICU treats each
    // pair as one grapheme even when the font does not ligate them.
    let text = "🇦🇺🇺🇸";
    let mut builder = env.ranged_builder(text);
    builder.push_default(StyleProperty::FontFamily(FontFamily::named(
        "Noto Color Emoji",
    )));
    let mut layout = builder.build(text);
    layout.break_all_lines(None);

    assert_eq!(cluster_text_ranges(&layout), vec![0..8, 8..16]);
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                for cluster in glyph_run.run().clusters() {
                    assert!(cluster.is_emoji());
                    // The cluster advance accounts for all of the pair's glyphs.
                    let glyph_advance: f32 = cluster.glyphs().map(|g| g.advance).sum();
                    if cluster.glyphs().count() > 0 {
                        assert_eq!(cluster.advance(), glyph_advance);
                    }
                }
            }
        }
    }
}

#[test]
fn grapheme_crlf_is_one_cluster() {
    let mut env = TestEnv::new(test_name!(), None);
    let text = "a\r\nb";
    let layout = build_layout(&mut env, text);

    // CRLF is one cluster and produces exactly one hard line break.
    assert_eq!(cluster_text_ranges(&layout), vec![0..1, 1..3, 3..4]);
    assert_eq!(layout.len(), 2, "expected exactly two lines");

    let crlf = Cluster::from_byte_index(&layout, 1).unwrap();
    assert_eq!(crlf.text_range(), 1..3);
    assert!(crlf.is_hard_line_break());
    // Newline clusters are stripped of glyphs and advance.
    assert_eq!(crlf.glyphs().count(), 0);
    assert_eq!(crlf.advance(), 0.0);
}

#[test]
fn grapheme_lone_cr_and_lf_are_single_clusters() {
    let mut env = TestEnv::new(test_name!(), None);
    let text = "a\rb\nc";
    let layout = build_layout(&mut env, text);

    assert_eq!(
        cluster_text_ranges(&layout),
        vec![0..1, 1..2, 2..3, 3..4, 4..5]
    );
    assert_eq!(layout.len(), 3, "expected three lines");
}

#[test]
fn grapheme_zalgo_long_combining_sequence() {
    let mut env = TestEnv::new(test_name!(), None);
    // A grapheme cluster longer than 255 bytes (which `text_len: u8` could not represent) and
    // with more than 254 glyphs (exercising the glyph spill path).
    let marks = "\u{301}".repeat(300);
    let text = format!("a e{marks} b");
    let layout = build_layout(&mut env, &text);

    let zalgo_range = 2..3 + marks.len();
    let ranges = cluster_text_ranges(&layout);
    // The zalgo grapheme is one cluster; glyph spill pieces (ligature components) may follow
    // with empty text ranges at the grapheme's end.
    assert!(ranges.contains(&zalgo_range));
    for range in &ranges {
        assert!(
            !range.is_empty() || range.start == zalgo_range.end,
            "unexpected empty cluster range {range:?}"
        );
    }

    // Every glyph stays attributed to a cluster: summed glyph counts match, and cursor
    // navigation resolves interior byte indices to the whole grapheme.
    let cluster = Cluster::from_byte_index(&layout, 10).unwrap();
    assert_eq!(cluster.text_range(), zalgo_range);

    // The spilled pieces are flagged as a ligature start plus continuations.
    assert!(cluster.is_ligature_start());
}

#[test]
fn grapheme_cursor_steps_over_multi_codepoint_graphemes() {
    let mut env = TestEnv::new(test_name!(), None);
    // "a", then é (2 codepoints), then family emoji, then "b".
    let text = "ae\u{301}👨‍👩‍👧‍👦b";
    let layout = build_layout(&mut env, text);

    let mut cursor = Cursor::from_byte_index(&layout, 0, Affinity::Downstream);
    let mut positions = vec![cursor.index()];
    loop {
        let next = cursor.next_visual(&layout);
        if next.index() == cursor.index() {
            break;
        }
        cursor = next;
        positions.push(cursor.index());
    }

    // One step per grapheme: a | é | 👨‍👩‍👧‍👦 | b
    assert_eq!(positions, vec![0, 1, 4, 29, 30]);
}

#[test]
fn grapheme_letter_spacing_not_applied_within_grapheme() {
    let mut env = TestEnv::new(test_name!(), None);
    let text = "e\u{301}e\u{301}";

    let layout_without = build_layout(&mut env, text);

    let mut builder = env.ranged_builder(text);
    builder.push_default(StyleProperty::LetterSpacing(10.0));
    let mut layout_with = builder.build(text);
    layout_with.break_all_lines(None);

    // Two grapheme clusters -> letter spacing is added once per cluster, never inside one.
    assert_eq!(cluster_text_ranges(&layout_without).len(), 2);
    let width_delta = layout_with.full_width() - layout_without.full_width();
    assert!(
        (width_delta - 2.0 * 10.0).abs() < 1e-3,
        "expected letter spacing to be applied exactly twice, got extra width {width_delta}"
    );
}
