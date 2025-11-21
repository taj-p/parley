// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binary to exercise the CodePointTrie composite lookup for size comparisons.

use parley_bench::lookup;

fn main() {
    let samples = lookup::codepoint_samples();
    let composite = lookup::composite_data();
    let checksum = lookup::checksum_trie(samples, composite);
    println!("Composite trie checksum: {checksum}");
}
