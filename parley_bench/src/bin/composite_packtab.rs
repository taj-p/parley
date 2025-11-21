// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Binary to exercise the PackTab composite lookup for size comparisons.

use parley_bench::lookup;

fn main() {
    let samples = lookup::codepoint_samples();
    let checksum = lookup::checksum_packtab(samples);
    println!("PackTab checksum: {checksum}");
}
