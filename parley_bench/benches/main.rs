// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Parley benchmarks.

use tango_bench::{tango_benchmarks, tango_main};

use parley_bench::benches::{composite_lookup_latency, defaults, styled};

tango_benchmarks!(defaults(), styled(), composite_lookup_latency());
tango_main!();
