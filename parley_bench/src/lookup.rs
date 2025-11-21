// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Helpers for comparing composite property lookups.

use std::sync::OnceLock;

use icu_provider::buf::AsDeserializingBufferProvider;
use icu_provider::{DataMarker, DataPayload, DataRequest, DataResponse, DynamicDataProvider};
use unicode_data::{CompositePropsV1, CompositePropsV1Data};

const SAMPLE_LEN: usize = 4096;

fn build_sample_codepoints() -> Vec<u32> {
    let mut values = Vec::with_capacity(SAMPLE_LEN);
    let mut state = 0x1234_5678_u32;
    for _ in 0..SAMPLE_LEN {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(state % 0x11_0000);
    }
    values
}

fn load_composite_payload() -> DataPayload<CompositePropsV1> {
    let provider = icu_provider_blob::BlobDataProvider::try_new_from_static_blob(
        unicode_data::generated::COMPOSITE_BLOB,
    )
    .expect("Composite blob should deserialize");

    let response: DataResponse<CompositePropsV1> = provider
        .as_deserializing()
        .load_data(CompositePropsV1::INFO, DataRequest::default())
        .expect("Composite data should load");

    response.payload
}

/// Returns the baked composite trie.
pub fn composite_data() -> &'static CompositePropsV1Data<'static> {
    static PAYLOAD: OnceLock<DataPayload<CompositePropsV1>> = OnceLock::new();
    PAYLOAD.get_or_init(load_composite_payload).get()
}

/// Returns a deterministic set of scalar values that cover all Unicode planes.
pub fn codepoint_samples() -> &'static [u32] {
    static SAMPLES: OnceLock<Vec<u32>> = OnceLock::new();
    SAMPLES.get_or_init(build_sample_codepoints)
}

/// Accumulates a checksum from the packed properties stored in the CodePointTrie.
pub fn checksum_trie(samples: &[u32], composite: &'static CompositePropsV1Data<'static>) -> u32 {
    samples
        .iter()
        .fold(0_u32, |acc, &cp| acc ^ u32::from(composite.properties(cp)))
}

/// Accumulates a checksum from the properties stored in the PackTab tables.
pub fn checksum_packtab(samples: &[u32]) -> u32 {
    samples.iter().fold(0_u32, |acc, &cp| {
        acc ^ u32::from(unicode_data::packtab_properties(cp))
    })
}
