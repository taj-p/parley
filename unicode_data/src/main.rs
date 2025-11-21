// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility binary that dumps the composite property trie into a JSON array.

#[cfg(feature = "baked")]
use std::env;
#[cfg(feature = "baked")]
use std::fs::File;
#[cfg(feature = "baked")]
use std::io::{BufWriter, Write};
#[cfg(feature = "baked")]
use std::path::PathBuf;

#[cfg(feature = "baked")]
use icu_provider::buf::AsDeserializingBufferProvider;
#[cfg(feature = "baked")]
use icu_provider::{DataMarker, DataRequest, DataResponse, DynamicDataProvider};
#[cfg(feature = "baked")]
use unicode_data::{CompositePropsV1, CompositePropsV1Data};

#[cfg(feature = "baked")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DEFAULT_OUTPUT: &str = "composite_props.json";
    const MAX_UNICODE_SCALAR: u32 = 0x10_FFFF;

    let output_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT));

    let provider = icu_provider_blob::BlobDataProvider::try_new_from_static_blob(
        unicode_data::generated::COMPOSITE_BLOB,
    )?;

    let response: DataResponse<CompositePropsV1> = provider
        .as_deserializing()
        .load_data(CompositePropsV1::INFO, DataRequest::default())?;
    let composite: &CompositePropsV1Data<'_> = response.payload.get();

    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(b"[")?;
    for cp in 0_u32..=MAX_UNICODE_SCALAR {
        let value: u32 = composite.properties(cp).into();
        if cp != 0 {
            writer.write_all(b",")?;
        }
        write!(writer, "{value}")?;
    }
    writer.write_all(b"]\n")?;

    println!(
        "Wrote {} composite property values to {}",
        (MAX_UNICODE_SCALAR + 1),
        output_path.display()
    );

    Ok(())
}

#[cfg(not(feature = "baked"))]
fn main() {
    eprintln!(
        "The unicode_data binary requires the `baked` feature. \
         Rebuild with `--features baked` (enabled by default)."
    );
    std::process::exit(1);
}
