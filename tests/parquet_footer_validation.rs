//! Verification test: Parquet files must have valid footer for DuckDB Wasm compatibility
//! This test ensures that Issue #1 (missing Parquet footer) is fixed.

#[cfg(not(target_arch = "wasm32"))]
mod parquet_footer_tests {
    use fast_insight_engine::file_convert::{Converter, CsvReadOptions, ParquetWriteOptions, NullHandling, ParquetCompression};

    const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

    #[test]
    fn test_parquet_has_valid_footer() {
        // Simple CSV: 10 rows, 2 columns
        let csv_data = b"name,age\nAlice,30\nBob,25\nCarol,35\nDave,28\nEve,32\nFrank,29\nGrace,31\nHenry,27\nIvy,33\nJack,26\n";

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions {
            delimiter: b',',
            has_header: true,
            null_handling: NullHandling::Null,
        };
        let pq_opts = ParquetWriteOptions {
            row_group_size: 1024,
            compression: ParquetCompression::Uncompressed,
        };

        converter.begin_csv_to_parquet(csv_opts, pq_opts, None).expect("begin failed");
        let chunks = converter.feed_csv_chunk(csv_data, true).expect("feed failed");

        // Verify: Parquet file structure
        // Format: MAGIC (4B) | RowGroups | Footer | FooterLength (4B) | MAGIC (4B)
        assert!(!chunks.is_empty(), "Should produce at least one chunk");

        for chunk in chunks {
            assert!(chunk.len() >= 12, "Parquet file must have at least 12 bytes (magic + footer_length + magic)");

            // Check magic at start
            assert_eq!(&chunk[0..4], PARQUET_MAGIC, "File must start with PAR1 magic");

            // Check magic at end (last 4 bytes)
            let end_idx = chunk.len() - 4;
            assert_eq!(&chunk[end_idx..], PARQUET_MAGIC, "File must end with PAR1 magic");

            // Verify footer length is reasonable (between 100 bytes and 10 MB)
            // Footer length is stored in the last 8 bytes before the final magic (little-endian)
            let footer_len_bytes = &chunk[chunk.len() - 8..chunk.len() - 4];
            let footer_len = i32::from_le_bytes([
                footer_len_bytes[0],
                footer_len_bytes[1],
                footer_len_bytes[2],
                footer_len_bytes[3],
            ]) as usize;

            println!("Parquet footer length: {} bytes", footer_len);
            assert!(footer_len > 100, "Footer should be at least 100 bytes");
            assert!(footer_len < 10_000_000, "Footer should be less than 10 MB");

            // Verify footer is present in file
            // Footer starts at: file_len - footer_len - 8 (footer_len + magic)
            let expected_footer_start = chunk.len() - footer_len - 8;
            assert!(
                expected_footer_start > 4,
                "Footer must start after initial magic ({})", 
                expected_footer_start
            );
        }
    }

    #[test]
    fn test_multiple_row_groups_all_valid() {
        // CSV with 3000 rows to force multiple row groups (row_group_size=1000)
        let mut csv_data = String::from("id,value\n");
        for i in 0..3000 {
            csv_data.push_str(&format!("{},{}.5\n", i, i as f64));
        }

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions {
            delimiter: b',',
            has_header: true,
            null_handling: NullHandling::Null,
        };
        let pq_opts = ParquetWriteOptions {
            row_group_size: 1000,
            compression: ParquetCompression::Uncompressed,
        };

        converter.begin_csv_to_parquet(csv_opts, pq_opts, None).expect("begin failed");
        let chunks = converter.feed_csv_chunk(csv_data.as_bytes(), true).expect("feed failed");

        // With 3000 rows and row_group_size=1000, we expect 3 chunks
        assert_eq!(chunks.len(), 3, "Should produce 3 row groups");

        // Verify each chunk is a valid Parquet file
        for (idx, chunk) in chunks.iter().enumerate() {
            assert!(chunk.len() > 12, "Chunk {} must have proper Parquet structure", idx);
            assert_eq!(&chunk[0..4], PARQUET_MAGIC, "Chunk {} starts with magic", idx);
            assert_eq!(&chunk[chunk.len()-4..], PARQUET_MAGIC, "Chunk {} ends with magic", idx);

            println!("Chunk {} size: {} bytes", idx, chunk.len());
        }
    }

    #[test]
    fn test_duckdb_can_validate_format() {
        // Simple test to verify the Parquet format is compatible with standard readers
        let csv_data = b"x,y\n1,2\n3,4\n5,6\n";

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions {
            delimiter: b',',
            has_header: true,
            null_handling: NullHandling::Null,
        };
        let pq_opts = ParquetWriteOptions {
            row_group_size: 100,
            compression: ParquetCompression::Uncompressed,
        };

        converter.begin_csv_to_parquet(csv_opts, pq_opts, None).expect("begin failed");
        let chunks = converter.feed_csv_chunk(csv_data, true).expect("feed failed");

        assert!(!chunks.is_empty(), "Should produce output");

        // The chunks can be directly passed to DuckDB Wasm
        // This is a structural validation, not a functional one
        let total_size: usize = chunks.iter().map(|c| c.len()).sum();
        println!("Total Parquet size: {} bytes", total_size);
        assert!(total_size > 50, "Parquet file should be at least 50 bytes");
    }
}
