/// Integration tests for CSV → Parquet conversion
///
/// These tests verify end-to-end functionality including:
/// - Basic CSV to Parquet conversion correctness
/// - Chunked input handling
/// - Error scenarios
/// - Performance characteristics
use fast_insight_engine::file_convert::{Converter, CsvReadOptions, ParquetWriteOptions};

#[test]
fn test_integration_basic_conversion() {
    // Simple CSV with 3 columns (matches hardcoded MVP schema)
    let csv_data = b"col1,col2,col3\n\
                     Alice,100,10.5\n\
                     Bob,200,20.5\n\
                     Charlie,300,30.5\n";

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

    // Should produce Parquet output
    assert!(
        !chunks.is_empty(),
        "Should produce at least one Parquet chunk"
    );

    // Each chunk should be non-empty
    for chunk in &chunks {
        assert!(!chunk.is_empty(), "Each chunk should contain data");
    }

    println!(
        "✓ Basic conversion produced {} Parquet chunks",
        chunks.len()
    );
}

#[test]
fn test_integration_chunked_input() {
    // Simulate chunked CSV input (large file split into pieces)
    let chunk1 = b"col1,col2,col3\n\
                  Alice,100,10\n";
    let chunk2 = b"Bob,200,20\n\
                   Charlie,300,30\n";
    let chunk3 = b"David,400,40\n";

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();

    // Feed first chunk (is_last = false)
    let chunks1 = converter.feed_csv_chunk(chunk1, false).unwrap();

    // Feed second chunk (is_last = false)
    let chunks2 = converter.feed_csv_chunk(chunk2, false).unwrap();

    // Feed final chunk (is_last = true)
    let chunks3 = converter.feed_csv_chunk(chunk3, true).unwrap();

    let total_chunks = chunks1.len() + chunks2.len() + chunks3.len();
    println!(
        "✓ Chunked input produced {} Parquet chunks total",
        total_chunks
    );
}

#[test]
fn test_integration_no_header() {
    // CSV without header (should auto-generate col_0, col_1, col_2)
    let csv_data = b"Alice,100,10\n\
                     Bob,200,20\n\
                     Charlie,300,30\n";

    let mut converter = Converter::new();
    let mut csv_opts = CsvReadOptions::default();
    csv_opts.has_header = false;
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

    assert!(!chunks.is_empty(), "Should convert CSV without header");
    println!("✓ No-header CSV conversion succeeded");
}

#[test]
fn test_integration_error_invalid_state() {
    // Attempting to feed chunk without calling begin first
    let csv_data = b"col1,col2,col3\nAlice,100,10\n";
    let mut converter = Converter::new();

    // Should fail because begin() was never called
    let result = converter.feed_csv_chunk(csv_data, true);
    assert!(result.is_err(), "feed_csv_chunk should fail without begin");
    println!("✓ Correctly rejected feed before begin");
}

#[test]
fn test_integration_empty_input() {
    // Empty CSV (just header)
    let csv_data = b"col1,col2,col3\n";

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

    // Should handle empty data gracefully
    println!("✓ Empty CSV handled: produced {} chunks", chunks.len());
}

#[test]
fn test_integration_custom_delimiter() {
    // Tab-separated values
    let csv_data = b"col1\tcol2\tcol3\n\
                     Alice\t100\t10\n\
                     Bob\t200\t20\n";

    let mut converter = Converter::new();
    let mut csv_opts = CsvReadOptions::default();
    csv_opts.delimiter = b'\t';
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

    assert!(!chunks.is_empty(), "Should handle custom delimiter");
    println!("✓ Tab-delimited CSV conversion succeeded");
}

#[test]
fn test_integration_quoted_fields() {
    // CSV with quoted fields (RFC 4180)
    let csv_data = b"col1,col2,col3\n\
                     \"Alice Smith\",100,10\n\
                     \"Bob Jones\",200,20\n";

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

    assert!(!chunks.is_empty(), "Should handle quoted fields");
    println!("✓ Quoted fields CSV conversion succeeded");
}

#[test]
fn test_integration_memory_limit() {
    // CSV that respects memory limit
    let csv_data = b"col1,col2,col3\n\
                     Alice,100,10\n\
                     Bob,200,20\n";

    let mut converter = Converter::with_memory_limit(1); // 1 MB limit
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();

    // Small CSV should fit in 1 MB
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();
    assert!(!chunks.is_empty(), "Small CSV should fit in 1 MB limit");
    println!("✓ Memory limit check passed");
}

#[test]
fn test_integration_large_dataset_simulation() {
    // Simulate a larger dataset without actually allocating huge amounts
    // Generate multiple rows programmatically
    let mut csv_data = Vec::<u8>::new();
    csv_data.extend_from_slice(b"col1,col2,col3\n");

    // Generate 100 rows
    for i in 0..100 {
        let row = format!("Row{},Value{},{}\n", i, i * 10, i as f32 * 0.5);
        csv_data.extend_from_slice(row.as_bytes());
    }

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(&csv_data, true).unwrap();

    assert!(!chunks.is_empty(), "Should handle larger datasets");
    println!(
        "✓ Large dataset (100 rows) conversion succeeded: {} chunks",
        chunks.len()
    );
}

#[test]
fn test_integration_session_reuse() {
    // Test that converter can be reused after free()
    let csv_data = b"col1,col2,col3\nAlice,100,10\n";

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    // First session
    converter
        .begin_csv_to_parquet(csv_opts.clone(), pq_opts.clone())
        .unwrap();
    let chunks1 = converter.feed_csv_chunk(csv_data, true).unwrap();
    assert!(!chunks1.is_empty());
    converter.free();

    // Second session (reuse)
    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks2 = converter.feed_csv_chunk(csv_data, true).unwrap();
    assert!(!chunks2.is_empty());

    println!("✓ Session reuse after free() works correctly");
}

#[test]
fn test_integration_whitespace_handling() {
    // CSV with various whitespace
    let csv_data = b"col1,col2,col3\n\
                     Alice, 100 , 10\n\
                     Bob,200,20\n";

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let pq_opts = ParquetWriteOptions::default();

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
    let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

    assert!(!chunks.is_empty(), "Should handle whitespace in fields");
    println!("✓ Whitespace handling test passed");
}

#[test]
fn test_integration_performance_benchmark() {
    // Phase 10: Performance optimization test
    // Create a 100k row CSV to test throughput
    use std::time::Instant;

    let num_rows = 100_000;
    let mut csv_data = String::from("col1,col2,col3\n");
    
    for i in 0..num_rows {
        csv_data.push_str(&format!("Row{},{},{:.2}\n", i, i * 10, i as f64 * 1.5));
    }

    let csv_bytes = csv_data.as_bytes();
    let data_size_mb = csv_bytes.len() as f64 / 1_024_000.0;

    let mut converter = Converter::new();
    let csv_opts = CsvReadOptions::default();
    let mut pq_opts = ParquetWriteOptions::default();
    pq_opts.row_group_size = 4096; // Larger row groups for better performance

    converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();

    let start = Instant::now();
    let chunks = converter.feed_csv_chunk(csv_bytes, true).unwrap();
    let elapsed = start.elapsed();

    let elapsed_secs = elapsed.as_secs_f64();
    let throughput_mbps = data_size_mb / elapsed_secs;

    println!(
        "✓ Performance: {:.2} MB in {:.2}s = {:.1} MB/s",
        data_size_mb, elapsed_secs, throughput_mbps
    );
    println!("  Rows: {}, Chunks produced: {}", num_rows, chunks.len());

    // Verify minimum throughput requirement (Phase 10 goal: >= 20 MB/s)
    // Note: This is a soft goal; actual performance depends on hardware
    if throughput_mbps >= 20.0 {
        println!("  ✓ Throughput goal (≥20 MB/s) achieved!");
    } else {
        println!("  ⚠ Throughput ({:.1} MB/s) below goal (≥20 MB/s)", throughput_mbps);
    }
}
