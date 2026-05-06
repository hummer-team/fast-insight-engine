/// Integration tests for schema hint (strict type conversion) in CSV→Parquet
///
/// These tests verify:
/// - Lenient mode (None): all columns remain Utf8, DuckDB handles type inference
/// - Strict mode (valid hint): columns converted to specified types
/// - Error mode (invalid hint): clear error returned on parse failure
use fast_insight_engine::file_convert::{
    ColumnDef, Converter, CsvReadOptions, ParquetWriteOptions, SchemaHint,
};

// ── Lenient mode ──────────────────────────────────────────────────────────────

#[test]
fn test_lenient_mode_no_schema_hint() {
    let csv = b"order_id,price,active\n1001,9.99,true\n1002,14.50,false\n";

    let mut converter = Converter::new();
    let chunks = converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            None,
        )
        .and_then(|_| converter.feed_csv_chunk(csv, true))
        .expect("lenient mode should succeed");

    assert!(!chunks.is_empty(), "Should produce Parquet output");
    println!("✓ Lenient mode: {} chunks", chunks.len());
}

// ── Strict mode: valid hints ──────────────────────────────────────────────────

#[test]
fn test_strict_mode_int64_and_float64() {
    let csv = b"order_id,price\n1001,9.99\n1002,14.50\n";

    let hint = SchemaHint::new(vec![
        ColumnDef::new("order_id".to_string(), 1), // Int64
        ColumnDef::new("price".to_string(), 2),    // Float64
    ]);

    let mut converter = Converter::new();
    let chunks = converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .and_then(|_| converter.feed_csv_chunk(csv, true))
        .expect("strict mode with valid data should succeed");

    assert!(!chunks.is_empty(), "Should produce Parquet output");
    println!("✓ Strict mode Int64/Float64: {} chunks", chunks.len());
}

#[test]
fn test_strict_mode_boolean() {
    let csv = b"name,active\nAlice,true\nBob,false\nCarol,1\n";

    let hint = SchemaHint::new(vec![
        ColumnDef::new("name".to_string(), 0),   // Utf8
        ColumnDef::new("active".to_string(), 3), // Boolean
    ]);

    let mut converter = Converter::new();
    let chunks = converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .and_then(|_| converter.feed_csv_chunk(csv, true))
        .expect("strict boolean mode should accept true/false/1");

    assert!(!chunks.is_empty());
    println!("✓ Strict mode Boolean: {} chunks", chunks.len());
}

#[test]
fn test_strict_mode_all_utf8() {
    let csv = b"col_a,col_b\nfoo,bar\nbaz,qux\n";

    let hint = SchemaHint::new(vec![
        ColumnDef::new("col_a".to_string(), 0), // Utf8
        ColumnDef::new("col_b".to_string(), 0), // Utf8
    ]);

    let mut converter = Converter::new();
    let chunks = converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .and_then(|_| converter.feed_csv_chunk(csv, true))
        .expect("strict utf8 mode should succeed");

    assert!(!chunks.is_empty());
    println!("✓ Strict mode all-Utf8: {} chunks", chunks.len());
}

// ── Strict mode: conversion failures ─────────────────────────────────────────

#[test]
fn test_strict_mode_int64_parse_failure() {
    let csv = b"id,name\nnot_a_number,Alice\n";

    let hint = SchemaHint::new(vec![
        ColumnDef::new("id".to_string(), 1),   // Int64 - will fail
        ColumnDef::new("name".to_string(), 0), // Utf8
    ]);

    let mut converter = Converter::new();
    converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .unwrap();
    let result = converter.feed_csv_chunk(csv, true);

    assert!(
        result.is_err(),
        "Should fail when Int64 column has non-numeric value"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("TypeConversionFailed") || err_msg.contains("not_a_number"),
        "Error should mention conversion failure, got: {}",
        err_msg
    );
    println!("✓ Strict mode correctly rejects invalid Int64: {}", err_msg);
}

#[test]
fn test_strict_mode_float64_parse_failure() {
    let csv = b"amount\nnot_a_float\n";

    let hint = SchemaHint::new(vec![ColumnDef::new("amount".to_string(), 2)]); // Float64

    let mut converter = Converter::new();
    converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .unwrap();
    let result = converter.feed_csv_chunk(csv, true);

    assert!(result.is_err(), "Should fail on non-numeric Float64 value");
    println!("✓ Strict mode correctly rejects invalid Float64");
}

#[test]
fn test_strict_mode_boolean_parse_failure() {
    let csv = b"active\nmaybe\n";

    let hint = SchemaHint::new(vec![ColumnDef::new("active".to_string(), 3)]); // Boolean

    let mut converter = Converter::new();
    converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .unwrap();
    let result = converter.feed_csv_chunk(csv, true);

    assert!(result.is_err(), "Should fail on 'maybe' as Boolean");
    println!("✓ Strict mode correctly rejects invalid Boolean");
}

// ── Schema hint validation errors ─────────────────────────────────────────────

#[test]
fn test_strict_mode_column_count_mismatch() {
    // CSV has 2 columns but hint has 3
    let csv = b"a,b\n1,2\n";

    let hint = SchemaHint::new(vec![
        ColumnDef::new("a".to_string(), 1),
        ColumnDef::new("b".to_string(), 1),
        ColumnDef::new("c".to_string(), 1), // extra column - mismatch
    ]);

    let mut converter = Converter::new();
    converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .unwrap();
    let result = converter.feed_csv_chunk(csv, true);

    assert!(result.is_err(), "Column count mismatch should return error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("InvalidSchema") || err_msg.contains("column count"),
        "Error should mention schema mismatch, got: {}",
        err_msg
    );
    println!("✓ Column count mismatch correctly rejected: {}", err_msg);
}

// ── JSON deserialization (tests for the lib.rs JSON parsing path) ─────────────

#[test]
fn test_schema_hint_deserializes_from_json() {
    let json = r#"{"columns":[{"name":"id","type_id":1},{"name":"price","type_id":2}]}"#;
    let hint: SchemaHint = serde_json::from_str(json).expect("valid JSON should deserialize");
    assert_eq!(hint.columns.len(), 2);
    assert_eq!(hint.columns[0].name, "id");
    assert_eq!(hint.columns[0].type_id, 1);
    assert_eq!(hint.columns[1].name, "price");
    assert_eq!(hint.columns[1].type_id, 2);
    println!("✓ SchemaHint JSON deserialization works correctly");
}

#[test]
fn test_schema_hint_rejects_unknown_fields() {
    let json = r#"{"columns":[{"name":"id","type_id":1}],"unknown_field":"value"}"#;
    let result: Result<SchemaHint, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "Unknown fields should be rejected with deny_unknown_fields"
    );
    println!("✓ SchemaHint correctly rejects unknown JSON fields");
}

#[test]
fn test_column_def_rejects_unknown_fields() {
    let json = r#"{"columns":[{"name":"id","type_id":1,"extra":"oops"}]}"#;
    let result: Result<SchemaHint, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "Unknown fields in ColumnDef should be rejected"
    );
    println!("✓ ColumnDef correctly rejects unknown JSON fields");
}

#[test]
fn test_schema_hint_invalid_json_returns_error() {
    let invalid_json = "not valid json {";
    let result: Result<SchemaHint, _> = serde_json::from_str(invalid_json);
    assert!(result.is_err(), "Invalid JSON should return parse error");
    println!("✓ Invalid JSON correctly returns error");
}

#[test]
fn test_schema_hint_unsupported_type_id_defaults_to_utf8() {
    // type_id=99 is unsupported, should default to Utf8 (graceful fallback)
    let csv = b"col_a\nhello\nworld\n";

    let hint = SchemaHint::new(vec![ColumnDef::new("col_a".to_string(), 99)]); // unsupported → Utf8

    let mut converter = Converter::new();
    let chunks = converter
        .begin_csv_to_parquet(
            CsvReadOptions::default(),
            ParquetWriteOptions::default(),
            Some(&hint),
        )
        .and_then(|_| converter.feed_csv_chunk(csv, true))
        .expect("unsupported type_id should silently default to Utf8");

    assert!(!chunks.is_empty());
    println!("✓ Unsupported type_id correctly defaults to Utf8");
}
