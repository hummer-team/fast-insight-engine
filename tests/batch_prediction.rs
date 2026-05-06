//! Integration tests for predict_inventory_demand_batch core logic.
//!
//! Tests use a non-Wasm helper `run_batch_prediction` that mirrors the
//! Wasm function's core logic without the wasm-bindgen boundary.

use arrow::array::{Array, Float64Array, Int32Array, StringArray};
use arrow::ipc::reader::StreamReader;
use fast_insight_engine::arrow_handler::{
    SkuPredictionResult, build_batch_regression_result, parse_batch_arrow_ipc,
};
use fast_insight_engine::insight_core::model::{PredictionMode, run_regression_with_mode};
use ndarray::Array1;
use std::io::Cursor;

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_batch_ipc(rows: Vec<(&str, f64, f64)>) -> Vec<u8> {
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("sku_id", DataType::Utf8, false),
        Field::new("time_index", DataType::Float64, false),
        Field::new("demand", DataType::Float64, false),
    ]));
    let sku_ids: Vec<&str> = rows.iter().map(|(s, _, _)| *s).collect();
    let time_indices: Vec<f64> = rows.iter().map(|(_, t, _)| *t).collect();
    let demands: Vec<f64> = rows.iter().map(|(_, _, d)| *d).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(sku_ids)) as _,
            Arc::new(Float64Array::from(time_indices)) as _,
            Arc::new(Float64Array::from(demands)) as _,
        ],
    )
    .unwrap();

    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    buf
}

/// Mirrors the core logic of `predict_inventory_demand_batch` without the Wasm boundary.
fn run_batch_prediction(
    data: &[u8],
    predict_steps: usize,
    mode: PredictionMode,
) -> Result<Vec<u8>, String> {
    let parsed = parse_batch_arrow_ipc(data).map_err(|e| e.to_string())?;

    let mut sku_map: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut sku_order: Vec<String> = Vec::new();
    for (i, sku_id) in parsed.order_ids.iter().enumerate() {
        if !sku_map.contains_key(sku_id) {
            sku_order.push(sku_id.clone());
        }
        sku_map
            .entry(sku_id.clone())
            .or_default()
            .push(parsed.features[[i, 1]]);
    }

    let mut results: Vec<SkuPredictionResult> = Vec::with_capacity(sku_order.len());
    for sku_id in &sku_order {
        let demands = sku_map.remove(sku_id).unwrap_or_default();
        let result = run_regression_with_mode(Array1::from(demands), predict_steps, mode);
        match result {
            Ok(predictions) => results.push(SkuPredictionResult::Success {
                sku_id: sku_id.clone(),
                predictions,
            }),
            Err(e) => {
                let s = e.to_string();
                let (code, msg) = if let Some(pos) = s.find(": ") {
                    (s[..pos].to_string(), s[pos + 2..].to_string())
                } else {
                    ("ModelError".to_string(), s)
                };
                results.push(SkuPredictionResult::Failure {
                    sku_id: sku_id.clone(),
                    error_code: code,
                    error_message: msg,
                });
            }
        }
    }

    build_batch_regression_result(results).map_err(|e| e.to_string())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn test_batch_all_success() {
    let ipc = make_batch_ipc(vec![
        ("SKU-A", 0.0, 100.0),
        ("SKU-A", 1.0, 110.0),
        ("SKU-A", 2.0, 120.0),
        ("SKU-B", 0.0, 50.0),
        ("SKU-B", 1.0, 55.0),
        ("SKU-B", 2.0, 60.0),
    ]);
    let result = run_batch_prediction(&ipc, 2, PredictionMode::Linear).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(result), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    // 2 SKUs × 2 steps = 4 rows
    assert_eq!(batch.num_rows(), 4);

    let err_codes = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    // All error_code columns should be null (success rows)
    for i in 0..4 {
        assert!(err_codes.is_null(i), "row {i} should be a success row");
    }
}

#[test]
fn test_batch_mixed_success_error() {
    // SKU-A has 3 samples (ok), SKU-B has only 1 sample (ValidationError)
    let ipc = make_batch_ipc(vec![
        ("SKU-A", 0.0, 100.0),
        ("SKU-A", 1.0, 110.0),
        ("SKU-A", 2.0, 120.0),
        ("SKU-B", 0.0, 50.0), // only 1 sample → error
    ]);
    let result = run_batch_prediction(&ipc, 1, PredictionMode::Linear).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(result), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    // SKU-A: 1 success row; SKU-B: 1 error row → 2 rows total
    assert_eq!(batch.num_rows(), 2);

    let err_codes = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    // row 0 = SKU-A success
    assert!(err_codes.is_null(0));
    // row 1 = SKU-B error
    assert!(!err_codes.is_null(1));
    assert_eq!(err_codes.value(1), "ValidationError");
}

#[test]
fn test_batch_sequential_step_indices() {
    let ipc = make_batch_ipc(vec![
        ("SKU-X", 0.0, 10.0),
        ("SKU-X", 1.0, 20.0),
        ("SKU-X", 2.0, 30.0),
    ]);
    let result = run_batch_prediction(&ipc, 3, PredictionMode::Linear).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(result), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    assert_eq!(batch.num_rows(), 3);
    let steps = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(steps.value(0), 0);
    assert_eq!(steps.value(1), 1);
    assert_eq!(steps.value(2), 2);
}

#[test]
fn test_batch_predictions_not_empty() {
    let ipc = make_batch_ipc(vec![
        ("SKU-Z", 0.0, 100.0),
        ("SKU-Z", 1.0, 200.0),
        ("SKU-Z", 2.0, 300.0),
    ]);
    let result = run_batch_prediction(&ipc, 2, PredictionMode::Linear).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(result), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    let preds = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    // All prediction values should be non-null and finite
    for i in 0..batch.num_rows() {
        assert!(!preds.is_null(i));
        assert!(preds.value(i).is_finite());
    }
}

#[test]
fn test_batch_preserves_sku_order() {
    // SKU-C appears first, SKU-A second — output should preserve this order
    let ipc = make_batch_ipc(vec![
        ("SKU-C", 0.0, 30.0),
        ("SKU-C", 1.0, 35.0),
        ("SKU-C", 2.0, 40.0),
        ("SKU-A", 0.0, 10.0),
        ("SKU-A", 1.0, 15.0),
        ("SKU-A", 2.0, 20.0),
    ]);
    let result = run_batch_prediction(&ipc, 1, PredictionMode::Linear).unwrap();
    let mut reader = StreamReader::try_new(Cursor::new(result), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    let sku_ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    // First row should be SKU-C (first seen)
    assert_eq!(sku_ids.value(0), "SKU-C");
    // Second row should be SKU-A (second seen)
    assert_eq!(sku_ids.value(1), "SKU-A");
}

#[test]
fn test_batch_duplicate_time_indices_same_sku() {
    // Test that duplicate time indices for same SKU doesn't cause issues
    // The current implementation just appends demand values regardless of time_index
    let ipc = make_batch_ipc(vec![
        ("SKU-A", 0.0, 100.0),
        ("SKU-A", 0.0, 105.0), // duplicate time 0.0
        ("SKU-A", 1.0, 110.0),
        ("SKU-A", 1.0, 115.0), // duplicate time 1.0
    ]);
    let result = run_batch_prediction(&ipc, 1, PredictionMode::Linear);

    // Should succeed - it treats all 4 values as a sequence
    assert!(result.is_ok(), "Should handle duplicate time indices");

    let mut reader = StreamReader::try_new(Cursor::new(result.unwrap()), None).unwrap();
    let batch = reader.next().unwrap().unwrap();

    let err_codes = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Should be a success row (error_code is null)
    assert!(err_codes.is_null(0));
}
