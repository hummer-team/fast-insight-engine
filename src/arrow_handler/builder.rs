use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int32Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::utils::AnalysisError;

/// Build Arrow IPC result for anomaly detection
///
/// # Arguments
/// * `order_ids` - Original order IDs (String type for business compatibility)
/// * `scores` - Anomaly scores (0-1 range, higher = more anomalous)
/// * `labels` - Binary anomaly labels (true = anomalous)
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError)` - If building fails
pub fn build_anomaly_result(
    order_ids: Vec<String>,
    scores: Vec<f64>,
    labels: Vec<bool>,
) -> Result<Vec<u8>, AnalysisError> {
    if order_ids.len() != scores.len() || scores.len() != labels.len() {
        return Err(AnalysisError::ValidationError(
            "order_ids, scores, and labels must have same length".to_string(),
        ));
    }

    // Define schema (order fixed: order_id, abnormal_score, is_abnormal)
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Utf8, false),
        Field::new("abnormal_score", DataType::Float64, false),
        Field::new("is_abnormal", DataType::Boolean, false),
    ]));

    // Build arrays
    let order_id_array = Arc::new(StringArray::from(order_ids)) as ArrayRef;
    let score_array = Arc::new(Float64Array::from(scores)) as ArrayRef;
    let label_array = Arc::new(BooleanArray::from(labels)) as ArrayRef;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![order_id_array, score_array, label_array],
    )
    .map_err(|e| AnalysisError::ArrowError(format!("failed to create RecordBatch: {}", e)))?;

    // Write to IPC Stream format
    serialize_to_ipc(schema, batch)
}

/// Build Arrow IPC result for clustering
///
/// # Arguments
/// * `order_ids` - Original order IDs (String type for business compatibility)
/// * `cluster_ids` - Cluster assignments (0, 1, 2, ...)
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError)` - If building fails
pub fn build_cluster_result(
    order_ids: Vec<String>,
    cluster_ids: Vec<usize>,
) -> Result<Vec<u8>, AnalysisError> {
    if order_ids.len() != cluster_ids.len() {
        return Err(AnalysisError::ValidationError(
            "order_ids and cluster_ids must have same length".to_string(),
        ));
    }

    // Define schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Utf8, false),
        Field::new("cluster_id", DataType::UInt64, false),
    ]));

    // Build arrays
    let order_id_array = Arc::new(StringArray::from(order_ids)) as ArrayRef;
    let cluster_array = Arc::new(UInt64Array::from(
        cluster_ids.iter().map(|&c| c as u64).collect::<Vec<_>>(),
    )) as ArrayRef;

    let batch = RecordBatch::try_new(schema.clone(), vec![order_id_array, cluster_array])
        .map_err(|e| AnalysisError::ArrowError(format!("failed to create RecordBatch: {}", e)))?;

    // Write to IPC Stream format
    serialize_to_ipc(schema, batch)
}

/// Build Arrow IPC result for regression predictions
///
/// # Arguments
/// * `predictions` - Predicted values
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError)` - If building fails
pub fn build_regression_result(predictions: Vec<f64>) -> Result<Vec<u8>, AnalysisError> {
    if predictions.is_empty() {
        return Err(AnalysisError::ValidationError(
            "predictions cannot be empty".to_string(),
        ));
    }

    // Define schema
    let schema = Arc::new(Schema::new(vec![Field::new(
        "prediction",
        DataType::Float64,
        false,
    )]));

    // Build array
    let pred_array = Arc::new(Float64Array::from(predictions)) as ArrayRef;

    let batch = RecordBatch::try_new(schema.clone(), vec![pred_array])
        .map_err(|e| AnalysisError::ArrowError(format!("failed to create RecordBatch: {}", e)))?;

    // Write to IPC Stream format
    serialize_to_ipc(schema, batch)
}

/// Outcome for a single SKU in a batch prediction call.
pub enum SkuPredictionResult {
    /// Regression succeeded; `predictions` has `predict_steps` values.
    Success {
        sku_id: String,
        predictions: Vec<f64>,
    },
    /// Regression failed; caller receives an error mapping row instead of predictions.
    Failure {
        sku_id: String,
        /// Short error category: "ValidationError" or "ModelError"
        error_code: String,
        /// Human-readable reason, e.g. "too few samples: got 1, need ≥ 2"
        error_message: String,
    },
}

/// Build Arrow IPC result for batch SKU regression predictions.
///
/// Output schema (5 columns):
/// - `sku_id`:        Utf8    NOT NULL
/// - `step_index`:    Int32   NULLABLE — 0,1,2,… for success rows; NULL for error rows
/// - `prediction`:    Float64 NULLABLE — forecasted value for success rows; NULL for error rows
/// - `error_code`:    Utf8    NULLABLE — "ValidationError"/"ModelError"; NULL for success rows
/// - `error_message`: Utf8    NULLABLE — human-readable reason; NULL for success rows
///
/// # Arguments
/// * `results` - Per-SKU prediction outcomes
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError::ValidationError)` - If `results` is empty
pub fn build_batch_regression_result(
    results: Vec<SkuPredictionResult>,
) -> Result<Vec<u8>, AnalysisError> {
    if results.is_empty() {
        return Err(AnalysisError::ValidationError(
            "results cannot be empty".to_string(),
        ));
    }

    // Expand each result into rows
    let mut owned_sku_ids: Vec<String> = Vec::new();
    let mut step_indices: Vec<Option<i32>> = Vec::new();
    let mut predictions: Vec<Option<f64>> = Vec::new();
    let mut owned_err_codes: Vec<String> = Vec::new();
    let mut owned_err_msgs: Vec<String> = Vec::new();
    let mut is_error: Vec<bool> = Vec::new();

    for result in results {
        match result {
            SkuPredictionResult::Success {
                sku_id,
                predictions: preds,
            } => {
                for (step, pred) in preds.iter().enumerate() {
                    owned_sku_ids.push(sku_id.clone());
                    step_indices.push(Some(step as i32));
                    predictions.push(Some(*pred));
                    owned_err_codes.push(String::new());
                    owned_err_msgs.push(String::new());
                    is_error.push(false);
                }
            }
            SkuPredictionResult::Failure {
                sku_id,
                error_code,
                error_message,
            } => {
                owned_sku_ids.push(sku_id);
                step_indices.push(None);
                predictions.push(None);
                owned_err_codes.push(error_code);
                owned_err_msgs.push(error_message);
                is_error.push(true);
            }
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("sku_id", DataType::Utf8, false),
        Field::new("step_index", DataType::Int32, true),
        Field::new("prediction", DataType::Float64, true),
        Field::new("error_code", DataType::Utf8, true),
        Field::new("error_message", DataType::Utf8, true),
    ]));

    // Build sku_id array (non-nullable)
    let sku_id_refs: Vec<Option<&str>> = owned_sku_ids.iter().map(|s| Some(s.as_str())).collect();
    let sku_id_array = Arc::new(StringArray::from(sku_id_refs)) as ArrayRef;

    let step_array = Arc::new(Int32Array::from(step_indices)) as ArrayRef;
    let pred_array = Arc::new(Float64Array::from(predictions)) as ArrayRef;

    // Nullable error columns: None for success rows
    let err_code_refs: Vec<Option<&str>> = owned_err_codes
        .iter()
        .zip(is_error.iter())
        .map(|(s, &is_err)| if is_err { Some(s.as_str()) } else { None })
        .collect();
    let err_code_array = Arc::new(StringArray::from(err_code_refs)) as ArrayRef;

    let err_msg_refs: Vec<Option<&str>> = owned_err_msgs
        .iter()
        .zip(is_error.iter())
        .map(|(s, &is_err)| if is_err { Some(s.as_str()) } else { None })
        .collect();
    let err_msg_array = Arc::new(StringArray::from(err_msg_refs)) as ArrayRef;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            sku_id_array,
            step_array,
            pred_array,
            err_code_array,
            err_msg_array,
        ],
    )
    .map_err(|e| AnalysisError::ArrowError(format!("failed to create RecordBatch: {}", e)))?;

    serialize_to_ipc(schema, batch)
}

/// Serialize RecordBatch to Arrow IPC Stream format
fn serialize_to_ipc(schema: Arc<Schema>, batch: RecordBatch) -> Result<Vec<u8>, AnalysisError> {
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).map_err(|e| {
            AnalysisError::ArrowError(format!("failed to create StreamWriter: {}", e))
        })?;
        writer
            .write(&batch)
            .map_err(|e| AnalysisError::ArrowError(format!("failed to write batch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| AnalysisError::ArrowError(format!("failed to finish writer: {}", e)))?;
    }
    Ok(buffer)
}

/// Build Arrow IPC result for FP-Growth frequent pattern mining.
///
/// # Output schema
/// - `pattern`: `List<Utf8>` — frequent itemset (e.g., `["milk", "bread"]`)
/// - `support`: `Int64` — number of transactions containing the pattern
///
/// An empty `patterns` slice produces a valid zero-row Arrow IPC result.
///
/// # Arguments
/// * `patterns` - Frequent patterns with support counts (from `FPResult::frequent_patterns()`).
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError)` - If Arrow serialization fails
pub fn build_pattern_result(
    patterns: Vec<(Vec<String>, usize)>,
) -> Result<Vec<u8>, AnalysisError> {
    use arrow::array::{Int64Array, ListBuilder, StringBuilder};

    // Build List<Utf8> array for pattern column.
    let mut list_builder = ListBuilder::new(StringBuilder::new());
    let mut supports: Vec<i64> = Vec::with_capacity(patterns.len());

    for (pattern, support) in &patterns {
        for item in pattern {
            list_builder.values().append_value(item);
        }
        list_builder.append(true);
        supports.push(*support as i64);
    }

    let pattern_array = Arc::new(list_builder.finish()) as ArrayRef;
    let support_array = Arc::new(Int64Array::from(supports)) as ArrayRef;

    // Derive schema from the built array to guarantee field-name consistency.
    let schema = Arc::new(Schema::new(vec![
        Field::new("pattern", pattern_array.data_type().clone(), false),
        Field::new("support", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![pattern_array, support_array])
        .map_err(|e| AnalysisError::ArrowError(format!("failed to create RecordBatch: {}", e)))?;

    serialize_to_ipc(schema, batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    #[test]
    fn test_build_anomaly_result() {
        let order_ids = vec![
            "ORD001".to_string(),
            "ORD002".to_string(),
            "ORD003".to_string(),
        ];
        let scores = vec![0.1, 0.5, 0.9];
        let labels = vec![false, false, true];

        let result = build_anomaly_result(order_ids, scores, labels).unwrap();
        assert!(!result.is_empty());

        // Verify by parsing back
        let cursor = Cursor::new(result);
        let reader = StreamReader::try_new(cursor, None).unwrap();
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.fields()[0].name(), "order_id");
        assert_eq!(schema.fields()[1].name(), "abnormal_score");
        assert_eq!(schema.fields()[2].name(), "is_abnormal");
    }

    #[test]
    fn test_build_cluster_result() {
        let order_ids = vec![
            "ORD001".to_string(),
            "ORD002".to_string(),
            "ORD003".to_string(),
        ];
        let cluster_ids = vec![0, 1, 0];

        let result = build_cluster_result(order_ids, cluster_ids).unwrap();
        assert!(!result.is_empty());

        // Verify by parsing back
        let cursor = Cursor::new(result);
        let reader = StreamReader::try_new(cursor, None).unwrap();
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.fields()[0].name(), "order_id");
        assert_eq!(schema.fields()[1].name(), "cluster_id");
    }

    #[test]
    fn test_build_regression_result() {
        let predictions = vec![10.5, 20.3, 30.1];

        let result = build_regression_result(predictions).unwrap();
        assert!(!result.is_empty());

        // Verify by parsing back
        let cursor = Cursor::new(result);
        let reader = StreamReader::try_new(cursor, None).unwrap();
        let schema = reader.schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.fields()[0].name(), "prediction");
    }

    #[test]
    fn test_build_anomaly_result_length_mismatch() {
        let result = build_anomaly_result(
            vec!["ORD001".to_string(), "ORD002".to_string()],
            vec![0.5],
            vec![false, true],
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must have same length")
        );
    }

    #[test]
    fn test_build_regression_result_empty() {
        let result = build_regression_result(vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cannot be empty"));
    }

    #[test]
    fn test_build_batch_regression_result_all_success() {
        let results = vec![
            SkuPredictionResult::Success {
                sku_id: "SKU-001".to_string(),
                predictions: vec![10.0, 20.0],
            },
            SkuPredictionResult::Success {
                sku_id: "SKU-002".to_string(),
                predictions: vec![5.0],
            },
        ];
        let bytes = build_batch_regression_result(results).unwrap();
        assert!(!bytes.is_empty());

        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;
        let reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
        let schema = reader.schema();
        assert_eq!(schema.fields()[0].name(), "sku_id");
        assert_eq!(schema.fields()[1].name(), "step_index");
        assert_eq!(schema.fields()[2].name(), "prediction");
        assert_eq!(schema.fields()[3].name(), "error_code");
        assert_eq!(schema.fields()[4].name(), "error_message");
    }

    #[test]
    fn test_build_batch_regression_result_mixed() {
        let results = vec![
            SkuPredictionResult::Success {
                sku_id: "SKU-A".to_string(),
                predictions: vec![100.0, 110.0, 120.0],
            },
            SkuPredictionResult::Failure {
                sku_id: "SKU-B".to_string(),
                error_code: "ValidationError".to_string(),
                error_message: "too few samples: got 1, need >= 2".to_string(),
            },
        ];
        let bytes = build_batch_regression_result(results).unwrap();
        assert!(!bytes.is_empty());

        use arrow::array::{Float64Array, Int32Array, StringArray};
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;
        let mut reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
        let batch = reader.next().unwrap().unwrap();

        // 3 success rows (SKU-A) + 1 error row (SKU-B) = 4 rows total
        assert_eq!(batch.num_rows(), 4);

        let sku_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(sku_ids.value(0), "SKU-A");
        assert_eq!(sku_ids.value(3), "SKU-B");

        // step_index: set for success rows, null for error row
        let steps = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(steps.value(0), 0);
        assert_eq!(steps.value(1), 1);
        assert_eq!(steps.value(2), 2);
        assert!(steps.is_null(3));

        // prediction: set for success rows, null for error row
        let preds = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((preds.value(0) - 100.0).abs() < 1e-9);
        assert!(preds.is_null(3));

        // error columns: null for success rows, set for error row
        let err_codes = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(err_codes.is_null(0));
        assert_eq!(err_codes.value(3), "ValidationError");

        let err_msgs = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(err_msgs.is_null(0));
        assert!(err_msgs.value(3).contains("too few samples"));
    }

    #[test]
    fn test_build_batch_regression_result_all_errors() {
        let results = vec![SkuPredictionResult::Failure {
            sku_id: "SKU-X".to_string(),
            error_code: "ModelError".to_string(),
            error_message: "singular matrix".to_string(),
        }];
        let bytes = build_batch_regression_result(results).unwrap();
        assert!(!bytes.is_empty());

        use arrow::array::{Int32Array, StringArray};
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;
        let mut reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let steps = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(steps.is_null(0));
        let err_codes = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(err_codes.value(0), "ModelError");
    }

    #[test]
    fn test_build_batch_regression_result_empty_returns_error() {
        let result = build_batch_regression_result(vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }
}
