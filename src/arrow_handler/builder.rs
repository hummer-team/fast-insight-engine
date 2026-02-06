use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::utils::AnalysisError;

/// Build Arrow IPC result for anomaly detection
///
/// # Arguments
/// * `order_ids` - Original order IDs
/// * `scores` - Anomaly scores (0-1 range, higher = more anomalous)
/// * `labels` - Binary anomaly labels (true = anomalous)
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError)` - If building fails
pub fn build_anomaly_result(
    order_ids: Vec<i64>,
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
        Field::new("order_id", DataType::Int64, false),
        Field::new("abnormal_score", DataType::Float64, false),
        Field::new("is_abnormal", DataType::Boolean, false),
    ]));

    // Build arrays
    let order_id_array = Arc::new(Int64Array::from(order_ids)) as ArrayRef;
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
/// * `order_ids` - Original order IDs
/// * `cluster_ids` - Cluster assignments (0, 1, 2, ...)
///
/// # Returns
/// * `Ok(Vec<u8>)` - Arrow IPC Stream format bytes
/// * `Err(AnalysisError)` - If building fails
pub fn build_cluster_result(
    order_ids: Vec<i64>,
    cluster_ids: Vec<usize>,
) -> Result<Vec<u8>, AnalysisError> {
    if order_ids.len() != cluster_ids.len() {
        return Err(AnalysisError::ValidationError(
            "order_ids and cluster_ids must have same length".to_string(),
        ));
    }

    // Define schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("cluster_id", DataType::UInt64, false),
    ]));

    // Build arrays
    let order_id_array = Arc::new(Int64Array::from(order_ids)) as ArrayRef;
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    #[test]
    fn test_build_anomaly_result() {
        let order_ids = vec![1, 2, 3];
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
        let order_ids = vec![1, 2, 3];
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
        let result = build_anomaly_result(vec![1, 2], vec![0.5], vec![false, true]);
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
}
