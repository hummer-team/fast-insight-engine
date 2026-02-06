use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Schema};
use arrow::ipc::reader::StreamReader;
use ndarray::Array2;
use std::io::Cursor;
use std::sync::Arc;

use crate::utils::AnalysisError;

/// Parsed data from Arrow IPC format
#[derive(Debug)]
pub struct ParsedData {
    pub order_ids: Vec<i64>,
    pub features: Array2<f64>,
}

/// Parse Arrow IPC Stream format data
///
/// # Arguments
/// * `data` - Raw bytes in Arrow IPC Stream format
///
/// # Returns
/// * `Ok(ParsedData)` with order_ids and feature matrix
/// * `Err(AnalysisError)` if parsing fails or schema validation fails
pub fn parse_arrow_ipc(data: &[u8]) -> Result<ParsedData, AnalysisError> {
    if data.is_empty() {
        return Err(AnalysisError::ArrowError("empty input data".to_string()));
    }

    // Create cursor and StreamReader
    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| AnalysisError::ArrowError(format!("failed to create StreamReader: {}", e)))?;

    // Read all batches
    let mut all_order_ids: Vec<i64> = Vec::new();
    let mut all_feature_rows: Vec<Vec<f64>> = Vec::new();
    let mut feature_count: Option<usize> = None;

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| AnalysisError::ArrowError(format!("failed to read batch: {}", e)))?;

        // Validate schema on first batch
        if feature_count.is_none() {
            validate_schema(batch.schema())?;
            feature_count = Some(batch.num_columns() - 1); // exclude order_id column
        }

        // Extract order_ids (first column must be order_id)
        let order_id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                AnalysisError::ArrowError("order_id column is not Int64Array".to_string())
            })?;

        for i in 0..batch.num_rows() {
            all_order_ids.push(order_id_col.value(i));
        }

        // Extract feature columns (all columns except first)
        for row_idx in 0..batch.num_rows() {
            let mut row_features = Vec::new();
            for col_idx in 1..batch.num_columns() {
                let col = batch.column(col_idx);
                let value = extract_float64_value(col, row_idx)?;
                row_features.push(value);
            }
            all_feature_rows.push(row_features);
        }
    }

    if all_order_ids.is_empty() {
        return Err(AnalysisError::ValidationError(
            "no data rows found".to_string(),
        ));
    }

    // Convert feature rows to Array2
    let num_rows = all_feature_rows.len();
    let num_cols = feature_count.unwrap_or(0);

    if num_cols == 0 {
        return Err(AnalysisError::ValidationError(
            "no feature columns found".to_string(),
        ));
    }

    let flat_features: Vec<f64> = all_feature_rows.into_iter().flatten().collect();
    let features = Array2::from_shape_vec((num_rows, num_cols), flat_features)
        .map_err(|e| AnalysisError::ArrowError(format!("failed to create Array2: {}", e)))?;

    Ok(ParsedData {
        order_ids: all_order_ids,
        features,
    })
}

/// Validate Arrow schema has required fields
fn validate_schema(schema: Arc<Schema>) -> Result<(), AnalysisError> {
    if schema.fields().is_empty() {
        return Err(AnalysisError::ArrowError(
            "schema has no fields".to_string(),
        ));
    }

    // First field must be order_id (Int64)
    let first_field = &schema.fields()[0];
    if first_field.name() != "order_id" {
        return Err(AnalysisError::ArrowError(format!(
            "first field must be 'order_id', got '{}'",
            first_field.name()
        )));
    }

    if !matches!(first_field.data_type(), DataType::Int64) {
        return Err(AnalysisError::ArrowError(format!(
            "order_id must be Int64, got {:?}",
            first_field.data_type()
        )));
    }

    // All other fields should be Float64 (features)
    for (idx, field) in schema.fields().iter().enumerate().skip(1) {
        if !matches!(field.data_type(), DataType::Float64) {
            return Err(AnalysisError::ArrowError(format!(
                "feature column '{}' at index {} must be Float64, got {:?}",
                field.name(),
                idx,
                field.data_type()
            )));
        }
    }

    Ok(())
}

/// Extract f64 value from array at given index
fn extract_float64_value(array: &ArrayRef, index: usize) -> Result<f64, AnalysisError> {
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| AnalysisError::ArrowError("column is not Float64Array".to_string()))?
        .value(index);

    let float_array = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| AnalysisError::ArrowError("column is not Float64Array".to_string()))?;

    Ok(float_array.value(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn create_test_arrow_data(
        order_ids: Vec<i64>,
        features: Vec<Vec<f64>>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let num_features = features.first().map(|f| f.len()).unwrap_or(0);

        // Build schema
        let mut fields = vec![Field::new("order_id", DataType::Int64, false)];
        for i in 0..num_features {
            fields.push(Field::new(
                &format!("feature_{}", i),
                DataType::Float64,
                false,
            ));
        }
        let schema = Arc::new(Schema::new(fields));

        // Build arrays
        let order_id_array = Arc::new(Int64Array::from(order_ids)) as ArrayRef;
        let mut feature_arrays: Vec<ArrayRef> = Vec::new();
        for i in 0..num_features {
            let col_data: Vec<f64> = features.iter().map(|row| row[i]).collect();
            feature_arrays.push(Arc::new(Float64Array::from(col_data)) as ArrayRef);
        }

        let mut columns = vec![order_id_array];
        columns.extend(feature_arrays);

        let batch = RecordBatch::try_new(schema.clone(), columns)?;

        // Write to IPC Stream format
        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema)?;
            writer.write(&batch)?;
            writer.finish()?;
        }

        Ok(buffer)
    }

    #[test]
    fn test_parse_arrow_ipc_normal() {
        let order_ids = vec![1, 2, 3];
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let data = create_test_arrow_data(order_ids.clone(), features.clone()).unwrap();

        let result = parse_arrow_ipc(&data).unwrap();
        assert_eq!(result.order_ids, order_ids);
        assert_eq!(result.features.nrows(), 3);
        assert_eq!(result.features.ncols(), 2);
        assert_eq!(result.features[[0, 0]], 1.0);
        assert_eq!(result.features[[2, 1]], 6.0);
    }

    #[test]
    fn test_parse_arrow_ipc_empty() {
        let result = parse_arrow_ipc(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty input"));
    }
}
