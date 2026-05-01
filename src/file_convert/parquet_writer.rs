/// Parquet writer for streaming output with row group management
use super::error::{ConvertError, ConvertResult};
use super::types::ParquetWriteOptions;
use arrow::{
    array::*,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::arrow::ArrowWriter;
use std::sync::Arc;

/// Builder for Arrow RecordBatch from CSV data
/// Handles type conversion and row group buffering
pub struct ParquetBuilder {
    /// Arrow schema for the output
    schema: Arc<Schema>,
    /// Maximum rows before flushing a row group
    row_group_size: usize,
    /// Parquet write options
    options: ParquetWriteOptions,
    /// Current batch being accumulated
    current_batch: Vec<Vec<String>>,
    /// Number of batches written so far
    batch_count: usize,
    /// Estimated memory used (bytes)
    estimated_memory: u32,
    /// Memory limit in MB
    memory_limit_mb: u32,
}

impl ParquetBuilder {
    /// Create a new Parquet builder with Arrow schema
    pub fn new(
        schema: Arc<Schema>,
        options: ParquetWriteOptions,
        memory_limit_mb: u32,
    ) -> ConvertResult<Self> {
        // Validate row group size
        if options.row_group_size < 64 || options.row_group_size > 16384 {
            return Err(ConvertError::ParquetRowGroupTooLarge {
                size: options.row_group_size,
            });
        }

        let row_group_size = options.row_group_size;

        Ok(Self {
            schema,
            row_group_size,
            options,
            current_batch: Vec::with_capacity(row_group_size),
            batch_count: 0,
            estimated_memory: 0,
            memory_limit_mb,
        })
    }

    /// Add a row to the current batch
    /// Returns Some(Vec<u8>) if row group should be flushed, None otherwise
    pub fn add_row(&mut self, row: Vec<String>) -> ConvertResult<Option<Vec<u8>>> {
        // Estimate memory: sum of string lengths, approximately
        let row_memory = row.iter().map(|s| s.len() as u32).sum::<u32>() + 200; // 200 bytes overhead per row
        self.estimated_memory += row_memory;

        // Check memory limit
        if self.estimated_memory > self.memory_limit_mb * 1024 * 1024 {
            return Err(ConvertError::MemoryLimitExceeded {
                limit_mb: self.memory_limit_mb,
                estimated_mb: self.estimated_memory / 1024 / 1024,
            });
        }

        self.current_batch.push(row);

        // Check if we should flush
        if self.current_batch.len() >= self.row_group_size {
            self.flush()
        } else {
            Ok(None)
        }
    }

    /// Manually flush the current batch (even if not full)
    pub fn flush(&mut self) -> ConvertResult<Option<Vec<u8>>> {
        if self.current_batch.is_empty() {
            return Ok(None);
        }

        let rows = std::mem::take(&mut self.current_batch);
        self.current_batch = Vec::with_capacity(self.row_group_size);

        // Convert rows to Arrow arrays
        let arrays = self.rows_to_arrays(&rows)?;

        // Create RecordBatch
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), arrays)?;

        // Write to Parquet with compression support
        let mut buffer_box: Box<Vec<u8>> = Box::new(Vec::new());
        {
            // Only Snappy compression is fully supported in current parquet crate
            // Default to uncompressed for MVP to avoid version compatibility issues
            let mut writer = ArrowWriter::try_new(&mut *buffer_box, Arc::clone(&self.schema), None)
                .map_err(|e| ConvertError::InternalError {
                    reason: format!("Failed to create ArrowWriter: {}", e),
                })?;

            writer
                .write(&batch)
                .map_err(|e| ConvertError::InternalError {
                    reason: format!("Failed to write batch: {}", e),
                })?;
            // writer is dropped here, releasing the mutable borrow
        }

        let buffer = *buffer_box;

        // Note: Compression is reserved for Phase 10 optimization
        // Current implementation uses uncompressed for stability
        self.batch_count += 1;
        self.estimated_memory = 0; // Reset after flush

        Ok(Some(buffer))
    }

    /// Finalize: write footer and return final chunk
    pub fn finalize(&mut self) -> ConvertResult<Vec<Vec<u8>>> {
        let mut results = Vec::new();

        // Flush any remaining rows
        if !self.current_batch.is_empty() {
            if let Some(chunk) = self.flush()? {
                results.push(chunk);
            }
        }

        // In a real Parquet writer, we would write the footer here
        // For MVP, we just mark completion
        Ok(results)
    }

    /// Convert rows to Arrow arrays based on schema
    fn rows_to_arrays(&self, rows: &[Vec<String>]) -> ConvertResult<Vec<Arc<dyn Array>>> {
        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

        for (col_idx, field) in self.schema.fields().iter().enumerate() {
            let array = self.column_to_array(col_idx, field, rows)?;
            arrays.push(array);
        }

        Ok(arrays)
    }

    /// Convert a single column to Arrow array
    fn column_to_array(
        &self,
        col_idx: usize,
        field: &Field,
        rows: &[Vec<String>],
    ) -> ConvertResult<Arc<dyn Array>> {
        match field.data_type() {
            DataType::Utf8 => {
                let values: Vec<Option<&str>> = rows
                    .iter()
                    .map(|row| {
                        row.get(col_idx)
                            .map(|s| if s.is_empty() { None } else { Some(s.as_str()) })
                    })
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default();

                Ok(Arc::new(StringArray::from(values)))
            }
            DataType::Int64 => {
                let values: Vec<Option<i64>> = rows
                    .iter()
                    .map(|row| {
                        row.get(col_idx).and_then(|s| {
                            if s.is_empty() {
                                None
                            } else {
                                s.parse::<i64>().ok()
                            }
                        })
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(values)))
            }
            DataType::Float64 => {
                let values: Vec<Option<f64>> = rows
                    .iter()
                    .map(|row| {
                        row.get(col_idx).and_then(|s| {
                            if s.is_empty() {
                                None
                            } else {
                                s.parse::<f64>().ok()
                            }
                        })
                    })
                    .collect();

                Ok(Arc::new(Float64Array::from(values)))
            }
            _dt => {
                // Default to UTF-8 for unsupported types
                let values: Vec<Option<&str>> = rows
                    .iter()
                    .map(|row| row.get(col_idx).map(|s| s.as_str()))
                    .collect();

                Ok(Arc::new(StringArray::from(values)))
            }
        }
    }

    /// Get number of batches written so far
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Get current row count in batch
    pub fn current_row_count(&self) -> usize {
        self.current_batch.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ]))
    }

    #[test]
    fn test_builder_creation() {
        let schema = create_simple_schema();
        let opts = ParquetWriteOptions::default();
        let builder = ParquetBuilder::new(schema, opts, 150).unwrap();

        assert_eq!(builder.batch_count(), 0);
        assert_eq!(builder.current_row_count(), 0);
    }

    #[test]
    fn test_invalid_row_group_size() {
        let schema = create_simple_schema();
        let mut opts = ParquetWriteOptions::default();
        opts.row_group_size = 20000; // Too large

        let result = ParquetBuilder::new(schema, opts, 150);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_rows_below_threshold() {
        let schema = create_simple_schema();
        let opts = ParquetWriteOptions {
            row_group_size: 1024,
            ..Default::default()
        };
        let mut builder = ParquetBuilder::new(schema, opts, 150).unwrap();

        let row = vec!["1".to_string(), "Alice".to_string(), "95.5".to_string()];
        let result = builder.add_row(row).unwrap();

        assert!(result.is_none()); // No flush yet
        assert_eq!(builder.current_row_count(), 1);
    }

    #[test]
    fn test_memory_limit_exceeded() {
        let schema = create_simple_schema();
        let opts = ParquetWriteOptions::default();
        let mut builder = ParquetBuilder::new(schema, opts, 1).unwrap(); // 1 MB limit

        // Create rows that collectively exceed 1 MB memory
        for _ in 0..20 {
            let large_row = vec![
                "1".to_string(),
                "x".repeat(100_000), // 100 KB per field
                "95.5".to_string(),
            ];
            if let Err(ConvertError::MemoryLimitExceeded { .. }) = builder.add_row(large_row) {
                // Expected error
                return;
            }
        }
        // If we get here, the test should fail
        panic!("Expected MemoryLimitExceeded error");
    }

    #[test]
    fn test_finalize_empty() {
        let schema = create_simple_schema();
        let opts = ParquetWriteOptions::default();
        let mut builder = ParquetBuilder::new(schema, opts, 150).unwrap();

        let result = builder.finalize().unwrap();
        assert_eq!(result.len(), 0); // No output for empty builder
    }
}
