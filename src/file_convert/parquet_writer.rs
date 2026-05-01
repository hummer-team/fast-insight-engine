/// Parquet writer for DuckDB Wasm compatibility
/// Generates complete, valid .parquet files with proper Parquet footer metadata
/// 
/// # Overview
/// Parquet file format: MAGIC (4B) | RowGroups | FileMetadata | MetadataLength (4B) | MAGIC (4B)
/// This writer produces valid Parquet files that DuckDB Wasm can directly load.
use super::error::{ConvertError, ConvertResult};
use super::types::ParquetWriteOptions;
use arrow::{
    array::*,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::arrow::ArrowWriter;
use std::sync::Arc;

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Builder for Arrow RecordBatch from CSV data
/// Handles type conversion, row group buffering, and complete Parquet file generation
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
    /// Accumulated row groups (each is complete Parquet file data)
    row_groups: Vec<Vec<u8>>,
    /// Whether this builder has been finalized
    finalized: bool,
}

impl ParquetBuilder {
    /// Create a new Parquet builder with Arrow schema
    pub fn new(
        schema: Arc<Schema>,
        options: ParquetWriteOptions,
        memory_limit_mb: u32,
    ) -> ConvertResult<Self> {
        // Validate row_group_size: determines memory-to-performance trade-off
        // - Minimum 64 rows: Too small groups cause excessive metadata overhead (per row group)
        // - Maximum 16384 rows: Larger groups optimize compression ratio and query performance
        //   (standard Parquet best practice for balanced I/O and memory usage)
        // Example: 1024 rows = good balance for typical data (recommended default)
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
            row_groups: Vec::new(),
            finalized: false,
        })
    }

    /// Add a row to the current batch
    /// Returns Some(Vec<u8>) if row group should be flushed, None otherwise
    pub fn add_row(&mut self, row: Vec<String>) -> ConvertResult<Option<Vec<u8>>> {
        if self.finalized {
            return Err(ConvertError::InternalError {
                reason: "Cannot add rows to finalized ParquetBuilder".to_string(),
            });
        }

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

        // Write to Parquet with footer (creates complete, valid Parquet file)
        let buffer = self.write_batch_to_parquet(&batch)?;

        self.batch_count += 1;
        self.estimated_memory = 0; // Reset after flush

        Ok(Some(buffer))
    }

    /// Write a single RecordBatch as a complete Parquet file with footer
    /// This ensures DuckDB Wasm can read each batch independently or merged
    fn write_batch_to_parquet(&mut self, batch: &RecordBatch) -> ConvertResult<Vec<u8>> {
        // Use ArrowWriter which handles complete Parquet file generation including footer
        // When finalized (calling finish()), it produces valid Parquet with footer
        let mut buffer = Vec::new();

        {
            let mut writer =
                ArrowWriter::try_new(&mut buffer, Arc::clone(&self.schema), None).map_err(|e| {
                    ConvertError::InternalError {
                        reason: format!("Failed to create ArrowWriter: {}", e),
                    }
                })?;

            writer
                .write(batch)
                .map_err(|e| ConvertError::InternalError {
                    reason: format!("Failed to write batch: {}", e),
                })?;

            // CRITICAL: finish() writes the Parquet footer and metadata
            // Without this, the file is incomplete and unreadable
            writer.finish().map_err(|e| ConvertError::InternalError {
                reason: format!("Failed to finalize Parquet: {}", e),
            })?;
            // writer is dropped here, releasing the mutable borrow on buffer
        }

        Ok(buffer)
    }

    /// Finalize: return the complete valid Parquet file
    /// DuckDB Wasm can directly load this file
    pub fn finalize(&mut self) -> ConvertResult<Vec<Vec<u8>>> {
        if self.finalized {
            return Err(ConvertError::InternalError {
                reason: "ParquetBuilder already finalized".to_string(),
            });
        }

        self.finalized = true;
        let mut results = Vec::new();

        // Flush any remaining rows
        if !self.current_batch.is_empty() {
            if let Some(chunk) = self.flush()? {
                results.push(chunk);
            }
        }

        // Return all row groups as valid Parquet files
        // Each is complete with magic, footer, and metadata
        // DuckDB Wasm receives ready-to-load Parquet data
        Ok(results)
    }

    /// Convert rows to Arrow arrays based on schema
    /// 
    /// # Data Type Handling
    /// This method automatically infers and converts CSV string values to appropriate types:
    /// - **Utf8**: Stored as-is (fallback for unparseable values)
    /// - **Int64**: Parsed from string representation; null if parsing fails
    /// - **Float64**: Parsed from string representation; null if parsing fails
    ///
    /// Rationale for type inference:
    /// - DuckDB Wasm CAN auto-cast types during queries (e.g., CAST string to int)
    /// - But pre-typing provides benefits:
    ///   * Query performance: No per-query conversion overhead
    ///   * Memory efficiency: Int64 uses less space than "123" as Utf8
    ///   * Faster aggregations: Numeric operations directly on typed data
    /// 
    /// Empty strings are converted to NULL for all types (standard behavior).
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
                        row.get(col_idx).and_then(|s| {
                            if s.is_empty() {
                                None
                            } else {
                                Some(s.as_str())
                            }
                        })
                    })
                    .collect();

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
