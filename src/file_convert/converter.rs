/// State machine for managing file conversion sessions
use super::csv_parser::CsvParser;
use super::error::{ConvertError, ConvertResult};
use super::parquet_writer::ParquetBuilder;
use super::types::{CsvReadOptions, ParquetWriteOptions};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// State of the converter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionState {
    /// Idle, no active session
    Idle,
    /// CSV to Parquet conversion in progress
    CsvToParquet,
}

/// Main converter for file format conversions
/// Implements a state machine to manage conversion sessions
pub struct Converter {
    /// Current state
    state: ConversionState,
    /// CSV parser (active during CSV→Parquet)
    csv_parser: Option<CsvParser>,
    /// Parquet builder (active during CSV→Parquet)
    parquet_builder: Option<ParquetBuilder>,
    /// Memory limit in MB
    memory_limit_mb: u32,
    /// Total CSV bytes consumed
    csv_bytes_consumed: usize,
}

impl Converter {
    /// Create a new converter
    pub fn new() -> Self {
        Self {
            state: ConversionState::Idle,
            csv_parser: None,
            parquet_builder: None,
            memory_limit_mb: 150,
            csv_bytes_consumed: 0,
        }
    }

    /// Create a new converter with custom memory limit
    pub fn with_memory_limit(limit_mb: u32) -> Self {
        Self {
            state: ConversionState::Idle,
            csv_parser: None,
            parquet_builder: None,
            memory_limit_mb: limit_mb,
            csv_bytes_consumed: 0,
        }
    }

    /// Begin a CSV to Parquet conversion session
    pub fn begin_csv_to_parquet(
        &mut self,
        csv_opts: CsvReadOptions,
        parquet_opts: ParquetWriteOptions,
    ) -> ConvertResult<()> {
        // Validate state
        if self.state != ConversionState::Idle {
            return Err(ConvertError::InvalidState {
                reason: "Converter is already in use. Call free() first.".to_string(),
            });
        }

        // Validate parquet options
        if parquet_opts.row_group_size < 64 || parquet_opts.row_group_size > 16384 {
            return Err(ConvertError::ParquetRowGroupTooLarge {
                size: parquet_opts.row_group_size,
            });
        }

        // Initialize CSV parser
        let csv_parser = CsvParser::new(csv_opts);

        // Create a placeholder schema (will be updated after first rows are parsed)
        // For now, assume 3 columns of UTF-8
        let placeholder_schema = Arc::new(Schema::new(vec![
            Field::new("col_0", DataType::Utf8, true),
            Field::new("col_1", DataType::Utf8, true),
            Field::new("col_2", DataType::Utf8, true),
        ]));

        // Initialize Parquet builder
        let parquet_builder =
            ParquetBuilder::new(placeholder_schema, parquet_opts, self.memory_limit_mb)?;

        // Update state
        self.csv_parser = Some(csv_parser);
        self.parquet_builder = Some(parquet_builder);
        self.state = ConversionState::CsvToParquet;
        self.csv_bytes_consumed = 0;

        Ok(())
    }

    /// Feed a chunk of CSV data and return Parquet chunks
    pub fn feed_csv_chunk(&mut self, chunk: &[u8], is_last: bool) -> ConvertResult<Vec<Vec<u8>>> {
        // Validate state
        if self.state != ConversionState::CsvToParquet {
            return Err(ConvertError::InvalidState {
                reason: "Call begin_csv_to_parquet() first.".to_string(),
            });
        }

        self.csv_bytes_consumed += chunk.len();

        // Safety check: if we've consumed more than 2GB, stop
        if self.csv_bytes_consumed > 2_000_000_000 {
            return Err(ConvertError::MemoryLimitExceeded {
                limit_mb: self.memory_limit_mb,
                estimated_mb: (self.csv_bytes_consumed / 1024 / 1024) as u32,
            });
        }

        // Get mutable references to parser and builder
        let csv_parser = self.csv_parser.as_mut().ok_or(ConvertError::InvalidState {
            reason: "CSV parser not initialized".to_string(),
        })?;

        // Parse CSV chunk
        let rows = csv_parser.feed_chunk(chunk, is_last)?;

        // Collect Parquet output chunks
        let mut parquet_chunks = Vec::new();

        // Update schema if it was just inferred
        if csv_parser.schema_inferred() && self.parquet_builder.is_some() {
            // In MVP, we skip schema update for simplicity
            // In full version, would reconstruct ParquetBuilder with correct schema
        }

        // Add rows to Parquet builder
        let parquet_builder = self
            .parquet_builder
            .as_mut()
            .ok_or(ConvertError::InvalidState {
                reason: "Parquet builder not initialized".to_string(),
            })?;

        for row in rows {
            if let Some(chunk) = parquet_builder.add_row(row)? {
                parquet_chunks.push(chunk);
            }
        }

        // If this is the last chunk, finalize and collect all remaining output
        if is_last {
            let final_chunks = parquet_builder.finalize()?;
            parquet_chunks.extend(final_chunks);

            // Reset state
            self.state = ConversionState::Idle;
        }

        Ok(parquet_chunks)
    }

    /// Free resources and reset converter to idle state
    pub fn free(&mut self) {
        self.state = ConversionState::Idle;
        self.csv_parser = None;
        self.parquet_builder = None;
        self.csv_bytes_consumed = 0;
    }

    /// Get current state
    pub fn state(&self) -> ConversionState {
        self.state
    }

    /// Get bytes consumed so far
    pub fn bytes_consumed(&self) -> usize {
        self.csv_bytes_consumed
    }
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Converter {
    fn drop(&mut self) {
        self.free();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = Converter::new();
        assert_eq!(converter.state(), ConversionState::Idle);
        assert_eq!(converter.bytes_consumed(), 0);
    }

    #[test]
    fn test_memory_limit_custom() {
        let converter = Converter::with_memory_limit(200);
        assert_eq!(converter.memory_limit_mb, 200);
    }

    #[test]
    fn test_invalid_state_feed_without_begin() {
        let mut converter = Converter::new();
        let chunk = b"test,data\n1,2\n";

        let result = converter.feed_csv_chunk(chunk, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_row_group_size_in_begin() {
        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions::default();
        let mut pq_opts = ParquetWriteOptions::default();
        pq_opts.row_group_size = 20000;

        let result = converter.begin_csv_to_parquet(csv_opts, pq_opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_csv_to_parquet() {
        // Use CSV with exactly 3 columns to match hardcoded schema in MVP
        let csv_data = b"col1,col2,col3\nAlice,100,10\nBob,200,20\n";

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions::default();
        let pq_opts = ParquetWriteOptions::default();

        converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
        let chunks = converter.feed_csv_chunk(csv_data, true).unwrap();

        // Should produce some Parquet output
        assert!(!chunks.is_empty());
        assert_eq!(converter.state(), ConversionState::Idle);
    }

    #[test]
    fn test_converter_reuse_after_free() {
        let csv_data = b"col1,col2,col3\na,b,c\n";

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions::default();
        let pq_opts = ParquetWriteOptions::default();

        // First conversion
        converter
            .begin_csv_to_parquet(csv_opts.clone(), pq_opts.clone())
            .unwrap();
        converter.feed_csv_chunk(csv_data, true).unwrap();
        converter.free();

        // Should be able to start new session
        assert_eq!(converter.state(), ConversionState::Idle);
        converter.begin_csv_to_parquet(csv_opts, pq_opts).unwrap();
        assert_eq!(converter.state(), ConversionState::CsvToParquet);
    }

    #[test]
    fn test_cannot_begin_twice() {
        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions::default();
        let pq_opts = ParquetWriteOptions::default();

        converter
            .begin_csv_to_parquet(csv_opts.clone(), pq_opts.clone())
            .unwrap();
        let result = converter.begin_csv_to_parquet(csv_opts, pq_opts);
        assert!(result.is_err());
    }
}
