/// State machine for managing file conversion sessions
use super::csv_parser::CsvParser;
use super::error::{ConvertError, ConvertResult};
use super::excel_loader::ExcelParser;
use super::parquet_writer::ParquetBuilder;
use super::types::{CsvReadOptions, ExcelLoadOptions, ParquetWriteOptions};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// State of the converter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionState {
    /// Idle, no active session
    Idle,
    /// CSV to Parquet conversion in progress
    CsvToParquet,
    /// Excel to Parquet conversion in progress
    ExcelToParquet,
}

/// Main converter for file format conversions
/// Implements a state machine to manage conversion sessions
pub struct Converter {
    /// Current state
    state: ConversionState,
    /// CSV parser (active during CSV→Parquet)
    csv_parser: Option<CsvParser>,
    /// Excel parser (active during Excel→Parquet)
    excel_parser: Option<ExcelParser>,
    /// Parquet builder (active during conversions)
    parquet_builder: Option<ParquetBuilder>,
    /// Optional schema hint for columns (for strict type conversion)
    schema_hint: Option<super::types::SchemaHint>,
    /// Excel load options (stored for use in feed_excel_chunk)
    excel_opts: Option<ExcelLoadOptions>,
    /// Memory limit in MB
    memory_limit_mb: u32,
    /// Total bytes consumed (CSV or Excel)
    bytes_consumed: usize,
}

impl Converter {
    /// Create a new converter
    pub fn new() -> Self {
        Self {
            state: ConversionState::Idle,
            csv_parser: None,
            excel_parser: None,
            parquet_builder: None,
            schema_hint: None,
            excel_opts: None,
            memory_limit_mb: 150,
            bytes_consumed: 0,
        }
    }

    /// Create a new converter with custom memory limit
    pub fn with_memory_limit(limit_mb: u32) -> Self {
        Self {
            state: ConversionState::Idle,
            csv_parser: None,
            excel_parser: None,
            parquet_builder: None,
            schema_hint: None,
            excel_opts: None,
            memory_limit_mb: limit_mb,
            bytes_consumed: 0,
        }
    }

    /// Begin an Excel to Parquet conversion session
    pub fn begin_excel_to_parquet(
        &mut self,
        excel_opts: ExcelLoadOptions,
        parquet_opts: ParquetWriteOptions,
        schema_hint: Option<&super::types::SchemaHint>,
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

        // Create a placeholder schema (will be updated after first rows are parsed)
        let placeholder_schema = Arc::new(Schema::new(vec![
            Field::new("col_0", DataType::Utf8, true),
            Field::new("col_1", DataType::Utf8, true),
            Field::new("col_2", DataType::Utf8, true),
        ]));

        // Initialize Parquet builder
        let parquet_builder =
            ParquetBuilder::new(placeholder_schema, parquet_opts, self.memory_limit_mb)?;

        // Update state (excel parser will be created when feed_excel_chunk is called with actual data)
        self.excel_parser = None; // Will be populated on first feed
        self.parquet_builder = Some(parquet_builder);
        self.schema_hint = schema_hint.cloned();
        self.excel_opts = Some(excel_opts);
        self.state = ConversionState::ExcelToParquet;
        self.bytes_consumed = 0;

        Ok(())
    }

    /// Feed a chunk of Excel data and return Parquet chunks
    pub fn feed_excel_chunk(&mut self, chunk: &[u8], is_last: bool) -> ConvertResult<Vec<Vec<u8>>> {
        // Validate state
        if self.state != ConversionState::ExcelToParquet {
            return Err(ConvertError::InvalidState {
                reason: "Call begin_excel_to_parquet() first.".to_string(),
            });
        }

        self.bytes_consumed += chunk.len();

        // Safety check: if we've consumed more than 2GB, stop
        if self.bytes_consumed > 2_000_000_000 {
            return Err(ConvertError::MemoryLimitExceeded {
                limit_mb: self.memory_limit_mb,
                estimated_mb: (self.bytes_consumed / 1024 / 1024) as u32,
            });
        }

        // Initialize Excel parser on first call using stored options
        if self.excel_parser.is_none() {
            let excel_opts = self.excel_opts.clone().unwrap_or_default();
            let excel_parser = ExcelParser::new(chunk, &excel_opts)?;
            self.excel_parser = Some(excel_parser);
        }

        // Get mutable reference to Excel parser
        let excel_parser = self
            .excel_parser
            .as_mut()
            .ok_or(ConvertError::InvalidState {
                reason: "Excel parser not initialized".to_string(),
            })?;

        // Rebuild Parquet schema from Excel header (same pattern as CSV)
        // Must happen before any rows are added to the builder
        if excel_parser.schema_inferred() && self.parquet_builder.is_some() {
            let mut needs_rebuild = false;

            if let Some(pb) = self.parquet_builder.as_ref() {
                let fields = pb.arrow_schema().fields();
                needs_rebuild = !fields.is_empty()
                    && fields[0].name().starts_with("col_")
                    && fields
                        .iter()
                        .enumerate()
                        .all(|(i, f)| f.name() == &format!("col_{}", i));
            }

            if needs_rebuild {
                let col_names = excel_parser.inferred_columns().cloned().unwrap_or_default();

                let opts = if let Some(pb) = self.parquet_builder.take() {
                    ParquetWriteOptions {
                        row_group_size: pb.row_group_size_value(),
                        compression: pb.parquet_options().compression,
                    }
                } else {
                    ParquetWriteOptions::default()
                };

                let new_builder = ParquetBuilder::new_with_optional_schema(
                    col_names,
                    self.schema_hint.as_ref(),
                    opts,
                    self.memory_limit_mb,
                )?;

                self.parquet_builder = Some(new_builder);
            }
        }

        // Get rows (this is simplified - real version would handle streaming)
        let rows = excel_parser.get_batch(1024);

        // Collect Parquet output chunks
        let mut parquet_chunks = Vec::new();

        // Add rows to Parquet builder
        let parquet_builder = self
            .parquet_builder
            .as_mut()
            .ok_or(ConvertError::InvalidState {
                reason: "Parquet builder not initialized".to_string(),
            })?;

        for row in rows {
            parquet_builder.add_row(row)?;

            // Check if we should flush (this is simplified)
            if parquet_builder.current_row_count() % 1024 == 0 {
                if let Some(pq_chunk) = parquet_builder.flush()? {
                    parquet_chunks.push(pq_chunk);
                }
            }
        }

        // If this is the last chunk, finalize
        if is_last {
            let final_chunks = parquet_builder.finalize()?;
            parquet_chunks.extend(final_chunks);
            self.state = ConversionState::Idle;
        }

        Ok(parquet_chunks)
    }

    /// Begin a CSV to Parquet conversion session
    pub fn begin_csv_to_parquet(
        &mut self,
        csv_opts: CsvReadOptions,
        parquet_opts: ParquetWriteOptions,
        schema_hint: Option<&super::types::SchemaHint>,
    ) -> ConvertResult<()> {
        // Validate state
        if self.state != ConversionState::Idle {
            return Err(ConvertError::InvalidState {
                reason: "Converter is already in use. Call free() first.".to_string(),
            });
        }

        // Validate parquet options: row_group_size determines memory-to-performance trade-off
        // - Minimum 64 rows: Too small groups cause excessive metadata overhead
        // - Maximum 16384 rows: Larger groups optimize compression ratio and query performance
        //   (standard Parquet practice for balanced I/O and memory usage)
        if parquet_opts.row_group_size < 64 || parquet_opts.row_group_size > 16384 {
            return Err(ConvertError::ParquetRowGroupTooLarge {
                size: parquet_opts.row_group_size,
            });
        }

        // Initialize CSV parser
        let csv_parser = CsvParser::new(csv_opts);

        // Store schema hint for later use in feed_csv_chunk
        // It will be applied after column names are inferred from CSV header
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
        self.schema_hint = schema_hint.cloned();
        self.state = ConversionState::CsvToParquet;
        self.bytes_consumed = 0;

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

        self.bytes_consumed += chunk.len();

        // Safety check: if we've consumed more than 2GB, stop
        if self.bytes_consumed > 2_000_000_000 {
            return Err(ConvertError::MemoryLimitExceeded {
                limit_mb: self.memory_limit_mb,
                estimated_mb: (self.bytes_consumed / 1024 / 1024) as u32,
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

        // Update schema based on CSV header when first inferred
        if csv_parser.schema_inferred() && self.parquet_builder.is_some() {
            let mut needs_rebuild = false;

            // Check if builder still has placeholder schema
            if let Some(pb) = self.parquet_builder.as_ref() {
                let fields = pb.arrow_schema().fields();
                needs_rebuild = fields.len() > 0
                    && fields[0].name().starts_with("col_")
                    && fields
                        .iter()
                        .enumerate()
                        .all(|(i, f)| f.name() == &format!("col_{}", i));
            }

            if needs_rebuild {
                // Get the inferred column names
                let col_names = csv_parser
                    .inferred_columns()
                    .map(|v| v.clone())
                    .unwrap_or_default();

                // Extract parquet options from current builder
                let opts = if let Some(pb) = self.parquet_builder.take() {
                    ParquetWriteOptions {
                        row_group_size: pb.row_group_size_value(),
                        compression: pb.parquet_options().compression,
                    }
                } else {
                    ParquetWriteOptions::default()
                };

                // Create new builder with correct schema
                let new_builder = ParquetBuilder::new_with_optional_schema(
                    col_names,
                    self.schema_hint.as_ref(),
                    opts,
                    self.memory_limit_mb,
                )?;

                self.parquet_builder = Some(new_builder);
            }
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
        self.excel_parser = None;
        self.parquet_builder = None;
        self.schema_hint = None;
        self.excel_opts = None;
        self.bytes_consumed = 0;
    }

    /// Get current state
    pub fn state(&self) -> ConversionState {
        self.state
    }

    /// Get bytes consumed so far
    pub fn bytes_consumed(&self) -> usize {
        self.bytes_consumed
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

        let result = converter.begin_csv_to_parquet(csv_opts, pq_opts, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_csv_to_parquet() {
        // Use CSV with exactly 3 columns to match hardcoded schema in MVP
        let csv_data = b"col1,col2,col3\nAlice,100,10\nBob,200,20\n";

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions::default();
        let pq_opts = ParquetWriteOptions::default();

        converter
            .begin_csv_to_parquet(csv_opts, pq_opts, None)
            .unwrap();
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
            .begin_csv_to_parquet(csv_opts.clone(), pq_opts.clone(), None)
            .unwrap();
        converter.feed_csv_chunk(csv_data, true).unwrap();
        converter.free();

        // Should be able to start new session
        assert_eq!(converter.state(), ConversionState::Idle);
        converter
            .begin_csv_to_parquet(csv_opts, pq_opts, None)
            .unwrap();
        assert_eq!(converter.state(), ConversionState::CsvToParquet);
    }

    #[test]
    fn test_cannot_begin_twice() {
        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions::default();
        let pq_opts = ParquetWriteOptions::default();

        converter
            .begin_csv_to_parquet(csv_opts.clone(), pq_opts.clone(), None)
            .unwrap();
        let result = converter.begin_csv_to_parquet(csv_opts, pq_opts, None);
        assert!(result.is_err());
    }
}
