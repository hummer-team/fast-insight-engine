/// Configuration types for file conversion operations

/// Options for reading CSV files
#[derive(Clone, Debug)]
pub struct CsvReadOptions {
    /// Delimiter character: ',' | '\t' | '|' | ';'
    pub delimiter: u8,
    /// Whether first row contains column headers
    pub has_header: bool,
    /// How to handle null/empty values: "null" | "empty_string"
    pub null_handling: NullHandling,
}

/// Options for writing CSV files
#[derive(Clone, Debug)]
pub struct CsvWriteOptions {
    /// Delimiter character: ',' | '\t' | '|' | ';'
    pub delimiter: u8,
    /// Include header row in output
    pub include_header: bool,
}

/// Options for writing Parquet files
#[derive(Clone, Debug)]
pub struct ParquetWriteOptions {
    /// Number of rows per row group (range: 64-16384, default: 1024)
    pub row_group_size: usize,
    /// Compression algorithm
    pub compression: ParquetCompression,
}

/// Options for loading Excel files
#[derive(Clone, Debug)]
pub struct ExcelLoadOptions {
    /// Sheet name or index (0-based). None = first sheet (index 0)
    pub sheet: Option<SheetSelector>,
    /// Maximum string table size in bytes (default: 100MB)
    pub max_string_table_bytes: Option<u64>,
    /// Whether the first row contains column headers (default: true)
    pub has_header: bool,
}

/// Sheet selector: by name or index
#[derive(Clone, Debug)]
pub enum SheetSelector {
    /// Select sheet by name
    ByName(String),
    /// Select sheet by 0-based index
    ByIndex(usize),
}

/// How to handle null or empty values during CSV parsing
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum NullHandling {
    /// Treat empty fields as null
    Null,
    /// Treat empty fields as empty strings
    EmptyString,
}

/// Parquet compression algorithm
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum ParquetCompression {
    /// No compression
    Uncompressed,
    /// Snappy compression
    Snappy,
    /// Zstandard compression
    Zstd,
}

/// Schema hint for CSV/Excel conversion - defines expected column names and types
/// Optional metadata that enables strict type conversion during file→Parquet
///
/// When provided:
/// - Each column will be converted to its specified type
/// - Parse failures return ConvertError (fail-fast approach)
/// - Must match file column count exactly
///
/// When not provided:
/// - All columns remain as Utf8 (raw string values)
/// - DuckDB Wasm handles type inference during loading
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaHint {
    /// Column definitions (order must match file column order)
    pub columns: Vec<ColumnDef>,
}

/// Column type definition for schema hints
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnDef {
    /// Column name (will override file header or col_N)
    pub name: String,
    /// Data type: 0=Utf8, 1=Int64, 2=Float64, 3=Boolean, others=Utf8 (default)
    pub type_id: u8,
}

impl SchemaHint {
    /// Create a new schema hint with given columns
    pub fn new(columns: Vec<ColumnDef>) -> Self {
        Self { columns }
    }

    /// Validate schema hint against column count
    pub fn validate(&self, expected_column_count: usize) -> Result<(), String> {
        if self.columns.is_empty() {
            return Err("SchemaHint must have at least one column".to_string());
        }
        if self.columns.len() != expected_column_count {
            return Err(format!(
                "SchemaHint column count {} does not match CSV column count {}",
                self.columns.len(),
                expected_column_count
            ));
        }
        Ok(())
    }
}

impl ColumnDef {
    /// Create a new column definition
    pub fn new(name: String, type_id: u8) -> Self {
        Self { name, type_id }
    }
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            null_handling: NullHandling::Null,
        }
    }
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            include_header: true,
        }
    }
}

impl Default for ParquetWriteOptions {
    fn default() -> Self {
        Self {
            row_group_size: 1024,
            compression: ParquetCompression::Snappy,
        }
    }
}

impl Default for ExcelLoadOptions {
    fn default() -> Self {
        Self {
            sheet: None,
            max_string_table_bytes: Some(100_000_000), // 100 MB default
            has_header: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_csv_read_options() {
        let opts = CsvReadOptions::default();
        assert_eq!(opts.delimiter, b',');
        assert!(opts.has_header);
        assert_eq!(opts.null_handling, NullHandling::Null);
    }

    #[test]
    fn test_default_parquet_write_options() {
        let opts = ParquetWriteOptions::default();
        assert_eq!(opts.row_group_size, 1024);
        assert_eq!(opts.compression, ParquetCompression::Snappy);
    }

    #[test]
    fn test_default_excel_load_options() {
        let opts = ExcelLoadOptions::default();
        assert!(opts.sheet.is_none());
        assert_eq!(opts.max_string_table_bytes, Some(100_000_000));
    }
}
