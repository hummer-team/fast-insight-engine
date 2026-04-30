pub mod converter;
pub mod csv_parser;
/// File format conversion module
///
/// Supports CSV ↔ Parquet conversion with streaming input/output
/// in browser environment (WASM).
pub mod error;
pub mod parquet_writer;
pub mod types;

// Re-export public API
pub use converter::{ConversionState, Converter};
pub use error::{ConvertError, ConvertResult};
pub use types::{
    CsvReadOptions, CsvWriteOptions, ExcelLoadOptions, NullHandling, ParquetCompression,
    ParquetWriteOptions, SheetSelector,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Verify all public types are accessible
        let _csv_opts = CsvReadOptions::default();
        let _pq_opts = ParquetWriteOptions::default();
        let _excel_opts = ExcelLoadOptions::default();
    }
}
