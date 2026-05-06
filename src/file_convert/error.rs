/// Error types for file conversion operations
use std::fmt;

/// Errors that can occur during file conversion
#[derive(Debug)]
pub enum ConvertError {
    /// Invalid CSV format (RFC 4180 violation)
    InvalidCsv { line: usize, reason: String },
    /// CSV has no header when hasHeader=true
    CsvMissingHeader,
    /// Excel file size exceeds limit (> 1GB)
    ExcelFileTooLarge { size_mb: u64 },
    /// Excel file is encrypted or write-protected
    ExcelEncrypted,
    /// Failed to parse Excel file
    ExcelLoadFailed { reason: String },
    /// Row group size out of valid range [64, 16384]
    ParquetRowGroupTooLarge { size: usize },
    /// Memory limit exceeded during conversion
    MemoryLimitExceeded { limit_mb: u32, estimated_mb: u32 },
    /// Invalid state (e.g., feed without begin)
    InvalidState { reason: String },
    /// Invalid schema or schema hint
    InvalidSchema { reason: String },
    /// Type conversion failed (when schema hint is provided)
    TypeConversionFailed {
        column: String,
        value: String,
        target_type: String,
    },
    /// Internal error (should not happen)
    InternalError { reason: String },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCsv { line, reason } => {
                write!(f, "InvalidCsv: Line {}: {}", line, reason)
            }
            Self::CsvMissingHeader => {
                write!(
                    f,
                    "CsvMissingHeader: Expected header row but file is empty. \
                     Fix: Set hasHeader=false or ensure first row contains headers."
                )
            }
            Self::ExcelFileTooLarge { size_mb } => {
                write!(
                    f,
                    "ExcelFileTooLarge: File is {} MB, limit is 1000 MB. \
                     Fix: Convert to CSV first or split file.",
                    size_mb
                )
            }
            Self::ExcelEncrypted => {
                write!(
                    f,
                    "ExcelEncrypted: File is encrypted or write-protected. \
                     Fix: Remove protection in Excel and re-save."
                )
            }
            Self::ExcelLoadFailed { reason } => {
                write!(
                    f,
                    "ExcelLoadFailed: {}. \
                     Fix: Re-save file in Microsoft Excel.",
                    reason
                )
            }
            Self::ParquetRowGroupTooLarge { size } => {
                write!(
                    f,
                    "ParquetRowGroupTooLarge: Row group size {} exceeds maximum 16384. \
                     Fix: Use rowGroupSize <= 16384.",
                    size
                )
            }
            Self::MemoryLimitExceeded {
                limit_mb,
                estimated_mb,
            } => {
                write!(
                    f,
                    "MemoryLimitExceeded: Estimated {} MB exceeds limit {} MB. \
                     Fix: Reduce rowGroupSize or split input into smaller chunks.",
                    estimated_mb, limit_mb
                )
            }
            Self::InvalidState { reason } => {
                write!(f, "InvalidState: {}. Fix: Check call sequence.", reason)
            }
            Self::InvalidSchema { reason } => {
                write!(
                    f,
                    "InvalidSchema: {}. Fix: Verify schema hint matches CSV.",
                    reason
                )
            }
            Self::TypeConversionFailed {
                column,
                value,
                target_type,
            } => {
                write!(
                    f,
                    "TypeConversionFailed: Cannot convert '{}' to {} for column '{}'. \
                     Fix: Verify schemaHint types match CSV data.",
                    value, target_type, column
                )
            }
            Self::InternalError { reason } => {
                write!(f, "InternalError: {}. Please report this issue.", reason)
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Convert Arrow errors to ConvertError
impl From<arrow::error::ArrowError> for ConvertError {
    fn from(err: arrow::error::ArrowError) -> Self {
        ConvertError::InternalError {
            reason: format!("Arrow error: {}", err),
        }
    }
}

/// Convenience type alias for conversion results
pub type ConvertResult<T> = Result<T, ConvertError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_csv_display() {
        let err = ConvertError::InvalidCsv {
            line: 42,
            reason: "Unmatched quote".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Line 42"));
        assert!(msg.contains("Unmatched quote"));
    }

    #[test]
    fn test_memory_limit_display() {
        let err = ConvertError::MemoryLimitExceeded {
            limit_mb: 150,
            estimated_mb: 300,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("300 MB"));
        assert!(msg.contains("150 MB"));
    }
}
