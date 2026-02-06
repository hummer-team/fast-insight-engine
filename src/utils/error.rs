use std::fmt;

/// Custom error type for analysis operations
#[derive(Debug, Clone)]
pub enum AnalysisError {
    /// Validation errors (e.g., invalid threshold, empty dataset)
    ValidationError(String),
    /// Arrow-related errors (parsing, schema mismatch)
    ArrowError(String),
    /// Model training/prediction errors
    ModelError(String),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisError::ValidationError(msg) => write!(f, "ValidationError: {}", msg),
            AnalysisError::ArrowError(msg) => write!(f, "ArrowError: {}", msg),
            AnalysisError::ModelError(msg) => write!(f, "ModelError: {}", msg),
        }
    }
}

impl std::error::Error for AnalysisError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AnalysisError::ValidationError("test error".to_string());
        assert_eq!(err.to_string(), "ValidationError: test error");

        let err = AnalysisError::ArrowError("arrow test".to_string());
        assert_eq!(err.to_string(), "ArrowError: arrow test");

        let err = AnalysisError::ModelError("model test".to_string());
        assert_eq!(err.to_string(), "ModelError: model test");
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<AnalysisError>();
        assert_sync::<AnalysisError>();
    }
}
