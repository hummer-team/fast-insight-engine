/// Utility modules for error handling and type conversions
pub mod error;
pub mod type_convert;
pub mod scaling;

// Re-export commonly used types
pub use error::AnalysisError;
pub use type_convert::{normalize_scores, validate_threshold};
pub use scaling::{min_max_scale, standard_scale, ScalingMethod};
