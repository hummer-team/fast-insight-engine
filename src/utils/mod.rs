/// Utility modules for error handling and type conversions
pub mod error;
pub mod scaling;
pub mod type_convert;

// Re-export commonly used types
pub use error::AnalysisError;
pub use scaling::{ScalingMethod, min_max_scale, standard_scale};
pub use type_convert::{normalize_scores, validate_threshold};
