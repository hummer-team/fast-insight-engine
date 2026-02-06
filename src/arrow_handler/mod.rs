pub mod builder;
/// Arrow IPC data processing modules
pub mod parser;

// Re-export commonly used functions
pub use builder::{build_anomaly_result, build_cluster_result, build_regression_result};
pub use parser::parse_arrow_ipc;
