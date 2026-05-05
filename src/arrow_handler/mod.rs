pub mod builder;
/// Arrow IPC data processing modules
pub mod parser;

// Re-export commonly used functions
pub use builder::{
    SkuPredictionResult, build_anomaly_result, build_batch_regression_result, build_cluster_result,
    build_regression_result,
};
pub use parser::{parse_arrow_ipc, parse_batch_arrow_ipc};
