use wasm_bindgen::prelude::*;

pub mod arrow_handler;
pub mod insight_core;
pub mod utils;

use arrow_handler::{
    build_anomaly_result, build_cluster_result, build_regression_result, parse_arrow_ipc,
};
use insight_core::{run_isolation_forest, run_kmeans, run_linear_regression};

/// Detect anomalous orders using Isolation Forest
///
/// # Arguments
/// * `data` - Arrow IPC Stream format bytes (Uint8Array from TypeScript)
/// * `threshold` - Anomaly threshold in [0, 1], scores >= threshold are anomalous
///
/// # Returns
/// * `Ok(Uint8Array)` - Arrow IPC result with order_id, abnormal_score, is_abnormal
/// * `Err(JsError)` - Error message
#[wasm_bindgen]
pub async fn detect_order_anomalies(data: &[u8], threshold: f64) -> Result<Vec<u8>, JsError> {
    // Parse Arrow IPC input
    let parsed = parse_arrow_ipc(data)?;

    // Run Isolation Forest
    let (scores, labels) = run_isolation_forest(parsed.features, threshold)?;

    // Build Arrow IPC result
    let result = build_anomaly_result(parsed.order_ids, scores, labels)?;

    Ok(result)
}

/// Segment customers/orders using K-Means clustering
///
/// # Arguments
/// * `data` - Arrow IPC Stream format bytes (Uint8Array from TypeScript)
/// * `n_clusters` - Number of clusters
///
/// # Returns
/// * `Ok(Uint8Array)` - Arrow IPC result with order_id, cluster_id
/// * `Err(JsError)` - Error message
#[wasm_bindgen]
pub async fn segment_customer_orders(data: &[u8], n_clusters: usize) -> Result<Vec<u8>, JsError> {
    // Parse Arrow IPC input
    let parsed = parse_arrow_ipc(data)?;

    // Run K-Means clustering
    let cluster_ids = run_kmeans(parsed.features, n_clusters)?;

    // Build Arrow IPC result
    let result = build_cluster_result(parsed.order_ids, cluster_ids)?;

    Ok(result)
}

/// Predict future inventory demand using Linear Regression
///
/// # Arguments
/// * `data` - Arrow IPC Stream format bytes with x (time/index) and y (demand) columns
/// * `predict_steps` - Number of future time steps to predict
///
/// # Returns
/// * `Ok(Uint8Array)` - Arrow IPC result with predictions
/// * `Err(JsError)` - Error message
#[wasm_bindgen]
pub async fn predict_inventory_demand(
    data: &[u8],
    predict_steps: usize,
) -> Result<Vec<u8>, JsError> {
    // Parse Arrow IPC input
    let parsed = parse_arrow_ipc(data)?;

    // Extract X (first feature column) and Y (order_id repurposed as target)
    // Note: In real usage, you'd pass proper x/y data via Arrow schema
    let x = parsed
        .features
        .column(0)
        .to_owned()
        .insert_axis(ndarray::Axis(1));
    let y = ndarray::Array1::from_vec(parsed.order_ids.iter().map(|&id| id as f64).collect());

    // Run Linear Regression
    let predictions = run_linear_regression(x, y, predict_steps)?;

    // Build Arrow IPC result
    let result = build_regression_result(predictions)?;

    Ok(result)
}

/// Get Wasm module version for compatibility checking
///
/// # Returns
/// * Version string (e.g., "0.1.0")
#[wasm_bindgen]
pub fn get_wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
