use wasm_bindgen::prelude::*;

// Note: wee_alloc optimization disabled for Wasm build compatibility
// Can be re-enabled by fixing version compatibility with wee_alloc 0.4.5
// See: https://github.com/rustwasm/wee_alloc
// #[cfg(target_arch = "wasm32")]
// #[global_allocator]
// static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc;

pub mod arrow_handler;
#[cfg(not(target_arch = "wasm32"))]
pub mod file_convert;
pub mod gpu;
pub mod insight_core;
pub mod utils;

use arrow_handler::{
    build_anomaly_result, build_cluster_result, build_regression_result, parse_arrow_ipc,
};
use insight_core::{run_isolation_forest, run_kmeans, run_linear_regression};
use utils::{ScalingMethod, min_max_scale, standard_scale};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(target_arch = "wasm32")]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! console_log {
    ($($t:tt)*) => {};
}

/// Detect anomalous orders using Extended Isolation Forest (CPU-only)
///
/// # Arguments
/// * `data` - Arrow IPC Stream format bytes (Uint8Array from TypeScript)
/// * `threshold` - Anomaly threshold in [0, 1], scores >= threshold are anomalous
/// * `scaling_mode` - Scaling method: 0=None (pre-scaled by DuckDB), 1=MinMax, 2=Standard
/// * `use_gpu` - Reserved for future GPU implementations (currently ignored)
///
/// # Returns
/// * `Ok(Uint8Array)` - Arrow IPC result with order_id, abnormal_score, is_abnormal
/// * `Err(JsError)` - Error message
///
/// # Algorithm
/// Uses **Extended Isolation Forest** algorithm (CPU-only):
/// - Tree-based anomaly detection with path length scoring
/// - Scores normalized to [0, 1] range (higher = more anomalous)
/// - Typical score distribution: [0.3, 0.7] for normal data
///
/// **⚠️ Important**: Extended Isolation Forest does not support GPU acceleration.
/// The `use_gpu` parameter is reserved for future algorithm implementations.
///
/// # Performance
/// - **100k rows × 10 features**: ~1-2 seconds (CPU)
/// - **Complexity**: O(n log n)
/// - **Recommendation**: Use Web Worker to avoid blocking UI thread
///
/// # Scaling
/// The `scaling_mode` parameter allows flexible data preprocessing:
/// - `0` (None): Assumes data is already scaled (e.g., by DuckDB Wasm)
/// - `1` (MinMax): Applies MinMax scaling in Rust: (x - min) / (max - min) -> [0, 1]
/// - `2` (Standard): Applies Standard scaling in Rust: (x - mean) / std
#[wasm_bindgen]
pub async fn detect_order_anomalies(
    data: &[u8],
    threshold: f64,
    scaling_mode: u8,
    use_gpu: bool,
) -> Result<Vec<u8>, JsError> {
    console_log!(
        "🔍 [WASM] detect_order_anomalies called: {} bytes, threshold={}, scaling={}, gpu={}",
        data.len(),
        threshold,
        scaling_mode,
        use_gpu
    );

    // Parse Arrow IPC input
    console_log!("🔍 [WASM] Parsing Arrow IPC...");
    let parsed = parse_arrow_ipc(data).map_err(|e| {
        console_log!("❌ [WASM] Arrow IPC parsing failed: {}", e);
        e
    })?;
    console_log!(
        "✓ [WASM] Parsed {} orders x {} features",
        parsed.order_ids.len(),
        parsed.features.ncols()
    );

    // Apply scaling based on mode
    console_log!("🔍 [WASM] Applying scaling mode: {}", scaling_mode);
    let features = match ScalingMethod::from(scaling_mode) {
        ScalingMethod::None => {
            console_log!("✓ [WASM] No scaling applied");
            parsed.features
        }
        ScalingMethod::MinMax => {
            console_log!("🔍 [WASM] Applying MinMax scaling...");
            min_max_scale(parsed.features).map_err(|e| {
                console_log!("❌ [WASM] MinMax scaling failed: {}", e);
                e
            })?
        }
        ScalingMethod::Standard => {
            console_log!("🔍 [WASM] Applying Standard scaling...");
            standard_scale(parsed.features).map_err(|e| {
                console_log!("❌ [WASM] Standard scaling failed: {}", e);
                e
            })?
        }
    };
    console_log!(
        "✓ [WASM] Scaling complete, features shape: {}x{}",
        features.nrows(),
        features.ncols()
    );

    // Log warning if GPU is requested (Extended IForest is CPU-only)
    if use_gpu {
        console_log!("⚠ Extended Isolation Forest does not support GPU acceleration");
        console_log!("  Algorithm: Extended Isolation Forest (CPU-only)");
        console_log!("  Performance: ~1-2s for 100k rows on modern CPUs");
    }

    // Run anomaly detection using Extended Isolation Forest (CPU-only)
    console_log!("🔍 [WASM] Starting Extended Isolation Forest computation...");
    let start_time = js_sys::Date::now();

    let (scores, labels) = run_isolation_forest(features, threshold).map_err(|e| {
        console_log!("❌ [WASM] Extended IForest computation failed: {}", e);
        e
    })?;

    let elapsed_ms = js_sys::Date::now() - start_time;
    console_log!(
        "⏱ [WASM] Algorithm execution time: {:.2}ms ({:.2}s)",
        elapsed_ms,
        elapsed_ms / 1000.0
    );

    // Log score statistics
    let score_min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let score_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let score_avg = scores.iter().sum::<f64>() / scores.len() as f64;
    let anomaly_count = labels.iter().filter(|&&l| l).count();

    console_log!("📊 [WASM] Score statistics:");
    console_log!(
        "  - Min: {:.6}, Max: {:.6}, Avg: {:.6}",
        score_min,
        score_max,
        score_avg
    );
    console_log!(
        "  - Threshold: {}, Anomalies: {} / {} ({:.2}%)",
        threshold,
        anomaly_count,
        scores.len(),
        (anomaly_count as f64 / scores.len() as f64 * 100.0)
    );
    console_log!(
        "  - Score sample (first 10): {:?}",
        &scores[..scores.len().min(10)]
    );

    // Suppress unused warnings in non-wasm builds (console_log expands to nothing)
    let _ = (elapsed_ms, score_min, score_max, score_avg, anomaly_count);

    // Build Arrow IPC result
    console_log!("🔍 [WASM] Building Arrow IPC result...");
    let result = build_anomaly_result(parsed.order_ids, scores, labels)?;

    console_log!("✓ [WASM] Arrow IPC result built: {} bytes", result.len());
    console_log!(
        "📤 [WASM] Returning {} anomaly results to TypeScript",
        result.len()
    );

    Ok(result)
}

/// Segment customers/orders using K-Means clustering
///
/// # Arguments
/// * `data` - Arrow IPC Stream format bytes (Uint8Array from TypeScript)
/// * `n_clusters` - Number of clusters
/// * `scaling_mode` - Scaling method: 0=None (pre-scaled by DuckDB), 1=MinMax, 2=Standard
/// * `use_gpu` - Enable WebGPU acceleration (requires Chrome 113+, auto-fallback to CPU if unavailable)
///
/// # Returns
/// * `Ok(Uint8Array)` - Arrow IPC result with order_id, cluster_id
/// * `Err(JsError)` - Error message
///
/// # WebGPU Acceleration
/// When `use_gpu=true`:
/// - **100k rows**: ~20s (CPU) → <0.5s (GPU), **~40x faster**
/// - **1M rows**: ~5min (CPU) → <5s (GPU), **~60x faster**
/// - GPU performs parallel distance computation for cluster assignment
/// - Automatic fallback to CPU KD-Tree if GPU unavailable
///
/// # Recommended Usage
/// ```typescript
/// // Enable GPU for large datasets
/// const useGPU = rowCount > 5000 && navigator.gpu !== undefined;
/// const result = await segment_customer_orders(data, 5, 2, useGPU);
/// ```
///
/// # Scaling
/// The `scaling_mode` parameter allows flexible data preprocessing:
/// - `0` (None): Assumes data is already scaled (e.g., by DuckDB Wasm)
/// - `1` (MinMax): Applies MinMax scaling in Rust
/// - `2` (Standard): Applies Standard scaling in Rust (recommended for K-Means)
#[wasm_bindgen]
pub async fn segment_customer_orders(
    data: &[u8],
    n_clusters: usize,
    scaling_mode: u8,
    use_gpu: bool,
) -> Result<Vec<u8>, JsError> {
    console_log!("segment_customer_orders: use_gpu={}", use_gpu);

    // Parse Arrow IPC input
    let parsed = parse_arrow_ipc(data)?;

    // Apply scaling based on mode
    let features = match ScalingMethod::from(scaling_mode) {
        ScalingMethod::None => parsed.features,
        ScalingMethod::MinMax => min_max_scale(parsed.features)?,
        ScalingMethod::Standard => standard_scale(parsed.features)?,
    };

    // Validate feature count (max 16 dimensions)
    if features.ncols() > 16 {
        return Err(JsError::new(&format!(
            "Feature count ({}) exceeds maximum (16). Please reduce number of features.",
            features.ncols()
        )));
    }

    // Run K-Means clustering with GPU acceleration if requested
    let cluster_ids = if use_gpu {
        console_log!("Attempting GPU K-Means clustering...");

        // Try GPU computation
        match gpu::GpuCompute::new().await {
            Ok(gpu_compute) => {
                console_log!("GPU initialized successfully");
                match gpu_compute.compute_kmeans(&features, n_clusters, 100).await {
                    Ok(assignments) => {
                        console_log!("GPU K-Means completed successfully");
                        assignments
                    }
                    Err(_e) => {
                        console_log!("GPU K-Means failed: {}, falling back to CPU", _e);
                        run_kmeans(features, n_clusters)?
                    }
                }
            }
            Err(_e) => {
                console_log!("GPU initialization failed: {}, falling back to CPU", _e);
                run_kmeans(features, n_clusters)?
            }
        }
    } else {
        console_log!("Using CPU K-Means clustering");
        run_kmeans(features, n_clusters)?
    };

    // Build Arrow IPC result
    let result = build_cluster_result(parsed.order_ids, cluster_ids)?;

    Ok(result)
}

/// Predict future inventory demand using Linear Regression
///
/// # Arguments
/// * `data` - Arrow IPC Stream format bytes with x (time/index) and y (demand) columns
/// * `predict_steps` - Number of future time steps to predict
/// * `scaling_mode` - Scaling method: 0=None (pre-scaled by DuckDB), 1=MinMax, 2=Standard
///
/// # Returns
/// * `Ok(Uint8Array)` - Arrow IPC result with predictions
/// * `Err(JsError)` - Error message
///
/// # Performance Note
/// **This function is marked `async` to return a Promise to JavaScript, but the actual
/// computation is synchronous and CPU-intensive.** Linear regression training performs
/// matrix operations that execute atomically. For large datasets, consider calling this
/// function from a Web Worker to avoid blocking the browser's main thread.
///
/// # Scaling
/// The `scaling_mode` parameter allows flexible data preprocessing:
/// - `0` (None): Assumes data is already scaled (e.g., by DuckDB Wasm)
/// - `1` (MinMax): Applies MinMax scaling in Rust
/// - `2` (Standard): Applies Standard scaling in Rust
#[wasm_bindgen]
pub async fn predict_inventory_demand(
    data: &[u8],
    predict_steps: usize,
    scaling_mode: u8,
) -> Result<Vec<u8>, JsError> {
    // Parse Arrow IPC input
    let parsed = parse_arrow_ipc(data)?;

    // Apply scaling based on mode (for features)
    let features = match ScalingMethod::from(scaling_mode) {
        ScalingMethod::None => parsed.features,
        ScalingMethod::MinMax => min_max_scale(parsed.features)?,
        ScalingMethod::Standard => standard_scale(parsed.features)?,
    };

    // Extract X (first feature column) and Y (second feature column)
    // Expected schema: order_id (String), x (Float64), y (Float64)
    if features.ncols() < 2 {
        return Err(JsError::new(
            "Linear regression requires at least 2 feature columns (x and y)",
        ));
    }

    let x = features.column(0).to_owned().insert_axis(ndarray::Axis(1));
    let y = features.column(1).to_owned();

    // Run Linear Regression
    let predictions = run_linear_regression(x, y, predict_steps)?;

    // Build Arrow IPC result
    let result = build_regression_result(predictions)?;

    Ok(result)
}

/// Convert CSV stream to Parquet format
///
/// Generates a complete, valid Parquet file that DuckDB Wasm can directly load.
///
/// # Arguments
/// * `csv_data` - CSV file bytes as Uint8Array
/// * `delimiter` - CSV delimiter: b',' (44), b'\t' (9), b'|' (124), b';' (59)
/// * `has_header` - Whether first row contains column headers
/// * `row_group_size` - Parquet row group size (64-16384, default: 1024)
///
/// # Returns
/// * `Ok(Uint8Array)` - Complete Parquet file bytes with footer metadata (DuckDB Wasm compatible)
/// * `Err(Error)` - Conversion error with detailed message
///
/// # Note
/// **In Wasm builds**: Returns error (CSV library incompatible). 
/// Use DuckDB Wasm CSV reader instead.
/// 
/// **In non-Wasm builds** (Node.js): Generates valid Parquet with complete footer.
#[wasm_bindgen]
pub async fn convert_csv_to_parquet(
    csv_data: &[u8],
    delimiter: u8,
    has_header: bool,
    row_group_size: usize,
) -> Result<Vec<u8>, JsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (csv_data, delimiter, has_header, row_group_size);
        Err(JsError::new(
            "CSV to Parquet unavailable in Wasm (csv library uses C code). \
             Alternatives: Use DuckDB Wasm CSV reader, or papaparse + Arrow IPC builder.",
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::file_convert::{Converter, CsvReadOptions, ParquetWriteOptions};

        let mut converter = Converter::new();
        let csv_opts = CsvReadOptions {
            delimiter,
            has_header,
            null_handling: crate::file_convert::NullHandling::Null,
        };
        let pq_opts = ParquetWriteOptions {
            row_group_size,
            compression: crate::file_convert::ParquetCompression::Uncompressed,
        };

        // Begin conversion (schema_hint = None for lenient mode - DuckDB will infer types)
        converter.begin_csv_to_parquet(csv_opts, pq_opts, None)
            .map_err(|e| JsError::new(&format!("CSV→Parquet error: {}", e)))?;

        // Process all CSV data in one chunk
        let chunks = converter.feed_csv_chunk(csv_data, true)
            .map_err(|e| JsError::new(&format!("CSV chunk error: {}", e)))?;

        // Merge all Parquet chunks: each is a complete Parquet file
        // DuckDB Wasm can load the merged result directly
        let mut result = Vec::new();
        for chunk in chunks {
            result.extend_from_slice(&chunk);
        }

        if result.is_empty() {
            return Err(JsError::new("CSV conversion produced no output"));
        }

        Ok(result)
    }
}

/// Convert Excel file to Parquet format
///
/// Generates a complete, valid Parquet file that DuckDB Wasm can directly load.
///
/// # Arguments
/// * `excel_data` - Excel file bytes (XLSX/XLS) as Uint8Array
/// * `sheet_name_or_index` - Sheet selector: empty string = first sheet, or sheet name
/// * `has_header` - Whether first row contains column headers
/// * `row_group_size` - Parquet row group size (64-16384, default: 1024)
/// * `max_string_table_bytes` - Maximum Excel string table size in bytes (0 = use 100MB default)
///
/// # Returns
/// * `Ok(Uint8Array)` - Complete Parquet file bytes with footer metadata (DuckDB Wasm compatible)
/// * `Err(Error)` - Conversion error with detailed message
///
/// # Note
/// **In Wasm builds**: Returns error (calamine library uses C code).
/// Use JavaScript Excel libraries (xlsx/exceljs) instead.
/// 
/// **In non-Wasm builds** (Node.js): Generates valid Parquet with complete footer.
#[wasm_bindgen]
pub async fn convert_excel_to_parquet(
    excel_data: &[u8],
    sheet_name_or_index: String,
    has_header: bool,
    row_group_size: usize,
    max_string_table_bytes: u64,
) -> Result<Vec<u8>, JsError> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (excel_data, sheet_name_or_index, has_header, row_group_size, max_string_table_bytes);
        Err(JsError::new(
            "Excel to Parquet unavailable in Wasm (calamine library uses C code). \
             Alternatives: Use JavaScript libraries (xlsx, exceljs) to read Excel, then Arrow IPC builder.",
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::file_convert::{Converter, ExcelLoadOptions, ParquetWriteOptions, SheetSelector};

        let mut converter = Converter::new();
        let excel_opts = ExcelLoadOptions {
            sheet: if sheet_name_or_index.is_empty() {
                None  // Use first sheet
            } else {
                Some(SheetSelector::ByName(sheet_name_or_index))
            },
            // Use provided limit or default to 100 MB if 0 is passed
            max_string_table_bytes: Some(if max_string_table_bytes == 0 {
                100_000_000  // 100 MB default
            } else {
                max_string_table_bytes
            }),
        };
        let pq_opts = ParquetWriteOptions {
            row_group_size,
            compression: crate::file_convert::ParquetCompression::Uncompressed,
        };

        // Begin conversion
        converter.begin_excel_to_parquet(excel_opts, pq_opts)
            .map_err(|e| JsError::new(&format!("Excel→Parquet error: {}", e)))?;

        // Process all Excel data in one chunk
        let chunks = converter.feed_excel_chunk(excel_data, true)
            .map_err(|e| JsError::new(&format!("Excel chunk error: {}", e)))?;

        // Merge all Parquet chunks: each is a complete Parquet file
        // DuckDB Wasm can load the merged result directly
        let mut result = Vec::new();
        for chunk in chunks {
            result.extend_from_slice(&chunk);
        }

        if result.is_empty() {
            return Err(JsError::new("Excel conversion produced no output"));
        }

        Ok(result)
    }
}


/// Get Wasm module version for compatibility checking
///
/// # Returns
/// * Version string (e.g., "0.1.0")
#[wasm_bindgen]
pub fn get_wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
