use extended_isolation_forest::{Forest, ForestOptions};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};

use crate::utils::{AnalysisError, validate_threshold};

/// Controls the feature engineering strategy used for inventory demand prediction.
///
/// All modes use ordinary least squares (OLS) regression internally — only the
/// feature matrix differs. Higher-order modes can model non-linear and seasonal
/// patterns that plain linear regression misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMode {
    /// `y = a + b·t` — straight-line trend (existing behavior)
    Linear,
    /// `y = a + b·t + c·t²` — polynomial trend (degree=2 only)
    ///
    /// Use for S-curves, saturation, or accelerating/decelerating growth.
    Polynomial { degree: u8 },
    /// `y = a + b·t + c·sin(2πt/P) + d·cos(2πt/P)` — trend + one Fourier term
    ///
    /// Use when demand has a known periodic cycle (e.g., period=7 for weekly).
    Seasonal { period: u32 },
    /// `y = a + b·t + c·t² + d·sin(2πt/P) + e·cos(2πt/P)` — full decomposition
    ///
    /// Combines polynomial trend with seasonal cycle. Recommended default for
    /// real-world inventory data with both non-linear growth and periodic patterns.
    Ensemble { degree: u8, period: u32 },
}

impl PredictionMode {
    /// Convert from Wasm-compatible `(mode_u8, season_period_u32)` pair.
    ///
    /// | mode | variant |
    /// |------|---------|
    /// | 0    | Linear |
    /// | 1    | Polynomial { degree: 2 } |
    /// | 2    | Seasonal { period } |
    /// | 3    | Ensemble { degree: 2, period } |
    /// | other | Linear (graceful fallback) |
    ///
    /// `season_period = 0` auto-defaults to `7` (weekly cycle).
    pub fn from_params(mode: u8, season_period: u32) -> Self {
        let period = if season_period == 0 { 7 } else { season_period };
        match mode {
            1 => PredictionMode::Polynomial { degree: 2 },
            2 => PredictionMode::Seasonal { period },
            3 => PredictionMode::Ensemble { degree: 2, period },
            _ => PredictionMode::Linear,
        }
    }
}

/// Run Isolation Forest anomaly detection using Extended Isolation Forest
///
/// # Arguments
/// * `features` - Feature matrix (rows=samples, cols=features, **max 16 columns**)
/// * `threshold` - Anomaly threshold in [0, 1], scores >= threshold are anomalous
///
/// # Returns
/// * `Ok((scores, labels))` - Anomaly scores (0-1) and boolean labels
/// * `Err(AnalysisError)` - If validation or training fails, or features > 16
///
/// # Algorithm
/// Uses Extended Isolation Forest algorithm with optimized tree-based isolation.
/// Scores are already normalized to [0, 1] range where higher = more anomalous.
///
/// # Performance
/// * Complexity: O(n log n)
/// * 100k samples: ~5-10 seconds
/// * Supports dynamic feature dimensions (common sizes: 5, 10, 15, 16)
///
/// # Note
/// Maximum 16 features supported for WASM compatibility. For >16 features,
/// consider feature selection or PCA dimensionality reduction.
pub fn run_isolation_forest(
    features: Array2<f64>,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<bool>), AnalysisError> {
    validate_threshold(threshold)?;

    if features.nrows() == 0 {
        return Err(AnalysisError::ValidationError(
            "empty feature matrix".to_string(),
        ));
    }

    let n_features = features.ncols();

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!(
            "🔍 [IsolationForest] Running Extended IForest with {} samples x {} features",
            features.nrows(),
            n_features
        ));
    }

    // Dispatch to appropriate implementation based on feature count
    match n_features {
        2 => run_iforest_impl::<2>(&features, threshold), // For unit tests
        5 => run_iforest_impl::<5>(&features, threshold),
        7 => run_iforest_impl::<7>(&features, threshold),
        10 => run_iforest_impl::<10>(&features, threshold),
        11 => run_iforest_impl::<11>(&features, threshold),
        13 => run_iforest_impl::<13>(&features, threshold),
        15 => run_iforest_impl::<15>(&features, threshold),
        16 => run_iforest_impl::<16>(&features, threshold),
        _ => Err(AnalysisError::ValidationError(format!(
            "Unsupported feature count: {}. Supported dimensions: 2, 5, 7, 10, 11,13, 15, 16. \
             Please use feature selection or add more cases.",
            n_features
        ))),
    }
}

/// Generic implementation of Extended Isolation Forest for dimension N
fn run_iforest_impl<const N: usize>(
    features: &Array2<f64>,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<bool>), AnalysisError> {
    let _n_samples = features.nrows(); // Used in logging

    // Convert Array2<f64> to Vec<[f64; N]>
    let data: Vec<[f64; N]> = features
        .rows()
        .into_iter()
        .map(|row| {
            let mut arr = [0.0; N];
            for (i, &val) in row.iter().enumerate().take(N) {
                arr[i] = val;
            }
            arr
        })
        .collect();

    // Configure Extended Isolation Forest
    let sample_size = if _n_samples < 256 {
        _n_samples // Use all samples if fewer than 256
    } else {
        256
    };

    let options = ForestOptions {
        n_trees: 100,         // Number of trees in the forest
        sample_size,          // Adaptive subsampling size
        max_tree_depth: None, // Unlimited depth (auto-calculated)
        extension_level: 1,   // Extension level (1 = extended, 0 = standard)
    };

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!(
            "  [IForest] Building forest with {} trees, sample_size={}",
            options.n_trees, options.sample_size
        ));
    }

    // Build the forest
    let forest: Forest<f64, N> = Forest::from_slice(&data, &options).map_err(|e| {
        AnalysisError::ModelError(format!("Extended IForest training failed: {:?}", e))
    })?;

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log("  [IForest] Forest built, computing anomaly scores...");
    }

    // Compute anomaly scores (already normalized to [0, 1])
    let scores: Vec<f64> = data.iter().map(|sample| forest.score(sample)).collect();

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        // Log score distribution for debugging
        let score_min = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let score_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let score_avg = scores.iter().sum::<f64>() / scores.len() as f64;
        log(&format!(
            "  [IForest] Score range: [{:.6}, {:.6}], avg: {:.6}",
            score_min, score_max, score_avg
        ));
    }

    // Apply threshold to generate labels
    let labels: Vec<bool> = scores.iter().map(|&s| s >= threshold).collect();

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        let anomaly_count = labels.iter().filter(|&&l| l).count();
        log(&format!(
            "✓ [IsolationForest] Complete: {} anomalies detected ({:.1}%)",
            anomaly_count,
            (anomaly_count as f64 / _n_samples as f64 * 100.0)
        ));
    }

    Ok((scores, labels))
}

/// Run K-Means clustering
///
/// # Arguments
/// * `features` - Feature matrix (rows=samples, cols=features)
/// * `n_clusters` - Number of clusters
///
/// # Returns
/// * `Ok(cluster_ids)` - Vector of cluster IDs (0, 1, 2, ...)
/// * `Err(AnalysisError)` - If validation or clustering fails
pub fn run_kmeans(features: Array2<f64>, n_clusters: usize) -> Result<Vec<usize>, AnalysisError> {
    // Validate inputs
    if features.nrows() == 0 {
        return Err(AnalysisError::ValidationError(
            "empty feature matrix".to_string(),
        ));
    }

    if n_clusters == 0 {
        return Err(AnalysisError::ValidationError(
            "n_clusters must be > 0".to_string(),
        ));
    }

    if n_clusters > features.nrows() {
        return Err(AnalysisError::ValidationError(format!(
            "n_clusters ({}) cannot exceed number of samples ({})",
            n_clusters,
            features.nrows()
        )));
    }

    // Create dataset (linfa needs records + targets, we use unit targets for unsupervised)
    let targets = Array1::from_elem(features.nrows(), ());
    let dataset = Dataset::new(features, targets);

    // Train K-Means model
    let model = KMeans::params(n_clusters)
        .max_n_iterations(100)
        .fit(&dataset)
        .map_err(|e| AnalysisError::ModelError(format!("K-Means clustering failed: {}", e)))?;

    // Get cluster assignments
    let predictions = model.predict(dataset.records());
    let cluster_ids: Vec<usize> = predictions.into_iter().collect();

    Ok(cluster_ids)
}

/// Build the feature matrix for a given time index slice and prediction mode.
///
/// Column layout by mode:
/// - Linear:     `[t]`               → shape (n, 1)
/// - Polynomial: `[t, t²]`           → shape (n, 2)
/// - Seasonal:   `[t, sin, cos]`     → shape (n, 3)
/// - Ensemble:   `[t, t², sin, cos]` → shape (n, 4)
///
/// The intercept term is handled internally by `linfa_linear`, so it is NOT
/// included in this matrix.
fn build_feature_matrix(t: &[f64], mode: &PredictionMode) -> Result<Array2<f64>, AnalysisError> {
    let n = t.len();
    if n == 0 {
        return Err(AnalysisError::ValidationError(
            "empty time index".to_string(),
        ));
    }

    // Validate period > 0 for seasonal modes to prevent division by zero (NaN features)
    match mode {
        PredictionMode::Seasonal { period } | PredictionMode::Ensemble { period, .. } => {
            if *period == 0 {
                return Err(AnalysisError::ValidationError(
                    "season period must be > 0 for Seasonal and Ensemble modes".to_string(),
                ));
            }
        }
        _ => {}
    }

    let tau = 2.0 * std::f64::consts::PI;

    let (data, ncols): (Vec<f64>, usize) = match mode {
        PredictionMode::Linear => (t.to_vec(), 1),

        PredictionMode::Polynomial { .. } => {
            let data: Vec<f64> = t.iter().flat_map(|&ti| [ti, ti * ti]).collect();
            (data, 2)
        }

        PredictionMode::Seasonal { period } => {
            let p = *period as f64;
            let data: Vec<f64> = t
                .iter()
                .flat_map(|&ti| [ti, (tau * ti / p).sin(), (tau * ti / p).cos()])
                .collect();
            (data, 3)
        }

        PredictionMode::Ensemble { period, .. } => {
            let p = *period as f64;
            let data: Vec<f64> = t
                .iter()
                .flat_map(|&ti| [ti, ti * ti, (tau * ti / p).sin(), (tau * ti / p).cos()])
                .collect();
            (data, 4)
        }
    };

    Array2::from_shape_vec((n, ncols), data)
        .map_err(|e| AnalysisError::ModelError(format!("failed to build feature matrix: {}", e)))
}

/// Run Linear Regression prediction
///
/// # Arguments
/// * `x` - Input features (rows=samples, must be 1 column)
/// * `y` - Target values (1D array)
/// * `predict_steps` - Number of future steps to predict
///
/// # Returns
/// * `Ok(predictions)` - Vector of predicted values
/// * `Err(AnalysisError)` - If validation or training fails
pub fn run_linear_regression(
    x: Array2<f64>,
    y: Array1<f64>,
    predict_steps: usize,
) -> Result<Vec<f64>, AnalysisError> {
    // Validate inputs
    if x.nrows() == 0 || y.is_empty() {
        return Err(AnalysisError::ValidationError(
            "empty input data".to_string(),
        ));
    }

    if x.nrows() != y.len() {
        return Err(AnalysisError::ValidationError(format!(
            "x rows ({}) must match y length ({})",
            x.nrows(),
            y.len()
        )));
    }

    if predict_steps == 0 {
        return Err(AnalysisError::ValidationError(
            "predict_steps must be > 0".to_string(),
        ));
    }

    // Create dataset
    let dataset = Dataset::new(x.clone(), y);

    // Train linear regression model
    let model = LinearRegression::default()
        .fit(&dataset)
        .map_err(|e| AnalysisError::ModelError(format!("linear regression failed: {}", e)))?;

    // Generate future x values for prediction
    let last_x = x.nrows() as f64;
    let future_x: Vec<f64> = (1..=predict_steps).map(|i| last_x + i as f64).collect();
    let future_x_matrix = Array2::from_shape_vec((predict_steps, 1), future_x).map_err(|e| {
        AnalysisError::ModelError(format!("failed to create prediction matrix: {}", e))
    })?;

    // Make predictions
    let predictions: Array1<f64> = model.predict(&future_x_matrix);
    Ok(predictions.to_vec())
}

/// Run regression prediction using configurable feature engineering modes.
///
/// Unlike `run_linear_regression`, this function generates a time index
/// `t = [0, 1, ..., n-1]` internally, making it suitable for sequential
/// time-series data. The caller only provides the target demand values `y`.
///
/// # Arguments
/// * `y`             - Target demand values, one per time step
/// * `predict_steps` - Number of future steps to forecast (must be > 0)
/// * `mode`          - Feature engineering strategy; see `PredictionMode`
///
/// # Returns
/// * `Ok(predictions)` - Predicted demand for the next `predict_steps` periods
/// * `Err(AnalysisError)` - If validation fails or model training fails
///
/// # Minimum data requirement
/// At least 2 samples are required. For Seasonal/Ensemble modes, providing
/// at least 2× the season period is recommended for reliable estimates.
pub fn run_regression_with_mode(
    y: Array1<f64>,
    predict_steps: usize,
    mode: PredictionMode,
) -> Result<Vec<f64>, AnalysisError> {
    let n = y.len();

    if n < 2 {
        return Err(AnalysisError::ValidationError(
            "prediction requires at least 2 data points".to_string(),
        ));
    }

    if predict_steps == 0 {
        return Err(AnalysisError::ValidationError(
            "predict_steps must be > 0".to_string(),
        ));
    }

    // Build time index for training: [0.0, 1.0, ..., n-1.0]
    let t_train: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x_train = build_feature_matrix(&t_train, &mode)?;

    // Train model (linfa_linear adds intercept by default)
    let dataset = Dataset::new(x_train, y);
    let model = LinearRegression::default()
        .fit(&dataset)
        .map_err(|e| AnalysisError::ModelError(format!("regression fitting failed: {}", e)))?;

    // Build future time index: [n.0, n+1.0, ..., n+predict_steps-1.0]
    let t_future: Vec<f64> = (0..predict_steps).map(|i| (n + i) as f64).collect();
    let x_future = build_feature_matrix(&t_future, &mode)?;

    let predictions: Array1<f64> = model.predict(&x_future);
    Ok(predictions.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_run_isolation_forest_normal() {
        let features = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [100.0, 200.0]]);
        let (scores, labels) = run_isolation_forest(features, 0.8).unwrap();

        assert_eq!(scores.len(), 4);
        assert_eq!(labels.len(), 4);

        // Last point (100, 200) should be more anomalous
        assert!(scores[3] > scores[0]);
    }

    #[test]
    fn test_run_isolation_forest_many_features() {
        // 20 features - should fail with KD-Tree (max 16)
        let mut data = vec![0.0; 200]; // 10 samples × 20 features
        for i in 0..10 {
            for j in 0..20 {
                // Make each point unique
                data[i * 20 + j] = (i as f64 * 20.0) + (j as f64 * 0.1);
            }
        }
        let features = Array2::from_shape_vec((10, 20), data).unwrap();
        let result = run_isolation_forest(features, 0.5);

        assert!(result.is_err());
        if let Err(AnalysisError::ValidationError(msg)) = result {
            assert!(msg.contains("16") || msg.contains("20"));
        }
    }

    #[test]
    fn test_run_isolation_forest_invalid_threshold() {
        let features = arr2(&[[1.0, 2.0]]);
        assert!(run_isolation_forest(features, 1.5).is_err());
    }

    #[test]
    fn test_run_isolation_forest_empty() {
        let features = Array2::<f64>::zeros((0, 2));
        assert!(run_isolation_forest(features, 0.5).is_err());
    }

    #[test]
    fn test_run_isolation_forest_7_features() {
        // Test 7 dimensions
        let mut data = vec![0.0; 70]; // 10 samples × 7 features
        for i in 0..10 {
            for j in 0..7 {
                data[i * 7 + j] = (i as f64) + (j as f64 * 0.1);
            }
        }
        let features = Array2::from_shape_vec((10, 7), data).unwrap();
        let (scores, labels) = run_isolation_forest(features, 0.5).unwrap();

        assert_eq!(scores.len(), 10);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn test_run_isolation_forest_11_features() {
        // Test 11 dimensions
        let mut data = vec![0.0; 110]; // 10 samples × 11 features
        for i in 0..10 {
            for j in 0..11 {
                data[i * 11 + j] = (i as f64) + (j as f64 * 0.1);
            }
        }
        let features = Array2::from_shape_vec((10, 11), data).unwrap();
        let (scores, labels) = run_isolation_forest(features, 0.5).unwrap();

        assert_eq!(scores.len(), 10);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn test_run_kmeans_normal() {
        let features = arr2(&[[1.0, 1.0], [1.5, 1.5], [10.0, 10.0], [10.5, 10.5]]);
        let cluster_ids = run_kmeans(features, 2).unwrap();

        assert_eq!(cluster_ids.len(), 4);
        // Points 0,1 should be in same cluster, points 2,3 in another
        assert_eq!(cluster_ids[0], cluster_ids[1]);
        assert_eq!(cluster_ids[2], cluster_ids[3]);
        assert_ne!(cluster_ids[0], cluster_ids[2]);
    }

    #[test]
    fn test_run_kmeans_invalid_n_clusters() {
        let features = arr2(&[[1.0, 2.0]]);
        assert!(run_kmeans(features.clone(), 0).is_err());
        assert!(run_kmeans(features, 2).is_err()); // more clusters than samples
    }

    #[test]
    fn test_run_linear_regression_normal() {
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let y = arr1(&[2.0, 4.0, 6.0, 8.0]); // y = 2x

        let predictions = run_linear_regression(x, y, 3).unwrap();

        assert_eq!(predictions.len(), 3);
        // Should predict approximately 10, 12, 14
        assert!((predictions[0] - 10.0).abs() < 1.0);
        assert!((predictions[1] - 12.0).abs() < 1.0);
        assert!((predictions[2] - 14.0).abs() < 1.0);
    }

    #[test]
    fn test_run_linear_regression_dimension_mismatch() {
        let x = arr2(&[[1.0], [2.0]]);
        let y = arr1(&[1.0, 2.0, 3.0]); // mismatched length

        assert!(run_linear_regression(x, y, 1).is_err());
    }

    #[test]
    fn test_run_linear_regression_invalid_predict_steps() {
        let x = arr2(&[[1.0]]);
        let y = arr1(&[1.0]);

        assert!(run_linear_regression(x, y, 0).is_err());
    }

    // ─── build_feature_matrix tests ───────────────────────────────────────────

    #[test]
    fn test_build_feature_matrix_linear_shape() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let mat = build_feature_matrix(&t, &PredictionMode::Linear).unwrap();
        assert_eq!(mat.shape(), &[4, 1]);
        assert_eq!(mat[[0, 0]], 0.0);
        assert_eq!(mat[[3, 0]], 3.0);
    }

    #[test]
    fn test_build_feature_matrix_polynomial_values() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let mat = build_feature_matrix(&t, &PredictionMode::Polynomial { degree: 2 }).unwrap();
        assert_eq!(mat.shape(), &[4, 2]);
        // Second column = t²
        assert_eq!(mat[[0, 1]], 0.0);
        assert_eq!(mat[[1, 1]], 1.0);
        assert_eq!(mat[[2, 1]], 4.0);
        assert_eq!(mat[[3, 1]], 9.0);
    }

    #[test]
    fn test_build_feature_matrix_seasonal_shape_and_values() {
        let t = vec![0.0, 1.0, 2.0, 3.0];
        let mat = build_feature_matrix(&t, &PredictionMode::Seasonal { period: 4 }).unwrap();
        assert_eq!(mat.shape(), &[4, 3]);
        // t=0: sin(0)=0, cos(0)=1
        assert!((mat[[0, 1]] - 0.0).abs() < 1e-10); // sin
        assert!((mat[[0, 2]] - 1.0).abs() < 1e-10); // cos
        // t=1, P=4: sin(π/2)=1, cos(π/2)=0
        assert!((mat[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((mat[[1, 2]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_feature_matrix_ensemble_shape() {
        let t = vec![0.0, 1.0, 2.0];
        let mat = build_feature_matrix(
            &t,
            &PredictionMode::Ensemble {
                degree: 2,
                period: 7,
            },
        )
        .unwrap();
        assert_eq!(mat.shape(), &[3, 4]); // [t, t², sin, cos]
    }

    #[test]
    fn test_build_feature_matrix_empty_t_error() {
        let result = build_feature_matrix(&[], &PredictionMode::Linear);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_feature_matrix_period_zero_error() {
        let t = vec![0.0, 1.0, 2.0];
        let result = build_feature_matrix(&t, &PredictionMode::Seasonal { period: 0 });
        assert!(result.is_err());

        let result2 = build_feature_matrix(&t, &PredictionMode::Ensemble { degree: 2, period: 0 });
        assert!(result2.is_err());
    }

    // ─── PredictionMode::from_params tests ────────────────────────────────────

    #[test]
    fn test_prediction_mode_from_params() {
        assert_eq!(PredictionMode::from_params(0, 7), PredictionMode::Linear);
        assert_eq!(PredictionMode::from_params(1, 0), PredictionMode::Polynomial { degree: 2 });
        assert_eq!(PredictionMode::from_params(2, 7), PredictionMode::Seasonal { period: 7 });
        // mode=2 with season_period=0 → auto-defaults to period=7
        assert_eq!(PredictionMode::from_params(2, 0), PredictionMode::Seasonal { period: 7 });
        assert_eq!(PredictionMode::from_params(3, 30), PredictionMode::Ensemble { degree: 2, period: 30 });
        // Unknown mode → Linear fallback
        assert_eq!(PredictionMode::from_params(99, 7), PredictionMode::Linear);
    }

    // ─── run_regression_with_mode tests ───────────────────────────────────────

    #[test]
    fn test_run_regression_linear_mode_accuracy() {
        // y = 2t  →  train on t=0..3, predict t=4,5,6 → expect ~8,10,12
        let y = arr1(&[0.0, 2.0, 4.0, 6.0]);
        let preds = run_regression_with_mode(y, 3, PredictionMode::Linear).unwrap();
        assert_eq!(preds.len(), 3);
        assert!((preds[0] - 8.0).abs() < 0.5, "expected ~8, got {}", preds[0]);
        assert!((preds[1] - 10.0).abs() < 0.5, "expected ~10, got {}", preds[1]);
        assert!((preds[2] - 12.0).abs() < 0.5, "expected ~12, got {}", preds[2]);
    }

    #[test]
    fn test_run_regression_polynomial_mode_accuracy() {
        // y = t²  →  train on t=0..4, predict t=5,6 → expect ~25, 36
        let y = arr1(&[0.0, 1.0, 4.0, 9.0, 16.0]);
        let preds = run_regression_with_mode(y, 2, PredictionMode::Polynomial { degree: 2 }).unwrap();
        assert_eq!(preds.len(), 2);
        assert!((preds[0] - 25.0).abs() < 1.0, "expected ~25, got {}", preds[0]);
        assert!((preds[1] - 36.0).abs() < 1.0, "expected ~36, got {}", preds[1]);
    }

    #[test]
    fn test_run_regression_seasonal_mode_shape_and_convergence() {
        // y = sin(2πt/4), 8 samples, period=4
        // Predict 4 steps → expect shape=[4] and values within [-2.0, 2.0]
        let tau = 2.0 * std::f64::consts::PI;
        let y: Array1<f64> = (0..8).map(|i| (tau * i as f64 / 4.0).sin()).collect();
        let preds = run_regression_with_mode(y, 4, PredictionMode::Seasonal { period: 4 }).unwrap();
        assert_eq!(preds.len(), 4);
        for p in &preds {
            assert!(p.abs() < 2.0, "seasonal prediction out of range: {}", p);
        }
    }

    #[test]
    fn test_run_regression_ensemble_mode_shape() {
        // y = t + sin(2πt/4), 8 samples, predict 2 steps
        let tau = 2.0 * std::f64::consts::PI;
        let y: Array1<f64> = (0..8)
            .map(|i| i as f64 + (tau * i as f64 / 4.0).sin())
            .collect();
        let preds = run_regression_with_mode(
            y,
            2,
            PredictionMode::Ensemble { degree: 2, period: 4 },
        )
        .unwrap();
        assert_eq!(preds.len(), 2);
        // t=8: 8 + sin(4π) ≈ 8.0; t=9: 9 + sin(4.5π) ≈ 10.0
        assert!((preds[0] - 8.0).abs() < 1.5, "expected ~8, got {}", preds[0]);
        assert!((preds[1] - 10.0).abs() < 1.5, "expected ~10, got {}", preds[1]);
    }

    #[test]
    fn test_run_regression_with_mode_empty_error() {
        let y = arr1(&[] as &[f64]);
        assert!(run_regression_with_mode(y, 3, PredictionMode::Linear).is_err());
    }

    #[test]
    fn test_run_regression_with_mode_single_sample_error() {
        let y = arr1(&[5.0]);
        assert!(run_regression_with_mode(y, 1, PredictionMode::Linear).is_err());
    }

    #[test]
    fn test_run_regression_with_mode_zero_steps_error() {
        let y = arr1(&[1.0, 2.0, 3.0]);
        assert!(run_regression_with_mode(y, 0, PredictionMode::Linear).is_err());
    }

    #[test]
    fn test_run_regression_with_mode_season_period_zero_defaults_to_weekly() {
        // from_params(2, 0) → Seasonal { period: 7 }, should not error on 14+ samples
        let y: Array1<f64> = (0..14).map(|i| i as f64).collect();
        let preds = run_regression_with_mode(
            y,
            1,
            PredictionMode::from_params(2, 0), // auto period → 7
        )
        .unwrap();
        assert_eq!(preds.len(), 1);
    }
}
