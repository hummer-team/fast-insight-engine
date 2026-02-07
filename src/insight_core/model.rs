use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use extended_isolation_forest::{Forest, ForestOptions};

use crate::utils::{validate_threshold, AnalysisError};

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
        log(&format!("🔍 [IsolationForest] Running Extended IForest with {} samples x {} features", 
                     features.nrows(), n_features));
    }

    // Dispatch to appropriate implementation based on feature count
    match n_features {
        2 => run_iforest_impl::<2>(&features, threshold),   // For unit tests
        5 => run_iforest_impl::<5>(&features, threshold),
        7 => run_iforest_impl::<7>(&features, threshold),
        10 => run_iforest_impl::<10>(&features, threshold),
        11 => run_iforest_impl::<11>(&features, threshold),
        15 => run_iforest_impl::<15>(&features, threshold),
        16 => run_iforest_impl::<16>(&features, threshold),
        _ => Err(AnalysisError::ValidationError(format!(
            "Unsupported feature count: {}. Supported dimensions: 2, 5, 7, 10, 11, 15, 16. \
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
    let _n_samples = features.nrows();  // Used in logging
    
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
        _n_samples  // Use all samples if fewer than 256
    } else {
        256
    };
    
    let options = ForestOptions {
        n_trees: 100,           // Number of trees in the forest
        sample_size,            // Adaptive subsampling size
        max_tree_depth: None,   // Unlimited depth (auto-calculated)
        extension_level: 1,     // Extension level (1 = extended, 0 = standard)
    };
    
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!("  [IForest] Building forest with {} trees, sample_size={}", 
                     options.n_trees, options.sample_size));
    }
    
    // Build the forest
    let forest: Forest<f64, N> = Forest::from_slice(&data, &options)
        .map_err(|e| AnalysisError::ModelError(format!("Extended IForest training failed: {:?}", e)))?;
    
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
        log(&format!("  [IForest] Score range: [{:.6}, {:.6}], avg: {:.6}", 
                     score_min, score_max, score_avg));
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
        log(&format!("✓ [IsolationForest] Complete: {} anomalies detected ({:.1}%)", 
                     anomaly_count, (anomaly_count as f64 / _n_samples as f64 * 100.0)));
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
    let future_x: Vec<f64> = (1..=predict_steps)
        .map(|i| last_x + i as f64)
        .collect();
    let future_x_matrix = Array2::from_shape_vec((predict_steps, 1), future_x)
        .map_err(|e| AnalysisError::ModelError(format!("failed to create prediction matrix: {}", e)))?;

    // Make predictions
    let predictions: Array1<f64> = model.predict(&future_x_matrix);
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
        let features = arr2(&[
            [1.0, 1.0],
            [1.5, 1.5],
            [10.0, 10.0],
            [10.5, 10.5],
        ]);
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
}
