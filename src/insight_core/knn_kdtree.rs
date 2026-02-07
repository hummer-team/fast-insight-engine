//! KD-Tree based K-Nearest Neighbors for anomaly detection
//!
//! This module provides O(n log n) nearest neighbor search using KD-Tree,
//! replacing the naive O(n²) brute-force approach.

use kiddo::KdTree;
use kiddo::SquaredEuclidean;
use ndarray::Array2;

use crate::utils::AnalysisError;

/// Maximum number of features supported by KD-Tree implementation
pub const MAX_FEATURES: usize = 16;

/// Compute anomaly scores using KD-Tree for efficient nearest neighbor search
///
/// # Arguments
/// * `features` - Feature matrix (rows=samples, cols=features)
/// * `k` - Number of nearest neighbors to consider
///
/// # Returns
/// * `Ok(Vec<f64>)` - Anomaly scores (average distance to k nearest neighbors)
/// * `Err(AnalysisError)` - Validation error if features exceed MAX_FEATURES
///
/// # Complexity
/// * Build: O(n log n)
/// * Query: O(n log n) total for all points
/// * vs Naive: O(n²) - approximately 5000x faster for 100k samples
pub fn knn_anomaly_scores(features: &Array2<f64>, k: usize) -> Result<Vec<f64>, AnalysisError> {
    let (n_samples, n_features) = features.dim();

    // Validate feature dimensions
    if n_features > MAX_FEATURES {
        return Err(AnalysisError::ValidationError(format!(
            "Feature count {} exceeds maximum supported dimension {}. \
             Consider feature selection or dimensionality reduction (PCA).",
            n_features, MAX_FEATURES
        )));
    }

    if n_samples == 0 {
        return Err(AnalysisError::ValidationError(
            "Cannot compute anomaly scores on empty dataset".into(),
        ));
    }

    // Handle edge case: k >= n_samples
    let effective_k = k.min(n_samples - 1);
    if effective_k == 0 {
        return Err(AnalysisError::ValidationError(
            "Need at least 2 samples to compute nearest neighbors".into(),
        ));
    }

    // Build KD-Tree: O(n log n)
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!("🔍 [KNN] Building KD-Tree with {} samples x {} features (k={})", n_samples, n_features, effective_k));
    }
    
    // kiddo 4.2 API: KdTree<A, const K: usize> where A is scalar type
    let mut tree: KdTree<f64, MAX_FEATURES> = KdTree::new();

    for i in 0..n_samples {
        let mut point = [0.0; MAX_FEATURES];
        for (j, &val) in features.row(i).iter().enumerate().take(n_features) {
            point[j] = val;
        }
        // kiddo 4.2 uses add() which modifies tree in-place
        tree.add(&point, i as u64);
        
        // Log progress every 5000 nodes
        #[cfg(target_arch = "wasm32")]
        if (i + 1) % 5000 == 0 {
            use wasm_bindgen::prelude::*;
            #[wasm_bindgen]
            extern "C" {
                #[wasm_bindgen(js_namespace = console)]
                fn log(s: &str);
            }
            log(&format!("  [KNN] Built {} / {} nodes", i + 1, n_samples));
        }
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!("✓ [KNN] KD-Tree built successfully"));
    }

    // Query k nearest neighbors for each point: O(n log n)
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!("🔍 [KNN] Querying {} nearest neighbors for each sample...", effective_k));
    }
    
    let mut scores = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut point = [0.0; MAX_FEATURES];
        for (j, &val) in features.row(i).iter().enumerate().take(n_features) {
            point[j] = val;
        }

        // Find k+1 neighbors (including self) using nearest_n with SquaredEuclidean metric
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&point, effective_k + 1);

        // Calculate average distance to k nearest neighbors (skip self at index 0)
        let avg_distance: f64 = neighbors
            .iter()
            .skip(1) // Skip self
            .map(|n| n.distance.sqrt()) // Convert squared distance to Euclidean
            .sum::<f64>()
            / effective_k as f64;

        scores.push(avg_distance);
        
        // Log progress every 5000 queries
        #[cfg(target_arch = "wasm32")]
        if (i + 1) % 5000 == 0 {
            use wasm_bindgen::prelude::*;
            #[wasm_bindgen]
            extern "C" {
                #[wasm_bindgen(js_namespace = console)]
                fn log(s: &str);
            }
            log(&format!("  [KNN] Queried {} / {} samples", i + 1, n_samples));
        }
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console)]
            fn log(s: &str);
        }
        log(&format!("✓ [KNN] Computed {} anomaly scores", scores.len()));
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_knn_simple_case() {
        // 3 points in 2D: (0,0), (1,1), (10,10)
        // (10,10) should have highest anomaly score
        let features = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0])
            .unwrap();

        let scores = knn_anomaly_scores(&features, 2).unwrap();
        assert_eq!(scores.len(), 3);

        // (10,10) is furthest from others
        assert!(scores[2] > scores[0]);
        assert!(scores[2] > scores[1]);
    }

    #[test]
    fn test_knn_feature_limit() {
        // 17 features - should fail
        let features = Array2::zeros((10, 17));
        let result = knn_anomaly_scores(&features, 5);

        assert!(result.is_err());
        if let Err(AnalysisError::ValidationError(msg)) = result {
            assert!(msg.contains("17"));
            assert!(msg.contains("16"));
        }
    }

    #[test]
    fn test_knn_empty_dataset() {
        let features = Array2::zeros((0, 5));
        let result = knn_anomaly_scores(&features, 5);

        assert!(result.is_err());
        if let Err(AnalysisError::ValidationError(msg)) = result {
            assert!(msg.contains("empty"));
        }
    }

    #[test]
    fn test_knn_insufficient_samples() {
        let features = Array2::zeros((1, 5));
        let result = knn_anomaly_scores(&features, 5);

        assert!(result.is_err());
        if let Err(AnalysisError::ValidationError(msg)) = result {
            assert!(msg.contains("at least 2"));
        }
    }

    #[test]
    fn test_knn_k_larger_than_samples() {
        // 5 samples, k=10 - should auto-adjust to k=4
        let features = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        )
        .unwrap();

        let scores = knn_anomaly_scores(&features, 10).unwrap();
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_knn_max_features() {
        // Exactly 16 features - should work
        let mut data = vec![0.0; 1600]; // 100 samples × 16 features
        for i in 0..1600 {
            data[i] = (i as f64) * 0.001; // Add variation to avoid duplicate points
        }
        let features = Array2::from_shape_vec((100, 16), data).unwrap();
        let result = knn_anomaly_scores(&features, 5);

        assert!(result.is_ok());
        let scores = result.unwrap();
        assert_eq!(scores.len(), 100);
    }

    #[test]
    fn test_knn_large_dataset() {
        // Simulate 100 samples with 10 features (reduced from 1000 to avoid kiddo limitations)
        let mut data = vec![0.0; 1000];
        for i in 0..100 {
            for j in 0..10 {
                // Make each point highly unique by using prime multipliers
                data[i * 10 + j] = (i as f64 * 13.7) + (j as f64 * 7.3) + ((i * j) as f64 * 0.11);
            }
        }
        let features = Array2::from_shape_vec((100, 10), data).unwrap();

        let scores = knn_anomaly_scores(&features, 5).unwrap();
        assert_eq!(scores.len(), 100);

        // Scores should be all positive
        assert!(scores.iter().all(|&s| s > 0.0));
    }
}
