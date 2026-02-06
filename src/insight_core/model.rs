use extended_isolation_forest::{Forest, ForestOptions};
use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};

use crate::utils::{normalize_scores, validate_threshold, AnalysisError};

/// Run Isolation Forest anomaly detection
///
/// # Arguments
/// * `features` - Feature matrix (rows=samples, cols=features)
/// * `threshold` - Anomaly threshold in [0, 1], scores >= threshold are anomalous
///
/// # Returns
/// * `Ok((scores, labels))` - Normalized scores (0-1) and boolean labels
/// * `Err(AnalysisError)` - If validation or training fails
pub fn run_isolation_forest(
    features: Array2<f64>,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<bool>), AnalysisError> {
    // Validate threshold
    validate_threshold(threshold)?;

    // Validate features
    if features.nrows() == 0 {
        return Err(AnalysisError::ValidationError(
            "empty feature matrix".to_string(),
        ));
    }

    // Convert Array2 to Vec of fixed-size arrays for extended-isolation-forest
    let n_features = features.ncols();
    if n_features == 0 {
        return Err(AnalysisError::ValidationError(
            "feature matrix must have at least one column".to_string(),
        ));
    }

    // For simplicity, we'll support up to 10 features
    // In production, you might want to handle this more dynamically
    match n_features {
        1 => run_isolation_forest_impl::<1>(features, threshold),
        2 => run_isolation_forest_impl::<2>(features, threshold),
        3 => run_isolation_forest_impl::<3>(features, threshold),
        4 => run_isolation_forest_impl::<4>(features, threshold),
        5 => run_isolation_forest_impl::<5>(features, threshold),
        _ => Err(AnalysisError::ValidationError(format!(
            "feature count {} exceeds maximum supported (5)",
            n_features
        ))),
    }
}

/// Internal implementation for specific feature dimensions
fn run_isolation_forest_impl<const N: usize>(
    features: Array2<f64>,
    threshold: f64,
) -> Result<(Vec<f64>, Vec<bool>), AnalysisError> {
    // Convert Array2 to Vec<[f64; N]>
    let data: Vec<[f64; N]> = features
        .rows()
        .into_iter()
        .map(|row| {
            let mut arr = [0.0; N];
            for (i, &val) in row.iter().enumerate() {
                arr[i] = val;
            }
            arr
        })
        .collect();

    // Configure and train model
    let options = ForestOptions {
        n_trees: 100,
        sample_size: 256.min(data.len()),
        max_tree_depth: None,
        extension_level: 0,
    };

    let forest = Forest::from_slice(data.as_slice(), &options).map_err(|e| {
        AnalysisError::ModelError(format!("failed to train isolation forest: {:?}", e))
    })?;

    // Get anomaly scores
    let raw_scores: Vec<f64> = data.iter().map(|sample| forest.score(sample)).collect();

    // Normalize scores to 0-1 range (higher = more anomalous)
    let normalized_scores = normalize_scores(&raw_scores);

    // Apply threshold to generate labels
    let labels: Vec<bool> = normalized_scores
        .iter()
        .map(|&score| score >= threshold)
        .collect();

    Ok((normalized_scores, labels))
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
    let future_x: Vec<f64> = (1..=predict_steps).map(|i| last_x + i as f64).collect();
    let future_x_matrix = Array2::from_shape_vec((predict_steps, 1), future_x).map_err(|e| {
        AnalysisError::ModelError(format!("failed to create prediction matrix: {}", e))
    })?;

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
}
