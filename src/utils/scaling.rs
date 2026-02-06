use ndarray::Array2;

use crate::utils::AnalysisError;

/// Scaling method for feature normalization
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalingMethod {
    /// No scaling (data already preprocessed, e.g., by DuckDB)
    None,
    /// MinMax scaling: (x - min) / (max - min) -> [0, 1]
    MinMax,
    /// Standard scaling: (x - mean) / std -> zero mean, unit variance
    Standard,
}

impl From<u8> for ScalingMethod {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::None,
            1 => Self::MinMax,
            2 => Self::Standard,
            _ => Self::None, // Default to None for invalid values
        }
    }
}

/// Apply MinMax scaling: (x - min) / (max - min)
///
/// # Arguments
/// * `features` - Feature matrix to scale
///
/// # Returns
/// * `Ok(scaled)` - Scaled feature matrix with values in [0, 1]
/// * `Err(AnalysisError)` - If scaling fails
///
/// # Note
/// Constant columns (min == max) are set to 0.0
pub fn min_max_scale(features: Array2<f64>) -> Result<Array2<f64>, AnalysisError> {
    let mut scaled = features.clone();

    for col_idx in 0..features.ncols() {
        let col = features.column(col_idx);
        let min = col
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = col
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range.abs() < f64::EPSILON {
            // Constant column, set all values to 0
            for row_idx in 0..features.nrows() {
                scaled[[row_idx, col_idx]] = 0.0;
            }
        } else {
            for row_idx in 0..features.nrows() {
                scaled[[row_idx, col_idx]] = (features[[row_idx, col_idx]] - min) / range;
            }
        }
    }

    Ok(scaled)
}

/// Apply Standard scaling: (x - mean) / std
///
/// # Arguments
/// * `features` - Feature matrix to scale
///
/// # Returns
/// * `Ok(scaled)` - Scaled feature matrix with zero mean and unit variance
/// * `Err(AnalysisError)` - If scaling fails
///
/// # Note
/// Constant columns (std == 0) are set to 0.0
pub fn standard_scale(features: Array2<f64>) -> Result<Array2<f64>, AnalysisError> {
    let mut scaled = features.clone();

    for col_idx in 0..features.ncols() {
        let col = features.column(col_idx);
        let mean = col.mean().unwrap_or(0.0);
        let std = col.std(0.0);

        if std.abs() < f64::EPSILON {
            // Constant column, set all values to 0
            for row_idx in 0..features.nrows() {
                scaled[[row_idx, col_idx]] = 0.0;
            }
        } else {
            for row_idx in 0..features.nrows() {
                scaled[[row_idx, col_idx]] = (features[[row_idx, col_idx]] - mean) / std;
            }
        }
    }

    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_scaling_method_from_u8() {
        assert_eq!(ScalingMethod::from(0), ScalingMethod::None);
        assert_eq!(ScalingMethod::from(1), ScalingMethod::MinMax);
        assert_eq!(ScalingMethod::from(2), ScalingMethod::Standard);
        assert_eq!(ScalingMethod::from(99), ScalingMethod::None); // Invalid -> None
    }

    #[test]
    fn test_min_max_scale_normal() {
        let features = arr2(&[[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]]);
        let scaled = min_max_scale(features).unwrap();

        // Column 0: min=0, max=10, range=10
        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-10); // (0-0)/10 = 0
        assert!((scaled[[1, 0]] - 0.5).abs() < 1e-10); // (5-0)/10 = 0.5
        assert!((scaled[[2, 0]] - 1.0).abs() < 1e-10); // (10-0)/10 = 1

        // Column 1: min=10, max=30, range=20
        assert!((scaled[[0, 1]] - 0.0).abs() < 1e-10); // (10-10)/20 = 0
        assert!((scaled[[1, 1]] - 0.5).abs() < 1e-10); // (20-10)/20 = 0.5
        assert!((scaled[[2, 1]] - 1.0).abs() < 1e-10); // (30-10)/20 = 1
    }

    #[test]
    fn test_min_max_scale_constant_column() {
        let features = arr2(&[[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]]);
        let scaled = min_max_scale(features).unwrap();

        // Column 0 is constant (all 5.0), should be 0.0 after scaling
        assert_eq!(scaled[[0, 0]], 0.0);
        assert_eq!(scaled[[1, 0]], 0.0);
        assert_eq!(scaled[[2, 0]], 0.0);

        // Column 1 should scale normally
        assert!((scaled[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((scaled[[2, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_scale_normal() {
        let features = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]);
        let scaled = standard_scale(features).unwrap();

        // Column 0: mean=2, std≈0.816
        let col0_mean = scaled.column(0).mean().unwrap();
        let col0_std = scaled.column(0).std(0.0);
        assert!((col0_mean - 0.0).abs() < 1e-10); // Mean should be ~0
        assert!((col0_std - 1.0).abs() < 1e-10);  // Std should be ~1

        // Column 1: mean=20, std≈8.165
        let col1_mean = scaled.column(1).mean().unwrap();
        let col1_std = scaled.column(1).std(0.0);
        assert!((col1_mean - 0.0).abs() < 1e-10);
        assert!((col1_std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standard_scale_constant_column() {
        let features = arr2(&[[5.0, 10.0], [5.0, 20.0], [5.0, 30.0]]);
        let scaled = standard_scale(features).unwrap();

        // Column 0 is constant, should be 0.0 after scaling
        assert_eq!(scaled[[0, 0]], 0.0);
        assert_eq!(scaled[[1, 0]], 0.0);
        assert_eq!(scaled[[2, 0]], 0.0);
    }

    #[test]
    fn test_min_max_scale_negative_values() {
        let features = arr2(&[[-10.0], [0.0], [10.0]]);
        let scaled = min_max_scale(features).unwrap();

        assert!((scaled[[0, 0]] - 0.0).abs() < 1e-10);  // (-10-(-10))/20 = 0
        assert!((scaled[[1, 0]] - 0.5).abs() < 1e-10);  // (0-(-10))/20 = 0.5
        assert!((scaled[[2, 0]] - 1.0).abs() < 1e-10);  // (10-(-10))/20 = 1
    }
}
