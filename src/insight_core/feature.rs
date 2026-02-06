use ndarray::Array2;

use crate::utils::AnalysisError;

/// Validate feature matrix dimensions and values
///
/// # Arguments
/// * `features` - Feature matrix to validate
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(AnalysisError::ValidationError)` if invalid
pub fn validate_features(features: &Array2<f64>) -> Result<(), AnalysisError> {
    if features.nrows() == 0 {
        return Err(AnalysisError::ValidationError(
            "feature matrix cannot be empty".to_string(),
        ));
    }

    if features.ncols() == 0 {
        return Err(AnalysisError::ValidationError(
            "feature matrix must have at least one column".to_string(),
        ));
    }

    // Check for NaN or Inf values
    for value in features.iter() {
        if value.is_nan() || value.is_infinite() {
            return Err(AnalysisError::ValidationError(
                "feature matrix contains NaN or Inf values".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_validate_features_valid() {
        let features = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(validate_features(&features).is_ok());
    }

    #[test]
    fn test_validate_features_empty_rows() {
        let features = Array2::<f64>::zeros((0, 2));
        assert!(validate_features(&features).is_err());
    }

    #[test]
    fn test_validate_features_empty_cols() {
        let features = Array2::<f64>::zeros((2, 0));
        assert!(validate_features(&features).is_err());
    }

    #[test]
    fn test_validate_features_with_nan() {
        let features = arr2(&[[1.0, f64::NAN], [3.0, 4.0]]);
        let result = validate_features(&features);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NaN"));
    }

    #[test]
    fn test_validate_features_with_inf() {
        let features = arr2(&[[1.0, f64::INFINITY], [3.0, 4.0]]);
        let result = validate_features(&features);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Inf"));
    }
}
