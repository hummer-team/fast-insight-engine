use super::error::AnalysisError;

/// Validate threshold is within valid range [0, 1]
///
/// # Arguments
/// * `threshold` - The threshold value to validate
///
/// # Returns
/// * `Ok(())` if valid
/// * `Err(AnalysisError::ValidationError)` if out of range
pub fn validate_threshold(threshold: f64) -> Result<(), AnalysisError> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(AnalysisError::ValidationError(format!(
            "threshold must be 0-1, got {}",
            threshold
        )));
    }
    Ok(())
}

/// Normalize scores to 0-1 range (higher = more anomalous)
///
/// # Arguments
/// * `raw_scores` - Raw anomaly scores from the model
///
/// # Returns
/// * Vector of normalized scores in [0, 1] range
pub fn normalize_scores(raw_scores: &[f64]) -> Vec<f64> {
    if raw_scores.is_empty() {
        return Vec::new();
    }

    let min = raw_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    raw_scores
        .iter()
        .map(|&s| if range > 0.0 { (s - min) / range } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_threshold_valid() {
        assert!(validate_threshold(0.0).is_ok());
        assert!(validate_threshold(0.5).is_ok());
        assert!(validate_threshold(1.0).is_ok());
    }

    #[test]
    fn test_validate_threshold_invalid() {
        assert!(validate_threshold(-0.1).is_err());
        assert!(validate_threshold(1.1).is_err());
        assert!(validate_threshold(-1.0).is_err());
        assert!(validate_threshold(2.0).is_err());
    }

    #[test]
    fn test_validate_threshold_error_message() {
        let result = validate_threshold(1.5);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(
            err.to_string(),
            "ValidationError: threshold must be 0-1, got 1.5"
        );
    }

    #[test]
    fn test_normalize_scores_empty() {
        let scores = normalize_scores(&[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_normalize_scores_single() {
        let scores = normalize_scores(&[5.0]);
        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn test_normalize_scores_normal() {
        let scores = normalize_scores(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(scores, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_normalize_scores_negative() {
        let scores = normalize_scores(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(scores, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_normalize_scores_all_same() {
        let scores = normalize_scores(&[3.0, 3.0, 3.0]);
        assert_eq!(scores, vec![0.0, 0.0, 0.0]);
    }
}
