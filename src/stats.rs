use crate::dataset::Dataset;

/// Statistics computed from a dataset
#[derive(Debug, Clone)]
pub struct Statistics {
    pub field: String,
    pub count: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
}

impl Statistics {
    /// Compute statistics for a numeric field in a dataset
    pub fn compute(dataset: &Dataset, field: &str) -> Option<Self> {
        let values: Vec<f64> = dataset
            .data
            .iter()
            .filter_map(|point| point.get_numeric(field))
            .collect();

        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Some(Statistics {
            field: field.to_string(),
            count,
            mean,
            min,
            max,
            sum,
        })
    }
}

/// Aggregation operations
pub enum AggregateOp {
    Sum,
    Average,
    Count,
    Min,
    Max,
}

/// Perform aggregation on a dataset field
pub fn aggregate(dataset: &Dataset, field: &str, op: AggregateOp) -> Option<f64> {
    let values: Vec<f64> = dataset
        .data
        .iter()
        .filter_map(|point| point.get_numeric(field))
        .collect();

    if values.is_empty() {
        return None;
    }

    match op {
        AggregateOp::Sum => Some(values.iter().sum()),
        AggregateOp::Average => Some(values.iter().sum::<f64>() / values.len() as f64),
        AggregateOp::Count => Some(values.len() as f64),
        AggregateOp::Min => values.iter().copied().fold(None, |acc, x| {
            Some(acc.map_or(x, |a| a.min(x)))
        }),
        AggregateOp::Max => values.iter().copied().fold(None, |acc, x| {
            Some(acc.map_or(x, |a| a.max(x)))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DataPoint;

    fn create_test_dataset() -> Dataset {
        let mut dataset = Dataset::new("test".to_string());
        for value in [10.0, 20.0, 30.0, 40.0, 50.0] {
            let mut point = DataPoint::new();
            point.add_field("value".to_string(), value.to_string());
            dataset.add_point(point);
        }
        dataset
    }

    #[test]
    fn test_statistics_compute() {
        let dataset = create_test_dataset();
        let stats = Statistics::compute(&dataset, "value").unwrap();

        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
        assert_eq!(stats.sum, 150.0);
    }

    #[test]
    fn test_aggregate_sum() {
        let dataset = create_test_dataset();
        let result = aggregate(&dataset, "value", AggregateOp::Sum);
        assert_eq!(result, Some(150.0));
    }

    #[test]
    fn test_aggregate_average() {
        let dataset = create_test_dataset();
        let result = aggregate(&dataset, "value", AggregateOp::Average);
        assert_eq!(result, Some(30.0));
    }

    #[test]
    fn test_aggregate_min_max() {
        let dataset = create_test_dataset();
        assert_eq!(aggregate(&dataset, "value", AggregateOp::Min), Some(10.0));
        assert_eq!(aggregate(&dataset, "value", AggregateOp::Max), Some(50.0));
    }

    #[test]
    fn test_statistics_empty_dataset() {
        let dataset = Dataset::new("empty".to_string());
        let stats = Statistics::compute(&dataset, "value");
        assert!(stats.is_none());
    }
}
