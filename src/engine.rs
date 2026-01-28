use crate::dataset::Dataset;
use crate::stats::{Statistics, aggregate, AggregateOp};
use std::collections::HashMap;

/// The main insight engine for processing datasets
#[derive(Debug)]
pub struct InsightEngine {
    datasets: HashMap<String, Dataset>,
}

impl InsightEngine {
    /// Create a new insight engine
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    /// Add a dataset to the engine
    pub fn add_dataset(&mut self, dataset: Dataset) {
        self.datasets.insert(dataset.name.clone(), dataset);
    }

    /// Get a dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    /// Remove a dataset from the engine
    pub fn remove_dataset(&mut self, name: &str) -> Option<Dataset> {
        self.datasets.remove(name)
    }

    /// Get all dataset names
    pub fn list_datasets(&self) -> Vec<String> {
        let mut names: Vec<String> = self.datasets.keys().cloned().collect();
        names.sort();
        names
    }

    /// Compute statistics for a field in a dataset
    pub fn compute_stats(&self, dataset_name: &str, field: &str) -> Option<Statistics> {
        let dataset = self.get_dataset(dataset_name)?;
        Statistics::compute(dataset, field)
    }

    /// Perform aggregation on a dataset field
    pub fn aggregate(&self, dataset_name: &str, field: &str, op: AggregateOp) -> Option<f64> {
        let dataset = self.get_dataset(dataset_name)?;
        aggregate(dataset, field, op)
    }

    /// Get a summary of all datasets
    pub fn summary(&self) -> Vec<DatasetSummary> {
        self.datasets
            .values()
            .map(|dataset| DatasetSummary {
                name: dataset.name.clone(),
                record_count: dataset.len(),
                fields: dataset.get_field_names(),
            })
            .collect()
    }

    /// Filter dataset by a predicate on a field
    pub fn filter(&self, dataset_name: &str, field: &str, predicate: fn(&str) -> bool) -> Option<Dataset> {
        let dataset = self.get_dataset(dataset_name)?;
        let mut filtered = Dataset::new(format!("{}_filtered", dataset_name));

        for point in &dataset.data {
            if let Some(value) = point.get_field(field) {
                if predicate(value) {
                    filtered.add_point(point.clone());
                }
            }
        }

        Some(filtered)
    }
}

impl Default for InsightEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary information about a dataset
#[derive(Debug, Clone)]
pub struct DatasetSummary {
    pub name: String,
    pub record_count: usize,
    pub fields: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_dataset() -> Dataset {
        let csv_data = "name,age,score\nAlice,30,95\nBob,25,87\nCharlie,35,92";
        Dataset::from_csv("test".to_string(), csv_data).unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let engine = InsightEngine::new();
        assert_eq!(engine.list_datasets().len(), 0);
    }

    #[test]
    fn test_add_and_get_dataset() {
        let mut engine = InsightEngine::new();
        let dataset = create_sample_dataset();
        engine.add_dataset(dataset);

        assert_eq!(engine.list_datasets(), vec!["test"]);
        assert!(engine.get_dataset("test").is_some());
        assert_eq!(engine.get_dataset("test").unwrap().len(), 3);
    }

    #[test]
    fn test_remove_dataset() {
        let mut engine = InsightEngine::new();
        let dataset = create_sample_dataset();
        engine.add_dataset(dataset);

        let removed = engine.remove_dataset("test");
        assert!(removed.is_some());
        assert_eq!(engine.list_datasets().len(), 0);
    }

    #[test]
    fn test_compute_stats() {
        let mut engine = InsightEngine::new();
        let dataset = create_sample_dataset();
        engine.add_dataset(dataset);

        let stats = engine.compute_stats("test", "age").unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 30.0);
    }

    #[test]
    fn test_aggregate() {
        let mut engine = InsightEngine::new();
        let dataset = create_sample_dataset();
        engine.add_dataset(dataset);

        let sum = engine.aggregate("test", "score", AggregateOp::Sum);
        assert_eq!(sum, Some(274.0));

        let avg = engine.aggregate("test", "score", AggregateOp::Average);
        assert!((avg.unwrap() - 91.333).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let mut engine = InsightEngine::new();
        let dataset = create_sample_dataset();
        engine.add_dataset(dataset);

        let summaries = engine.summary();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].name, "test");
        assert_eq!(summaries[0].record_count, 3);
    }

    #[test]
    fn test_filter() {
        let mut engine = InsightEngine::new();
        let dataset = create_sample_dataset();
        engine.add_dataset(dataset);

        let filtered = engine.filter("test", "name", |name| name.starts_with('A')).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.data[0].get_field("name"), Some(&"Alice".to_string()));
    }
}
