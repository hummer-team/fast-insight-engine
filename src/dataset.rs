use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a single data point with named fields
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataPoint {
    pub fields: HashMap<String, String>,
}

impl DataPoint {
    /// Create a new data point
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Add a field to the data point
    pub fn add_field(&mut self, key: String, value: String) {
        self.fields.insert(key, value);
    }

    /// Get a field value
    pub fn get_field(&self, key: &str) -> Option<&String> {
        self.fields.get(key)
    }

    /// Parse a numeric field value
    pub fn get_numeric(&self, key: &str) -> Option<f64> {
        self.get_field(key)?.parse().ok()
    }
}

impl Default for DataPoint {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of data points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub data: Vec<DataPoint>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(name: String) -> Self {
        Self {
            name,
            data: Vec::new(),
        }
    }

    /// Add a data point to the dataset
    pub fn add_point(&mut self, point: DataPoint) {
        self.data.push(point);
    }

    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get all unique field names across all data points
    pub fn get_field_names(&self) -> Vec<String> {
        let mut fields = std::collections::HashSet::new();
        for point in &self.data {
            for key in point.fields.keys() {
                fields.insert(key.clone());
            }
        }
        let mut result: Vec<String> = fields.into_iter().collect();
        result.sort();
        result
    }

    /// Load dataset from CSV
    pub fn from_csv(name: String, csv_data: &str) -> crate::Result<Self> {
        let mut dataset = Dataset::new(name);
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_data.as_bytes());

        let headers = reader.headers()?.clone();

        for result in reader.records() {
            let record = result?;
            let mut point = DataPoint::new();

            for (i, field) in record.iter().enumerate() {
                if let Some(header) = headers.get(i) {
                    point.add_field(header.to_string(), field.to_string());
                }
            }
            dataset.add_point(point);
        }

        Ok(dataset)
    }

    /// Load dataset from JSON array
    pub fn from_json(name: String, json_data: &str) -> crate::Result<Self> {
        let mut dataset = Dataset::new(name);
        let data: Vec<HashMap<String, serde_json::Value>> = serde_json::from_str(json_data)?;

        for item in data {
            let mut point = DataPoint::new();
            for (key, value) in item {
                let value_str = match value {
                    serde_json::Value::String(s) => s,
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Null => "null".to_string(),
                    _ => value.to_string(),
                };
                point.add_field(key, value_str);
            }
            dataset.add_point(point);
        }

        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_creation() {
        let mut point = DataPoint::new();
        point.add_field("name".to_string(), "test".to_string());
        point.add_field("value".to_string(), "42".to_string());

        assert_eq!(point.get_field("name"), Some(&"test".to_string()));
        assert_eq!(point.get_numeric("value"), Some(42.0));
    }

    #[test]
    fn test_dataset_creation() {
        let mut dataset = Dataset::new("test".to_string());
        let mut point = DataPoint::new();
        point.add_field("id".to_string(), "1".to_string());

        dataset.add_point(point);
        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_csv_loading() {
        let csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago";
        let dataset = Dataset::from_csv("users".to_string(), csv_data).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.data[0].get_field("name"), Some(&"Alice".to_string()));
        assert_eq!(dataset.data[1].get_numeric("age"), Some(25.0));
    }

    #[test]
    fn test_json_loading() {
        let json_data = r#"[
            {"name": "Alice", "age": 30, "city": "NYC"},
            {"name": "Bob", "age": 25, "city": "LA"}
        ]"#;
        let dataset = Dataset::from_json("users".to_string(), json_data).unwrap();

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.data[0].get_field("name"), Some(&"Alice".to_string()));
        assert_eq!(dataset.data[1].get_numeric("age"), Some(25.0));
    }

    #[test]
    fn test_get_field_names() {
        let mut dataset = Dataset::new("test".to_string());
        let mut point1 = DataPoint::new();
        point1.add_field("a".to_string(), "1".to_string());
        point1.add_field("b".to_string(), "2".to_string());

        let mut point2 = DataPoint::new();
        point2.add_field("b".to_string(), "3".to_string());
        point2.add_field("c".to_string(), "4".to_string());

        dataset.add_point(point1);
        dataset.add_point(point2);

        let fields = dataset.get_field_names();
        assert_eq!(fields, vec!["a", "b", "c"]);
    }
}
