# fast-insight-engine

A high-performance data processing engine built in Rust for generating insights from datasets. The engine provides fast data ingestion, statistical analysis, and aggregation capabilities for CSV and JSON data sources.

## Features

- 🚀 **Fast Data Processing**: Built with Rust for maximum performance
- 📊 **Statistical Analysis**: Compute mean, min, max, sum, and count
- 📈 **Aggregations**: Support for sum, average, min, max, and count operations
- 📁 **Multiple Formats**: Load data from CSV and JSON files
- 🔍 **Filtering**: Filter datasets based on custom predicates
- 🛠️ **CLI Interface**: Easy-to-use command-line tool

## Installation

Ensure you have Rust installed (1.70 or later). Then build the project:

```bash
cargo build --release
```

## Usage

### As a Command-Line Tool

#### Load and Analyze CSV Data

```bash
cargo run -- csv --name employees --file examples/employees.csv --analyze age
```

Output:
```
Loaded CSV dataset 'employees' with 10 records
Fields: ["age", "department", "name", "salary"]

=== Statistics for 'age' ===
Count: 10
Sum:   296.00
Mean:  29.60
Min:   25.00
Max:   35.00
```

#### Load and Analyze JSON Data

```bash
cargo run -- json --name products --file examples/products.json --analyze price
```

Output:
```
Loaded JSON dataset 'products' with 5 records
Fields: ["category", "price", "product", "quantity"]

=== Statistics for 'price' ===
Count: 5
Sum:   1609.95
Mean:  321.99
Min:   29.99
Max:   999.99
```

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
fast-insight-engine = "0.1.0"
```

Example usage:

```rust
use fast_insight_engine::{Dataset, InsightEngine};

fn main() -> anyhow::Result<()> {
    // Create an engine
    let mut engine = InsightEngine::new();

    // Load CSV data
    let csv_data = "name,age,score\nAlice,30,95\nBob,25,87";
    let dataset = Dataset::from_csv("test".to_string(), csv_data)?;
    engine.add_dataset(dataset);

    // Compute statistics
    if let Some(stats) = engine.compute_stats("test", "age") {
        println!("Mean age: {:.2}", stats.mean);
        println!("Min age: {:.2}", stats.min);
        println!("Max age: {:.2}", stats.max);
    }

    Ok(())
}
```

## API Overview

### Core Components

- **`DataPoint`**: Represents a single data record with named fields
- **`Dataset`**: A collection of data points with methods for loading from CSV/JSON
- **`InsightEngine`**: Main engine for managing and analyzing datasets
- **`Statistics`**: Statistical computations (mean, min, max, sum, count)

### Key Methods

```rust
// Load data
let dataset = Dataset::from_csv(name, csv_string)?;
let dataset = Dataset::from_json(name, json_string)?;

// Create engine and add datasets
let mut engine = InsightEngine::new();
engine.add_dataset(dataset);

// Compute statistics
let stats = engine.compute_stats("dataset_name", "field_name");

// Perform aggregations
use fast_insight_engine::stats::AggregateOp;
let sum = engine.aggregate("dataset_name", "field", AggregateOp::Sum);
let avg = engine.aggregate("dataset_name", "field", AggregateOp::Average);

// Filter datasets
let filtered = engine.filter("dataset_name", "field", |value| value.starts_with("A"));

// Get dataset summary
let summaries = engine.summary();
```

## Examples

See the `examples/` directory for sample data files:
- `employees.csv` - Employee data with age, salary, and department information
- `products.json` - Product catalog with prices and quantities

## Testing

Run the test suite:

```bash
cargo test
```

## Performance

The engine is designed for speed:
- Zero-copy data processing where possible
- Efficient memory usage with Rust's ownership model
- Parallel-ready architecture for future enhancements

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

