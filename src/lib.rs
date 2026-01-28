//! Fast Insight Engine - A high-performance data processing engine
//!
//! This library provides fast data ingestion, processing, and analytics capabilities
//! for generating insights from various data sources.

pub mod dataset;
pub mod engine;
pub mod stats;

pub use dataset::{DataPoint, Dataset};
pub use engine::InsightEngine;
pub use stats::Statistics;

/// Result type used throughout the library
pub type Result<T> = anyhow::Result<T>;
