//! FP-Growth frequent pattern mining algorithm.
//!
//! This module provides an implementation of the FP-Growth algorithm for
//! discovering frequent itemsets in transaction data (e.g., market basket analysis).
//!
//! # Quick start
//! ```rust
//! use fast_insight_engine::fp_growth::FPGrowth;
//!
//! let transactions = vec![
//!     vec!["milk", "bread", "butter"],
//!     vec!["milk", "bread"],
//!     vec!["milk"],
//! ];
//! let fp = FPGrowth::<&str>::new(transactions, 2);
//! let result = fp.find_frequent_patterns();
//! println!("{} patterns found", result.frequent_patterns_num());
//! ```

use std::fmt::Debug;
use std::hash::Hash;

pub mod algorithm;
pub mod tree;

pub use algorithm::{FPGrowth, FPResult};

/// Trait bound for items used in FP-Growth transactions.
///
/// Any type satisfying `Copy + Eq + Hash + Ord + Debug + Send + Sync + 'static`
/// automatically implements this trait via the blanket implementation.
///
/// Common implementations: `u32`, `u64`, `i32`, `&'static str`, `char`.
pub trait ItemType: Copy + Eq + Hash + Ord + Debug + Send + Sync + 'static {}

impl<T: Copy + Eq + Hash + Ord + Debug + Send + Sync + 'static> ItemType for T {}
