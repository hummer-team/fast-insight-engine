//! `algorithm` is the core module of FP-Growth algorithm.
//! It implements the algorithm based on the internal data structs [`crate::tree::Node<T>`] and [`crate::tree::Tree<T>`].

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    usize,
};

use super::ItemType;
use super::tree::Tree;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct FPResult<T> {
    frequent_patterns: Vec<(Vec<T>, usize)>,
    elimination_sets: HashSet<Vec<T>>,
}

impl<T: ItemType> FPResult<T> {
    pub fn new(
        frequent_patterns: Vec<(Vec<T>, usize)>,
        elimination_sets: HashSet<Vec<T>>,
    ) -> FPResult<T> {
        FPResult {
            frequent_patterns,
            elimination_sets,
        }
    }

    pub fn frequent_patterns_num(&self) -> usize {
        self.frequent_patterns.len()
    }

    pub fn frequent_patterns(&self) -> Vec<(Vec<T>, usize)> {
        self.frequent_patterns.clone()
    }

    pub fn elimination_sets_num(&self) -> usize {
        self.elimination_sets.len()
    }

    pub fn elimination_sets(&self) -> Vec<Vec<T>> {
        self.elimination_sets.clone().into_iter().collect()
    }
}

/// `FPGrowth<T>` represents an algorithm instance, it should include the `transactions` input
/// and minimum support value as the initial config. Once it is created, you could run
/// [`FPGrowth::find_frequent_patterns()`] to start the frequent pattern mining.
// `transactions` will be sorted and deduplicated before starting the algorithm.
#[allow(clippy::upper_case_acronyms)]
pub struct FPGrowth<T> {
    transactions: Vec<Vec<T>>,
    minimum_support: usize,
}

impl<T: ItemType> FPGrowth<T> {
    /// Create a FP-Growth algorithm instance with the given `transactions` and `minimum_support`.
    pub fn new(transactions: Vec<Vec<T>>, minimum_support: usize) -> FPGrowth<T> {
        FPGrowth {
            transactions,
            minimum_support,
        }
    }

    /// Find frequent patterns in the given transactions using FP-Growth.
    pub fn find_frequent_patterns(&self) -> FPResult<T> {
        // Collect and preprocess the transactions.
        let mut items = HashMap::new();
        for transaction in self.transactions.clone().into_iter() {
            let mut item_set: HashSet<T> = HashSet::new();
            for &item in transaction.iter() {
                // Check whether we have inserted the same item in a transaction before,
                // make sure we won't calculate the wrong support.
                match item_set.contains(&item) {
                    true => continue,
                    false => {
                        item_set.insert(item);
                        let count = items.entry(item).or_insert(0);
                        *count += 1;
                    }
                };
            }
        }

        // Clean up the items whose support is lower than the minimum_support.
        let cleaned_items: HashMap<&T, &usize> = items
            .iter()
            .filter(|&(_, count)| *count >= self.minimum_support)
            .collect();
        let mut elimination_sets = HashSet::new();

        let mut tree = Tree::<T>::new();
        for transaction in self.transactions.clone().into_iter() {
            let mut cleaned_transaction: Vec<T> = transaction
                .clone()
                .into_iter()
                .filter(|item| cleaned_items.contains_key(item))
                .collect();
            if cleaned_transaction.len() != transaction.len() {
                elimination_sets.insert(transaction);
            }
            cleaned_transaction.sort_by(|a, b| {
                let a_counter = cleaned_items.get(a).map_or(0usize, |&&v| v);
                let b_counter = cleaned_items.get(b).map_or(0usize, |&&v| v);
                match b_counter.cmp(&a_counter) {
                    Ordering::Equal => match b.cmp(a) {
                        Ordering::Greater => Ordering::Less,
                        Ordering::Less => Ordering::Greater,
                        Ordering::Equal => Ordering::Equal,
                    },
                    other => other,
                }
            });
            // After sort cleaned_transaction, remove consecutive items from it then.
            cleaned_transaction.dedup();
            tree.add_transaction(cleaned_transaction);
        }

        let mut fp_result = self.find_with_suffix(&tree, &[]);
        fp_result.elimination_sets.extend(elimination_sets);
        fp_result
    }

    fn find_with_suffix(&self, tree: &Tree<T>, suffix: &[T]) -> FPResult<T> {

        let mut fp_result = FPResult::new(vec![], HashSet::new());
        for (item, nodes) in tree.get_all_items_nodes().iter() {
            let mut support = 0;
            for node in nodes.iter() {
                support += node.count();
            }
            let mut frequent_pattern = vec![*item];
            frequent_pattern.append(&mut Vec::from(suffix));
            if support >= self.minimum_support && !suffix.contains(item) {
                fp_result
                    .frequent_patterns
                    .push((frequent_pattern.clone(), support));

                let partial_tree = Tree::generate_partial_tree(&tree.generate_prefix_path(*item));
                let mut mid_fp_result = self.find_with_suffix(&partial_tree, &frequent_pattern);
                fp_result
                    .frequent_patterns
                    .append(&mut mid_fp_result.frequent_patterns);
                fp_result
                    .elimination_sets
                    .extend(mid_fp_result.elimination_sets);
            } else {
                fp_result.elimination_sets.insert(frequent_pattern);
            }
        }
        fp_result
    }
}

#[cfg(test)]
mod tests {
    use super::FPGrowth;

    /// Canonical test fixture matching the reference Python implementation.
    fn sample_transactions() -> Vec<Vec<&'static str>> {
        vec![
            vec!["e", "c", "a", "b", "f", "h"],
            vec!["a", "c", "g"],
            vec!["e"],
            vec!["e", "c", "a", "g", "d"],
            vec!["a", "c", "e", "g"],
            vec!["e"],
            vec!["a", "c", "e", "b", "f"],
            vec!["a", "c", "d"],
            vec!["g", "c", "e", "a"],
            vec!["a", "c", "e", "g"],
            vec!["i"],
        ]
    }

    /// Normal scenario: verify that known frequent patterns are found.
    ///
    /// Input: 11 transactions, min_support=2
    /// Expected: "a" (8 transactions) and "e" (7 transactions) are frequent singletons.
    #[test]
    fn test_normal_finds_frequent_patterns() {
        let fp = FPGrowth::<&str>::new(sample_transactions(), 2);
        let result = fp.find_frequent_patterns();

        assert!(
            result.frequent_patterns_num() > 0,
            "expected non-zero frequent patterns for min_support=2"
        );

        let patterns = result.frequent_patterns();
        let has_a = patterns.iter().any(|(p, sup)| p == &vec!["a"] && *sup >= 2);
        assert!(has_a, "item 'a' should be a frequent singleton pattern");

        let has_e = patterns.iter().any(|(p, sup)| p == &vec!["e"] && *sup >= 2);
        assert!(has_e, "item 'e' should be a frequent singleton pattern");
    }

    /// High min_support: no pattern should meet the threshold.
    ///
    /// Input: 11 transactions, min_support=999
    /// Expected: frequent_patterns_num() == 0
    #[test]
    fn test_high_min_support_returns_empty() {
        let fp = FPGrowth::<&str>::new(sample_transactions(), 999);
        let result = fp.find_frequent_patterns();
        assert_eq!(
            result.frequent_patterns_num(),
            0,
            "expected 0 patterns when min_support exceeds all transaction counts"
        );
    }

    /// Duplicate item within a single transaction must not be double-counted.
    ///
    /// Input: [["a","a","b"], ["a","b"]], min_support=2
    /// Expected: support("a") == 2 (deduplicated per transaction, not 3).
    #[test]
    fn test_duplicate_item_in_transaction_not_double_counted() {
        let transactions = vec![
            vec!["a", "a", "b"], // "a" appears twice — must count as 1
            vec!["a", "b"],
        ];
        let fp = FPGrowth::<&str>::new(transactions, 2);
        let result = fp.find_frequent_patterns();

        let patterns = result.frequent_patterns();
        let a_support = patterns
            .iter()
            .find(|(p, _)| p == &vec!["a"])
            .map(|(_, s)| *s);
        assert_eq!(a_support, Some(2), "'a' support should be 2 (dedup'd), not 3");
    }

    /// Integer item type (u32) works via the blanket ItemType implementation.
    ///
    /// Input: [[1,2,3],[1,2],[1]], min_support=2
    /// Expected: item 1 (support 3) and item 2 (support 2) are frequent.
    #[test]
    fn test_u32_item_type() {
        let transactions = vec![vec![1u32, 2, 3], vec![1, 2], vec![1]];
        let fp = FPGrowth::<u32>::new(transactions, 2);
        let result = fp.find_frequent_patterns();

        let patterns = result.frequent_patterns();
        let has_1 = patterns.iter().any(|(p, _)| p == &vec![1u32]);
        assert!(has_1, "item 1 should be a frequent pattern with support 3");

        let has_2 = patterns.iter().any(|(p, _)| p == &vec![2u32]);
        assert!(has_2, "item 2 should be a frequent pattern with support 2");
    }

    /// Single transaction with min_support=1: all items must be frequent.
    ///
    /// Input: [["x","y","z"]], min_support=1
    /// Expected: at least 3 patterns (one per singleton item).
    #[test]
    fn test_single_transaction_all_frequent() {
        let transactions = vec![vec!["x", "y", "z"]];
        let fp = FPGrowth::<&str>::new(transactions, 1);
        let result = fp.find_frequent_patterns();
        assert!(
            result.frequent_patterns_num() >= 3,
            "all 3 items must be frequent when min_support=1"
        );
    }
}
