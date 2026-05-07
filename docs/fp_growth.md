# FP-Growth — Frequent Pattern Mining / Product Association Prediction

## Overview

FP-Growth (Frequent Pattern Growth) is an efficient frequent itemset mining algorithm that avoids
repeated dataset scanning by compressing transactions into an FP-tree structure.

**Primary use case in this project**: Market basket analysis — discover which products are frequently
purchased together to drive cross-sell recommendations.

---

## Data Flow

```
TypeScript (Arrow IPC Stream)
  ┌─────────────────────────────────┐
  │ order_id: Utf8 | item_id: Utf8  │  ← one row per (order, item) pair
  │ "ORD001"       | "milk"         │
  │ "ORD001"       | "bread"        │
  │ "ORD002"       | "bread"        │
  │ "ORD002"       | "eggs"         │
  └─────────────────────────────────┘
        ↓  find_association_patterns(data, min_support)
  ┌─────────────────────────────────────────┐
  │ pattern: List<Utf8>  | support: Int64   │
  │ ["milk", "bread"]    | 5                │
  │ ["milk"]             | 8                │
  │ ["bread"]            | 7                │
  └─────────────────────────────────────────┘
```

---

## Wasm API Reference

### `find_association_patterns(data, min_support)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `Uint8Array` | Arrow IPC Stream bytes. Schema: `order_id: Utf8, item_id: Utf8` (long format, one row per order-item pair) |
| `min_support` | `number` | Minimum support threshold (integer ≥ 1). A pattern must appear in at least N distinct orders to be considered frequent. |

**Returns**: `Promise<Uint8Array>` — Arrow IPC Stream with schema: `pattern: List<Utf8>, support: Int64`

**Throws**: `JsError` if the input Arrow schema is invalid or data is malformed.

---

## TypeScript Usage Example

```typescript
import { tableToIPC, tableFromIPC } from "@apache-arrow/es2015-esm";
import * as arrow from "@apache-arrow/es2015-esm";
import init, { find_association_patterns } from "./pkg/fast_insight_engine.js";

await init();

// 1. Build input Arrow Table (long format: one row per order-item pair)
const orderIds = ["ORD001", "ORD001", "ORD001", "ORD002", "ORD002", "ORD003", "ORD003", "ORD003"];
const itemIds  = ["milk",   "bread",  "butter", "milk",   "bread",  "bread",  "butter", "eggs"];

const inputTable = new arrow.Table({
  order_id: arrow.vectorFromArray(orderIds, new arrow.Utf8()),
  item_id:  arrow.vectorFromArray(itemIds,  new arrow.Utf8()),
});

const inputBytes = tableToIPC(inputTable, "stream");

// 2. Run FP-Growth (patterns appearing in at least 2 distinct orders)
const resultBytes = await find_association_patterns(inputBytes, 2);

// 3. Parse and use results
const resultTable = tableFromIPC(resultBytes);
for (const row of resultTable) {
  console.log(`Pattern: ${JSON.stringify([...row.pattern])}, Support: ${row.support}`);
}
// Example output (order may vary):
// Pattern: ["milk","bread"], Support: 2
// Pattern: ["bread","butter"], Support: 2
// Pattern: ["milk"], Support: 2
// Pattern: ["bread"], Support: 3
// Pattern: ["butter"], Support: 2
```

### Recommended: Web Worker (avoids blocking the UI thread)

```typescript
// worker.ts
import init, { find_association_patterns } from "../pkg/fast_insight_engine.js";

self.onmessage = async ({ data: { inputBytes, minSupport } }) => {
  await init();
  try {
    const result = await find_association_patterns(inputBytes, minSupport);
    self.postMessage({ ok: true, result }, [result.buffer]);
  } catch (err) {
    self.postMessage({ ok: false, error: String(err) });
  }
};

// main.ts
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
worker.postMessage({ inputBytes, minSupport: 2 });
worker.onmessage = ({ data }) => {
  if (!data.ok) { console.error(data.error); return; }
  const table = tableFromIPC(data.result);
  console.table([...table]);
};
```

---

## Rust Direct Usage

```rust
use fast_insight_engine::fp_growth::FPGrowth;

fn main() {
    let transactions = vec![
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
    ];

    let minimum_support = 2;
    let fp = FPGrowth::<&str>::new(transactions, minimum_support);
    let result = fp.find_frequent_patterns();

    println!("Frequent patterns found: {}", result.frequent_patterns_num());
    for (pattern, support) in result.frequent_patterns() {
        println!("  {:?}  support={}", pattern, support);
    }
    println!("Eliminated sets: {}", result.elimination_sets_num());
}
```

**Expected output** (pattern order may vary):
```
Frequent patterns found: N
  ["a"]       support=8
  ["c"]       support=8
  ["e"]       support=7
  ["g"]       support=5
  ["a", "c"]  support=7
  ["a", "e"]  support=6
  ...
```

---

## Parameter Selection Guide

| Scenario | Recommended `min_support` |
|----------|--------------------------|
| Small store (< 1k orders) | 2–5 |
| Mid-size (1k–10k orders) | 5–20 |
| Large (10k–100k orders) | 20–100 |
| Exploratory analysis | Start at 1%–2% of total order count |

> **Rule of thumb**: too low → noisy patterns; too high → misses real associations.

---

## Performance Characteristics

| Dataset | Expected time (browser CPU) |
|---------|----------------------------|
| 1k orders × 20 items | < 10ms |
| 10k orders × 100 items | ~50ms |
| 100k orders × 200 items | ~1–3s |

Use a Web Worker for datasets larger than ~5k rows to avoid blocking the UI thread.

---

## Algorithm Summary

1. **Count frequencies**: Scan all transactions, count each item, discard items below `min_support`.
2. **Build FP-Tree**: Insert each transaction (sorted by descending frequency) into a prefix tree; nodes with the same item are linked via a neighbor chain.
3. **Mine recursively**: For each frequent item, extract conditional pattern bases (prefix paths), build a conditional FP-Tree, and recurse to find multi-item patterns.
4. **Output**: All itemsets with `support ≥ min_support` plus their support counts.

---

## Notes

- Duplicate items within the same transaction are deduplicated before counting.
- `min_support = 0` will return an error (all items would qualify, which is meaningless).
- Output pattern order is determined by internal frequency sorting, not lexicographic order.
- Internally uses `u32` indices for item mapping; maximum distinct item count is `u32::MAX`.
