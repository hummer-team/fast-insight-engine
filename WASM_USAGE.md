# Wasm TypeScript 集成示例

本文档展示如何在 TypeScript/JavaScript 中使用生成的 Wasm 模块。

## 模块导入

```typescript
// 导入 Wasm 模块和类型定义
import * as fastInsight from '@hummer-team/fast-insight-engine';
```

## 六个导出函数

### 1. CSV → Parquet 转换 ✨ (新增)

```typescript
const csvBytes = /* CSV file bytes */;

const result = await fastInsight.convert_csv_to_parquet(
  csvBytes,
  44,         // delimiter (44=',' | 9='\t' | 124='|' | 59=';')
  true,       // has_header
  1024        // row_group_size (64-16384)
);

// 返回: Promise<Uint8Array> (Parquet 格式)
```

**关键点**:
- **在 Node.js 中**: 真正执行 CSV→Parquet 转换，调用 `file_convert::Converter`
- **在 Wasm 中**: 返回错误提示，建议使用 DuckDB Wasm 或 papaparse

### 2. Excel → Parquet 转换 ✨ (新增)

```typescript
const excelBytes = /* XLSX/XLS file bytes */;

const result = await fastInsight.convert_excel_to_parquet(
  excelBytes,
  "Sheet1",   // sheet name (或空字符串 "" 表示第一个 sheet)
  true,       // has_header
  1024        // row_group_size (64-16384)
);

// 返回: Promise<Uint8Array> (Parquet 格式)
```

**关键点**:
- **在 Node.js 中**: 真正执行 Excel→Parquet 转换，支持 XLSX 和 XLS 格式
- **在 Wasm 中**: 返回错误提示，建议使用 xlsx/exceljs 库

### 3. 异常订单检测 (Isolation Forest)

```typescript
// 假设 data 是 Arrow IPC Stream 格式的 Uint8Array
// 其中包含特征列（price, quantity, customer_id 等）
const data: Uint8Array = /* Arrow IPC Stream bytes */;

const result = await fastInsight.detect_order_anomalies(
  data,
  0.7,        // 异常分数阈值
  2,          // 使用 Standard scaling
  false       // 不使用 GPU
);

// result 是 Arrow IPC Stream 格式的 Uint8Array
```

### 4. 订单分组聚类 (K-Means)

```typescript
const useGPU = rowCount > 5000 && navigator.gpu !== undefined;

const result = await fastInsight.segment_customer_orders(
  data,
  5,          // 聚类数（k=5）
  2,          // 使用 Standard scaling
  useGPU      // 如果可用则启用 GPU
);
```

### 5. 库存需求预测 (Multi-mode Regression)

支持四种预测模式，通过 `prediction_mode` 和 `season_period` 参数控制：

| `prediction_mode` | 模式 | 适用场景 |
|---|---|---|
| `0` | Linear | 稳定增长/下降趋势 |
| `1` | Polynomial | S 型增长、需求饱和 |
| `2` | Seasonal | 周期性波动（需设置 `season_period`） |
| `3` | Ensemble | 多项式 + 季节性（推荐用于生产环境） |

#### 输入格式

`data` 为 **Arrow IPC Stream**（`Uint8Array`），必须包含 **至少 2 列 Float64**：

| 列索引 | 列名（任意） | 说明 |
|---|---|---|
| col[0] | `time_index` | 时间占位列，**值被忽略**，传全 0 即可 |
| col[1] | `demand` | 历史需求量，**实际用于训练** |

#### 输出格式

返回 **Arrow IPC Stream**（`Uint8Array`），包含 **1 列**：

| 列名 | 类型 | 说明 |
|---|---|---|
| `prediction` | Float64 | 未来 `predict_steps` 个时间步的预测值 |

#### 完整示例

以 12 个月历史销售数据预测未来 3 个月为例：

```typescript
import init, * as fastInsight from './pkg/fast_insight_engine.js';
import {
  tableFromIPC,
  tableToIPC,
  Table,
  Schema,
  Field,
  Float64,
  makeData,
} from 'apache-arrow';

async function forecastDemand() {
  await init();

  // 1. 准备历史需求数据（12 个月）
  const historicalDemand = [
    120, 135, 128, 142, 160, 175,
    168, 183, 195, 210, 225, 238,
  ];

  // col[0]: 时间占位列（值任意，函数内部忽略，传全 0 即可）
  // col[1]: 历史需求量（实际训练用）
  const timePlaceholder = new Float64Array(historicalDemand.length).fill(0);
  const demandValues    = new Float64Array(historicalDemand);

  // 2. 序列化为 Arrow IPC Stream
  const schema = new Schema([
    Field.new('time_index', new Float64()),
    Field.new('demand',     new Float64()),
  ]);

  const inputTable = new Table(schema, [
    makeData({ type: new Float64(), data: timePlaceholder }),
    makeData({ type: new Float64(), data: demandValues }),
  ]);

  const inputBytes = tableToIPC(inputTable, 'stream'); // Uint8Array

  // 3. 调用预测（Ensemble 模式，年周期=12 个月）
  const outputBytes = await fastInsight.predict_inventory_demand(
    inputBytes,
    3,    // predict_steps: 预测未来 3 个月
    0,    // scaling_mode:  0=None, 1=MinMax, 2=Standard
    3,    // prediction_mode: 3=Ensemble（推荐）
    12    // season_period: 12=年周期（月粒度）
  );

  // 4. 解析输出
  const resultTable = tableFromIPC(outputBytes);
  const predictions = Array.from(
    resultTable.getChild('prediction')!.toArray() as Float64Array
  );

  console.log('未来 3 个月预测需求:', predictions);
  // 示例输出 → [251.4, 264.8, 278.2]
}

forecastDemand().catch(console.error);
```

> **提示**：
> - `season_period=0` 时自动默认为 7（周粒度）
> - 对于大数据集（>10k 行）建议在 Web Worker 中调用以避免阻塞主线程
> - `prediction_mode=0` 与旧版线性回归行为完全兼容

### 5b. 批量多SKU库存预测 (Batch Multi-SKU Prediction)

预测多个SKU的库存需求，单次调用处理所有SKU，每个SKU独立运算。失败的SKU以错误行返回，不会中断整个批次。

#### 函数签名

```typescript
predict_inventory_demand_batch(
  data: Uint8Array,        // Arrow IPC Stream 格式输入
  predict_steps: number,   // 每个SKU预测步数 (≥ 1)
  mode: string,            // 预测模式: "linear" | "polynomial_2" | "polynomial_3" | "exponential"
  scaling: string,         // 缩放方式: "none" | "min_max" | "standard"
  threshold: number        // 保留参数，传 0.0
): Promise<Uint8Array>     // Arrow IPC Stream 格式输出
```

#### 输入 Arrow Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `sku_id` | `Utf8` | SKU 标识符（相同 sku_id 的行属于同一时间序列） |
| `time_index` | `Float64` | 时间索引（库内保留，可传 0,1,2,…） |
| `demand` | `Float64` | 历史需求量（训练数据） |

- 同一 SKU 的行无需连续，但会按首次出现顺序处理
- 每个 SKU 至少需要 **2 行**数据，否则返回 ValidationError 行

#### 输出 Arrow Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `sku_id` | `Utf8` | SKU 标识符 |
| `step_index` | `Int32?` | 预测步序号 0,1,2,…（成功行有值；错误行为 null） |
| `prediction` | `Float64?` | 预测值（成功行有值；错误行为 null） |
| `error_code` | `Utf8?` | 错误类型：`"ValidationError"` 或 `"ModelError"`（错误行有值；成功行为 null） |
| `error_message` | `Utf8?` | 错误描述（错误行有值；成功行为 null） |

#### 完整调用示例

```typescript
import * as arrow from "apache-arrow";
import init, { predict_inventory_demand_batch } from "./pkg/fast_insight_engine.js";

await init();

// 构建输入：3 个 SKU，每个 3 个历史数据点
const skuIds    = ["SKU-A", "SKU-A", "SKU-A", "SKU-B", "SKU-B", "SKU-B", "SKU-C"];
const timeIdx   = [0, 1, 2, 0, 1, 2, 0];
const demands   = [100, 110, 120, 50, 55, 60, 30]; // SKU-C 只有 1 行 → 将产生错误行

const inputTable = new arrow.Table({
  sku_id:     arrow.vectorFromArray(skuIds,  new arrow.Utf8()),
  time_index: arrow.vectorFromArray(timeIdx, new arrow.Float64()),
  demand:     arrow.vectorFromArray(demands, new arrow.Float64()),
});

const writer = new arrow.RecordBatchStreamWriter();
writer.writeAll(inputTable);
const inputBytes = writer.toUint8Array();

// 调用：预测 2 步，线性模式，不缩放
const resultBytes = await predict_inventory_demand_batch(
  inputBytes,
  2,          // predict_steps
  "linear",   // mode
  "none",     // scaling
  0.0         // threshold (保留参数)
);

// 解析输出
const resultTable = arrow.tableFromIPC(resultBytes);

// 筛选成功行（有预测值）
const successRows = resultTable
  .filter(row => row.error_code === null)
  .select(["sku_id", "step_index", "prediction"]);

// 筛选错误行（预测失败的 SKU）
const errorRows = resultTable
  .filter(row => row.error_code !== null)
  .select(["sku_id", "error_code", "error_message"]);

console.log("预测结果：", successRows.toArray());
// 输出示例：
// [
//   { sku_id: "SKU-A", step_index: 0, prediction: 130.0 },
//   { sku_id: "SKU-A", step_index: 1, prediction: 140.0 },
//   { sku_id: "SKU-B", step_index: 0, prediction: 65.0  },
//   { sku_id: "SKU-B", step_index: 1, prediction: 70.0  },
// ]

console.log("错误 SKU：", errorRows.toArray());
// 输出示例：
// [
//   { sku_id: "SKU-C", error_code: "ValidationError", error_message: "..." }
// ]
```

#### DuckDB-Wasm 查询示例

```sql
-- 只看成功预测的 SKU
SELECT sku_id, step_index, prediction
FROM batch_result
WHERE error_code IS NULL
ORDER BY sku_id, step_index;

-- 只看失败的 SKU
SELECT sku_id, error_code, error_message
FROM batch_result
WHERE error_code IS NOT NULL;
```

#### 注意事项

- `predict_steps` 必须 ≥ 1，否则函数会抛出 `JsError`
- SKU 数量没有限制，但大批量时建议测试浏览器内存
- 每个 SKU 使用相同的 `mode` 和 `scaling` 参数；如需不同参数，需分批调用
- `threshold` 参数当前未使用，保留 `0.0` 即可

### 6. 获取版本信息

```typescript
const version = fastInsight.get_wasm_version();
console.log(`Wasm 模块版本: ${version}`);
```

## 实际使用示例

### 完整的异常检测流程

```typescript
import init, * as fastInsight from './pkg/fast_insight_engine.js';

async function detectAnomalies() {
  // 初始化 Wasm 模块
  await init();

  // 假设从 DuckDB Wasm 获取 Arrow IPC 数据
  const arrowData = getArrowDataFromDuckDB();
  
  try {
    // 调用异常检测
    const result = await fastInsight.detect_order_anomalies(
      arrowData,
      0.7,   // 异常阈值
      0,     // DuckDB 已预处理，不需要再 scaling
      false  // 不使用 GPU
    );
    
    // 解析结果
    parseArrowResults(result);
  } catch (error) {
    console.error('异常检测失败:', error);
  }
}

// 调用
detectAnomalies();
```

### 在 Web Worker 中运行（推荐用于大数据）

```typescript
// worker.ts
importScripts('./pkg/fast_insight_engine.js');

self.onmessage = async (event) => {
  const { data, type, params } = event.data;
  
  try {
    let result;
    switch (type) {
      case 'anomalies':
        result = await self.detect_order_anomalies(...params);
        break;
      case 'clustering':
        result = await self.segment_customer_orders(...params);
        break;
      default:
        throw new Error(`Unknown type: ${type}`);
    }
    
    self.postMessage({ success: true, result });
  } catch (error) {
    self.postMessage({ success: false, error: error.message });
  }
};
```

## Arrow IPC 数据格式

所有函数接收和返回 **Arrow IPC Stream 格式**的二进制数据。

### 输入 Schema 示例（异常检测）

```
order_id: Int64 (主键)
price: Float64
quantity: Int32
customer_id: Int64
day_of_week: Int32
hour: Int32
... 其他特征列
```

### 输出 Schema 示例

```
order_id: Int64
abnormal_score: Float64 [0, 1]
is_abnormal: Boolean
```

## 性能指标

| 算法 | 数据量 | CPU | GPU |
|------|--------|-----|-----|
| 异常检测 (Isolation Forest) | 100k 行 | 1-2s | - |
| K-Means 聚类 | 100k 行 | ~20s | <0.5s (40x) |
| K-Means 聚类 | 1M 行 | ~5min | <5s (60x) |
| Linear Regression | 10k 点 | <100ms | - |

## 二进制大小

- **Wasm 二进制**: 1.7 MB (包含 ML 算法库)
- **JS 包装**: 52 KB
- **TypeScript 定义**: 6.6 KB

## 常见问题

### Q: 如何处理大于 1GB 的数据？

A: Wasm 运行在浏览器内存限制内（通常 2GB）。建议分批处理：
- 每批 100k-500k 行
- 使用 Web Worker 避免阻塞 UI
- 或在服务器端运行

### Q: 支持 GPU 加速吗？

A: K-Means 支持 WebGPU（Chrome 113+）。其他算法目前仅支持 CPU。

### Q: 数据预处理（Scaling）需要吗？

A: 可选。Wasm 函数支持 3 种 scaling 模式：
- `0`: 无（假设数据已预处理）
- `1`: MinMax (0-1 范围)
- `2`: Standard (mean=0, std=1，推荐)

### Q: 如何调试 Wasm 中的错误？

A: 所有函数返回 `Promise<Uint8Array>` 或抛出 `JsError`。使用 try-catch 捕获错误：

```typescript
try {
  const result = await fastInsight.detect_order_anomalies(...);
} catch (error: any) {
  console.error('Wasm 错误:', error.message);
}
```

## 构建 Wasm 包

```bash
# 重新构建 Wasm（发布版本）
wasm-pack build --target web --out-dir pkg --release

# 开发版本（带调试信息）
wasm-pack build --target web --out-dir pkg --dev
```

## 与 DuckDB Wasm 集成

```typescript
import * as duckdb from '@duckdb/wasm';
import * as fastInsight from './pkg/fast_insight_engine.js';

// 使用 DuckDB 读取 CSV，然后使用 FastInsight 分析
async function analyzeCsvFile(csvFile: File) {
  // 1. 用 DuckDB 加载 CSV
  const db = new duckdb.AsyncDatabase();
  const query = `SELECT * FROM read_csv_auto('${csvFile.name}')`;
  const arrowTable = await db.query(query);
  const arrowBytes = arrowTable.toIPCStream();
  
  // 2. 用 FastInsight 进行异常检测
  const result = await fastInsight.detect_order_anomalies(
    new Uint8Array(arrowBytes),
    0.7,
    0,  // DuckDB 已处理数据类型，不需要再 scaling
    false
  );
  
  // 3. 解析结果
  handleResults(result);
}
```

---

**最后更新**: 2026-05-01  
**Wasm 模块版本**: 0.1.0  
**二进制大小**: 1.7 MB
