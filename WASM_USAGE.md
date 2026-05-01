# Wasm TypeScript 集成示例

本文档展示如何在 TypeScript/JavaScript 中使用生成的 Wasm 模块。

## 模块导入

```typescript
// 导入 Wasm 模块和类型定义
import * as fastInsight from '@hummer-team/fast-insight-engine';
```

## 四个导出函数

### 1. 异常订单检测 (Isolation Forest)

```typescript
// 假设 data 是 Arrow IPC Stream 格式的 Uint8Array
// 其中包含特征列（price, quantity, customer_id 等）
const data: Uint8Array = /* Arrow IPC Stream bytes */;

// 使用参数:
// - threshold: [0, 1] 的异常分数阈值（分数 >= 阈值 = 异常）
// - scaling_mode: 0=无, 1=MinMax, 2=Standard
// - use_gpu: false (Extended Isolation Forest 不支持 GPU)

const result = await fastInsight.detect_order_anomalies(
  data,
  0.7,        // 异常分数阈值
  2,          // 使用 Standard scaling
  false       // 不使用 GPU
);

// result 是 Arrow IPC Stream 格式的 Uint8Array，包含:
// - order_id (Int64)
// - abnormal_score (Float64) [0, 1]
// - is_abnormal (Boolean)
```

### 2. 订单分组聚类 (K-Means)

```typescript
// K-Means 可以使用 WebGPU 加速（Chrome 113+）
const useGPU = rowCount > 5000 && navigator.gpu !== undefined;

const result = await fastInsight.segment_customer_orders(
  data,
  5,          // 聚类数（k=5）
  2,          // 使用 Standard scaling（推荐用于 K-Means）
  useGPU      // 如果可用则启用 GPU，自动降级到 CPU
);

// result 是 Arrow IPC Stream，包含:
// - order_id (Int64)
// - cluster_id (Int32) [0, k-1]
```

### 3. 库存需求预测 (Linear Regression)

```typescript
// 输入数据应该包含：
// - 时间/索引列 (x)
// - 历史需求数据 (y)

const result = await fastInsight.predict_inventory_demand(
  data,
  12,         // 预测 12 个时间步
  2           // 使用 Standard scaling
);

// result 是 Arrow IPC Stream，包含:
// - time_step (Int64)
// - predicted_demand (Float64)
```

### 4. 获取版本信息

```typescript
const version = fastInsight.get_wasm_version();
console.log(`Wasm 模块版本: ${version}`);
// 输出: "0.1.0" 或类似
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
