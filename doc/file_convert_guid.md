# CSV/Excel → Parquet 转换 API 使用指南

本文档详细说明如何在 TypeScript/JavaScript 中调用 Wasm 导出的文件转换函数。

---

## 快速开始

### 环境准备

```typescript
// 导入 Wasm 模块
import * as wasm from 'fast-insight-engine';

// 等待 Wasm 初始化
await wasm.default();  // 或 await wasm.initSync(buffer)
```

---

## API 参考

### 1. `convert_csv_to_parquet()`

将 CSV 数据转换为 Parquet 格式。支持两种模式：**宽松模式**（DuckDB 自动类型推断）和**严格模式**（Rust 强制类型转换）。

#### 函数签名

```typescript
async function convert_csv_to_parquet(
  csvData: Uint8Array,           // CSV 文件字节
  delimiter: number,              // 分隔符 ASCII 码：44=',', 9='\t', 124='|', 59=';'
  hasHeader: boolean,             // 第一行是否为列名
  rowGroupSize: number,           // 行组大小 (64-16384, 推荐 1024)
  schemaHintJson?: string         // 可选 JSON 格式的类型元数据
): Promise<Uint8Array>
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `csvData` | `Uint8Array` | ✓ | CSV 原始字节。可通过 `new TextEncoder().encode(csvString)` 或 `File.arrayBuffer()` 获取 |
| `delimiter` | `number` | ✓ | CSV 分隔符的 ASCII 码。常用值：`44` (`,`), `9` (`\t`), `124` (`\|`), `59` (`;`) |
| `hasHeader` | `boolean` | ✓ | 若 `true`，第一行作为列名；若 `false`，自动生成 `col_0`, `col_1`, ... |
| `rowGroupSize` | `number` | ✓ | Parquet 行组大小（每个行组的行数）。范围 64-16384；建议 1024（平衡内存和压缩）|
| `schemaHintJson` | `string \| undefined` | ✗ | JSON 格式的列类型定义（见下方「严格模式」章节）。不提供时采用宽松模式 |

#### 返回值

- **成功**：`Uint8Array`，完整的 Parquet 文件字节，可直接传给 DuckDB Wasm 或保存为 `.parquet` 文件
- **失败**：抛出 `Error`，包含详细的失败原因（例如 `"TypeConversionFailed: Cannot convert 'abc' to Int64 for column 'id'"`）

#### 宽松模式（默认）

所有列保持为 Utf8（字符串）格式，由 DuckDB Wasm 在加载时自动推断实际类型。

**优点**：无需预先指定类型，容错性强  
**缺点**：每次查询时都需要类型推断，性能略低

**示例**：

```typescript
// 简单的 CSV 转换，所有列都作为字符串
const csvText = "order_id,price,customer\n1001,9.99,Alice\n1002,14.50,Bob\n";
const csvBytes = new TextEncoder().encode(csvText);

const parquetBytes = await convert_csv_to_parquet(
  csvBytes,
  44,          // ','
  true,        // has header
  1024,        // row group size
  undefined    // ← 不传 schemaHintJson，采用宽松模式
);

// parquetBytes 可传给 DuckDB
await db.open(Module.Database);
const tbl = await db.insertArrowTable(parquetBytes, { create: true });
// 此时 DuckDB 会推断 order_id 为 Int64, price 为 Float64
```

#### 严格模式

提供 `schemaHintJson`，Rust 会按指定类型转换各列。若某列数据无法转换，立即返回错误。

**优点**：
- 类型确定，查询性能更好
- 及早发现数据质量问题
- 支持 Boolean、Int64、Float64 等多种类型

**缺点**：
- 需提前了解数据模式
- 数据不符合预期时会失败

**支持的类型**：

| type_id | Arrow 类型 | 说明 | 示例值 |
|---------|-----------|------|--------|
| 0 | Utf8 | 字符串 | `"hello"`, `"123"` |
| 1 | Int64 | 64 位整数 | `123`, `-456` |
| 2 | Float64 | 64 位浮点数 | `3.14`, `-0.5`, `1e10` |
| 3 | Boolean | 布尔值 | `"true"`, `"false"`, `"1"`, `"0"`, `"yes"`, `"no"` |
| 其他 | (默认 Utf8) | 不支持的 type_id 自动降级为 Utf8 | - |

**schemaHintJson 格式**：

```json
{
  "columns": [
    {"name": "order_id", "type_id": 1},
    {"name": "price", "type_id": 2},
    {"name": "customer", "type_id": 0},
    {"name": "is_valid", "type_id": 3}
  ]
}
```

**示例**：

```typescript
const csvText = "order_id,price,customer,is_valid\n1001,9.99,Alice,true\n1002,14.50,Bob,false\n";
const csvBytes = new TextEncoder().encode(csvText);

const schemaHint = {
  columns: [
    { name: "order_id", type_id: 1 },   // Int64
    { name: "price", type_id: 2 },      // Float64
    { name: "customer", type_id: 0 },   // Utf8
    { name: "is_valid", type_id: 3 }    // Boolean
  ]
};

const parquetBytes = await convert_csv_to_parquet(
  csvBytes,
  44,
  true,
  1024,
  JSON.stringify(schemaHint)  // ← 严格模式：传入 JSON
);

// 若 CSV 某行的 order_id 是 "abc"（非整数），会立即抛出错误：
// Error: TypeConversionFailed: Cannot convert 'abc' to Int64 for column 'order_id'
```

#### 常见错误及解决

| 错误信息 | 原因 | 解决方案 |
|---------|------|--------|
| `CSV→Parquet error: InvalidState: Call begin_csv_to_parquet() first` | 内部状态不一致 | 不应发生；若出现请报告 bug |
| `TypeError: 'schemaHintJson' is not valid JSON` | schemaHintJson 格式错误 | 检查 JSON 有效性：`JSON.parse(schemaHintJson)` |
| `TypeConversionFailed: Cannot convert 'abc' to Int64` | 严格模式下数据类型不匹配 | 检查 CSV 数据或调整 type_id |
| `ParquetRowGroupTooLarge: size 20000 exceeds 16384` | 行组大小超限 | 使用 `rowGroupSize <= 16384` |

---

### 2. `convert_excel_to_parquet()`

将 Excel 文件转换为 Parquet 格式。支持 .xlsx 和 .xls 格式。

#### 函数签名

```typescript
async function convert_excel_to_parquet(
  excelData: Uint8Array,          // Excel 文件字节
  sheetNameOrIndex: string,       // 工作表名称或 "" (空=首个工作表)
  hasHeader: boolean,             // 第一行是否为列名
  rowGroupSize: number,           // 行组大小 (64-16384, 推荐 1024)
  maxStringTableBytes: number,    // 最大字符串表大小，字节 (0=100MB 默认)
  schemaHintJson?: string         // 可选 JSON 格式的类型元数据
): Promise<Uint8Array>
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `excelData` | `Uint8Array` | ✓ | Excel 原始字节。通过 `File.arrayBuffer()` 或 `await file.arrayBuffer()` 获取 |
| `sheetNameOrIndex` | `string` | ✓ | 工作表选择器。`""` 表示第一个工作表；或传入工作表名称（如 `"Sheet1"`）|
| `hasHeader` | `boolean` | ✓ | 若 `true`，第一行作为列名；若 `false`，自动生成 `col_0`, `col_1`, ... |
| `rowGroupSize` | `number` | ✓ | Parquet 行组大小。范围 64-16384；建议 1024 |
| `maxStringTableBytes` | `number` | ✓ | 最大字符串表大小（字节）。`0` 表示使用默认值 100 MB；单位字节，例如 `50_000_000` (50MB) |
| `schemaHintJson` | `string \| undefined` | ✗ | JSON 格式的列类型定义（同 CSV）。不提供时采用宽松模式 |

#### 返回值

同 `convert_csv_to_parquet()`：成功返回 Parquet 字节，失败抛出 Error。

#### 使用示例

**宽松模式**（推荐，无需预知数据类型）：

```typescript
// 用户选择 Excel 文件
const fileInput = document.querySelector('input[type="file"]');
const file = fileInput.files[0];
const excelBytes = new Uint8Array(await file.arrayBuffer());

// 转换为 Parquet（所有列作为字符串）
const parquetBytes = await convert_excel_to_parquet(
  excelBytes,
  "",              // 使用第一个工作表
  true,            // 第一行是列名
  1024,            // 行组大小
  0,               // 使用默认 100MB 字符串表限制
  undefined        // 宽松模式
);

// 保存或加载到 DuckDB
const link = document.createElement('a');
link.href = URL.createObjectURL(new Blob([parquetBytes]));
link.download = 'output.parquet';
link.click();
```

**严格模式**（已知数据类型）：

```typescript
const excelBytes = new Uint8Array(await file.arrayBuffer());

const schemaHint = JSON.stringify({
  columns: [
    { name: "Employee ID", type_id: 1 },    // Int64
    { name: "Salary", type_id: 2 },         // Float64
    { name: "Name", type_id: 0 },           // Utf8
    { name: "Active", type_id: 3 }          // Boolean
  ]
});

const parquetBytes = await convert_excel_to_parquet(
  excelBytes,
  "HR Data",       // 指定工作表名称
  true,
  2048,            // 更大的行组（若数据量大）
  50_000_000,      // 限制字符串表为 50MB
  schemaHint       // 严格模式
);
```

#### 常见错误及解决

| 错误信息 | 原因 | 解决方案 |
|----------|------|--------|
| `ExcelFileTooLarge: File is 1500 MB, limit is 1000 MB` | 文件超过 1GB | 分割 Excel 文件或使用服务端转换 |
| `ExcelEncrypted: File is encrypted or write-protected` | Excel 设置了密码保护 | 在 Excel 中移除保护后重试 |
| `ExcelLoadFailed: Unrecognized Excel format` | 非 .xlsx/.xls 文件 | 确保上传的是真实 Excel 文件 |
| `ExcelLoadFailed: String table size ... exceeds limit` | 字符串数据过多 | 增大 `maxStringTableBytes` 参数 |

---

## 完整示例：浏览器端 CSV 上传 → Parquet → DuckDB 查询

```typescript
import * as wasm from 'fast-insight-engine';
import initSql, { Database } from '@duckdb/wasm';

// 1. 初始化 Wasm
await wasm.default();
const SQL = await initSql();
const db = new SQL.Database();

// 2. 读取并转换 CSV
async function convertAndQuery(file: File) {
  const csvBytes = new Uint8Array(await file.arrayBuffer());
  
  // 转换：宽松模式
  const parquetBytes = await wasm.convert_csv_to_parquet(
    csvBytes,
    44,    // ','
    true,  // 有表头
    1024,  // 行组大小
    undefined // 宽松模式
  );
  
  // 3. 加载到 DuckDB
  const tbl = await db.insertArrowTable(parquetBytes, {
    create: true,
    name: 'my_data'
  });
  
  // 4. 查询
  const result = await db.query('SELECT * FROM my_data LIMIT 10');
  console.log(result.toArray());
  
  return result;
}

// 5. 绑定 UI
document.getElementById('uploadBtn').addEventListener('change', (e) => {
  const file = e.target.files[0];
  convertAndQuery(file).catch(err => alert('Error: ' + err.message));
});
```

---

## 性能建议

1. **行组大小选择**：
   - 小数据（<10MB）：1024 行
   - 中等数据（10-100MB）：2048-4096 行
   - 大数据（>100MB）：8192 行

2. **内存限制**：
   - 浏览器限制：单个 Wasm 实例 ~150MB
   - 若数据>100MB，分多次转换（逐块喂入）

3. **类型指定**：
   - 若频繁查询某列，使用严格模式预指定类型
   - 布尔、日期字段一定要用严格模式（以防 DuckDB 推断错误）

---

## 常见问题 (FAQ)

**Q: 我应该用宽松模式还是严格模式？**  
A: 
- 默认用宽松模式（无需 schemaHintJson）
- 若数据不确定或希望 DuckDB 自动推断，用宽松模式
- 若数据已验证且需要确定类型，用严格模式
- 对于 Boolean 字段，强烈推荐严格模式

**Q: 支持什么 Excel 格式？**  
A: 支持 .xlsx（OOXML，推荐）和 .xls（OLE2）；不支持加密文件

**Q: 转换后的 Parquet 能直接用吗？**  
A: 是的！生成的 Parquet 文件完整有效，可保存为 .parquet 文件或直接传给 DuckDB Wasm

**Q: 能否部分转换或流式输出？**  
A: 当前 API 是一次性转换，需要完整输入。后续版本可能支持逐块转换

**Q: 性能如何？**  
A: 
- CSV→Parquet：~20-50 MB/s（Chrome，中等硬件）
- Excel→Parquet：~10-30 MB/s（取决于文件复杂度）
- 内存峰值：行组大小 × 平均行宽 × 2，通常 10-50MB

---

## 相关链接

- [技术需求文档](./file_convert.md)
- [DuckDB Wasm 文档](https://duckdb.org/docs/api/wasm.html)
- [Parquet 规范](https://parquet.apache.org/docs/file-format/)
