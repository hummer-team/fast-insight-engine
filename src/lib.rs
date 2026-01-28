use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_isofold::{IsolationForest, IsolationForestParams};
use linfa_linear::LinearRegression;

#[wasm_bindgen]
pub fn detect_order_anomalies(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // 将一维数组转为矩阵：每行是一个订单，每列是一个特征（如：实付、数量、退货历史）
    let dataset = Dataset::from(Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap());

    // 训练模型
    let model = IsolationForestParams::new(100)
        .fit(&dataset)
        .expect("模型训练失败");

    // 返回每个订单的“离群得分”，分数越低越可疑
    model.predict(&dataset).to_vec()
}

#[wasm_bindgen]
pub fn segment_customer_orders(
    data: &[f64],
    rows: usize,
    cols: usize,
    clusters: usize,
) -> Vec<usize> {
    let dataset = Dataset::from(Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap());

    let model = KMeans::params(clusters)
        .max_n_iterations(100)
        .fit(&dataset)
        .expect("聚类失败");

    // 返回每个订单所属的簇 ID（0, 1, 2...）
    model.predict(&dataset).targets().to_vec()
}

#[wasm_bindgen]
pub fn predict_inventory_demand(x_data: &[f64], y_data: &[f64], rows: usize) -> Vec<f64> {
    let x = Array2::from_shape_vec((rows, 1), x_data.to_vec()).unwrap();
    let y = Array1::from_vec(y_data.to_vec());
    let dataset = Dataset::new(x, y);

    let model = LinearRegression::default().fit(&dataset).unwrap();

    // 预测未来 7 个时间步
    let future =
        Array2::from_shape_vec((7, 1), (1..=7).map(|i| (rows + i) as f64).collect()).unwrap();
    model.predict(&future).to_vec()
}
