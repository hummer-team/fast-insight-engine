// WebGPU Compute Shader for K-Means Clustering
// 
// This shader performs parallel distance computation and cluster assignment
// for K-Means clustering algorithm.
//
// Algorithm:
// 1. Each thread computes distances from one sample to all centroids
// 2. Finds the nearest centroid and assigns cluster ID
//
// Performance: O(n*k*d) distance computations parallelized across n threads
// Centroid updates are done on CPU to avoid complex atomic operations

@group(0) @binding(0)
var<storage, read> data: array<f32>;  // Flattened data matrix [n_samples * n_features]

@group(0) @binding(1)
var<storage, read> centroids: array<f32>;  // Flattened centroids matrix [k_clusters * n_features]

@group(0) @binding(2)
var<storage, read_write> assignments: array<u32>;  // Cluster assignments [n_samples]

@group(0) @binding(3)
var<storage, read> params: array<u32>;  // [n_samples, n_features, k_clusters]

@workgroup_size(256)
@compute
fn kmeans_assign(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    let n_samples = params[0];
    let n_features = params[1];
    let k_clusters = params[2];
    
    // Bounds check
    if (sample_idx >= n_samples) {
        return;
    }
    
    // Find nearest centroid
    var min_distance = 1e38;
    var nearest_cluster = 0u;
    
    for (var k = 0u; k < k_clusters; k++) {
        var distance = 0.0;
        
        // Compute squared Euclidean distance
        for (var d = 0u; d < n_features; d++) {
            let data_val = data[sample_idx * n_features + d];
            let centroid_val = centroids[k * n_features + d];
            let diff = data_val - centroid_val;
            distance += diff * diff;
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            nearest_cluster = k;
        }
    }
    
    // Assign to nearest cluster
    assignments[sample_idx] = nearest_cluster;
}
