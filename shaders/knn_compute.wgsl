// WebGPU Compute Shader for K-Nearest Neighbors Anomaly Detection

struct Params {
    n_samples: u32,
    n_features: u32,
    k: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const MAX_K: u32 = 10u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= params.n_samples) {
        return;
    }
    
    var min_dists: array<f32, MAX_K>;
    for (var i = 0u; i < MAX_K; i++) {
        min_dists[i] = 1e10;
    }
    
    for (var j = 0u; j < params.n_samples; j++) {
        if (j == idx) {
            continue;
        }
        
        var dist_sq: f32 = 0.0;
        for (var d = 0u; d < params.n_features; d++) {
            let a = features[idx * params.n_features + d];
            let b = features[j * params.n_features + d];
            let diff = a - b;
            dist_sq += diff * diff;
        }
        let dist = sqrt(dist_sq);
        
        if (dist < min_dists[params.k - 1u]) {
            for (var k_idx = 0u; k_idx < params.k; k_idx++) {
                if (dist < min_dists[k_idx]) {
                    for (var shift = params.k - 1u; shift > k_idx; shift--) {
                        min_dists[shift] = min_dists[shift - 1u];
                    }
                    min_dists[k_idx] = dist;
                    break;
                }
            }
        }
    }
    
    var sum: f32 = 0.0;
    for (var i = 0u; i < params.k; i++) {
        sum += min_dists[i];
    }
    scores[idx] = sum / f32(params.k);
}
