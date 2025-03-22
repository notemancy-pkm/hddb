use std::ops::Mul;
use tch::{Device, Kind, Tensor};

/// Generate `n` random bipolar hypervectors of dimension `dim`.
pub fn gen(n: i64, dim: i64) -> Tensor {
    // Create a tensor with random values in [0, 1) and convert to bipolar {-1, +1}.
    let t = Tensor::empty(&[n, dim], (Kind::Float, Device::Cpu)).bernoulli_p(0.5);
    2.0 * t - 1.0
}

/// Bind two sets of hypervectors using element-wise multiplication.
/// If `v2` is provided, it checks if the shapes are broadcastable and returns v1 * v2.
/// If `v2` is `None`, it multiplies all rows of `v1` (i.e. reduces along dimension 0).
pub fn bind(v1: &Tensor, v2: Option<&Tensor>) -> Result<Tensor, String> {
    match v2 {
        Some(v) => {
            let shape1 = v1.size();
            let shape2 = v.size();
            if shape1.len() != shape2.len() {
                return Err("Shapes not compatible for binding".to_string());
            }
            for (s1, s2) in shape1.iter().zip(shape2.iter()) {
                if *s1 != *s2 && *s1 != 1 && *s2 != 1 {
                    return Err("Shapes not broadcastable for binding".to_string());
                }
            }
            Ok(v1 * v)
        }
        None => {
            // Multiply along dimension 0 (reducing all rows).
            Ok(v1.prod_dim_int(0, true, Kind::Float))
        }
    }
}

/// Bundle two sets of hypervectors using element-wise addition.
/// If `v2` is provided, it checks for broadcastable shapes and returns v1 + v2.
/// If `v2` is `None`, it sums all rows of `v1` (reducing along dimension 0).
pub fn bundle(v1: &Tensor, v2: Option<&Tensor>) -> Result<Tensor, String> {
    match v2 {
        Some(v) => {
            let shape1 = v1.size();
            let shape2 = v.size();
            if shape1.len() != shape2.len() {
                return Err("Shapes not compatible for bundling".to_string());
            }
            for (s1, s2) in shape1.iter().zip(shape2.iter()) {
                if *s1 != *s2 && *s1 != 1 && *s2 != 1 {
                    return Err("Shapes not broadcastable for bundling".to_string());
                }
            }
            Ok(v1 + v)
        }
        None => {
            // Sum along dimension 0 (reducing all rows).
            Ok(v1.sum_dim_intlist(0, true, Kind::Float))
        }
    }
}

/// Normalize each vector (row) in `v1` to have L2 norm equal to `mag`.
pub fn normalize(v1: &Tensor, mag: f64) -> Tensor {
    // Compute L2 norm along dimension 1: sqrt(sum(x^2)).
    let pow = Tensor::from_slice(&[2]);
    let norm = v1.pow(&pow).sum_dim_intlist(1, true, Kind::Float).sqrt();
    v1 * (mag / norm)
}

/// Compute pairwise cosine similarity between the two sets of vectors.
/// v1 and v2 can each be either 1×dim or n×dim. If both have multiple rows, they must have the same n;
/// otherwise, the single-row tensor is expanded to match.
/// This function computes the cosine similarity along the last dimension (assumed to be 1).
pub fn cosine_sim(v1: &Tensor, v2: &Tensor) -> Tensor {
    let n1 = v1.size()[0];
    let n2 = v2.size()[0];
    if n1 != n2 && n1 != 1 && n2 != 1 {
        panic!("v1 and v2 must have the same number of rows or one must be a single row.");
    }
    let (v1_exp, v2_exp) = if n1 == n2 {
        (v1.shallow_clone(), v2.shallow_clone())
    } else if n1 == 1 {
        (v1.expand(&[n2, v1.size()[1]], true), v2.shallow_clone())
    } else {
        (v1.shallow_clone(), v2.expand(&[n1, v2.size()[1]], true))
    };
    // Compute cosine similarity along dimension 1 with a small epsilon for numerical stability.
    Tensor::cosine_similarity(&v1_exp, &v2_exp, 1, 1e-8)
}

/// Compute pairwise dot similarity between the two sets of vectors.
/// v1 and v2 can each be either 1×dim or n×dim. If both have multiple rows, they must have the same n;
/// otherwise, the single-row tensor is expanded to match.
/// This function computes the dot product for corresponding vectors (i.e. pairwise dot product).
pub fn dot_sim(v1: &Tensor, v2: &Tensor) -> Tensor {
    let n1 = v1.size()[0];
    let n2 = v2.size()[0];
    if n1 != n2 && n1 != 1 && n2 != 1 {
        panic!("v1 and v2 must have the same number of rows or one must be a single row.");
    }
    let (v1_exp, v2_exp) = if n1 == n2 {
        (v1.shallow_clone(), v2.shallow_clone())
    } else if n1 == 1 {
        (v1.expand(&[n2, v1.size()[1]], true), v2.shallow_clone())
    } else {
        (v1.shallow_clone(), v2.expand(&[n1, v2.size()[1]], true))
    };
    // Compute tensordot over the vector dimension (dimension 1 of each), resulting in an (n x n) matrix.
    let dot_matrix = v1_exp.tensordot(&v2_exp, &[1], &[1]);
    // Extract the diagonal to obtain the pairwise dot products.
    dot_matrix.diagonal(0, 0, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Kind, Tensor};

    /// Helper function to compute cosine similarity between two 1-D tensors.
    fn cosine_similarity(a: &Tensor, b: &Tensor) -> f64 {
        let dot = a.dot(b).double_value(&[]);
        let pow = Tensor::from_slice(&[2]);
        let norm_a = a.pow(&pow).sum(Kind::Float).sqrt().double_value(&[]);
        let norm_b = b.pow(&pow).sum(Kind::Float).sqrt().double_value(&[]);
        dot / (norm_a * norm_b)
    }

    #[test]
    fn test_gen_orthogonality() {
        let v = gen(2, 10000);
        let v0 = v.get(0);
        let v1 = v.get(1);
        let cosine = cosine_similarity(&v0, &v1);
        assert!(
            cosine.abs() < 0.1,
            "Cosine similarity is too high: {}",
            cosine
        );
    }

    #[test]
    fn test_bind_orthogonality() {
        let v1 = gen(1, 10000);
        let v2 = gen(1, 10000);
        let bound = bind(&v1, Some(&v2)).unwrap();
        let b = bound.get(0);
        let v1_flat = v1.get(0);
        let v2_flat = v2.get(0);
        let cosine1 = cosine_similarity(&b, &v1_flat);
        let cosine2 = cosine_similarity(&b, &v2_flat);
        assert!(
            cosine1.abs() < 0.1,
            "Cosine similarity with v1 is too high: {}",
            cosine1
        );
        assert!(
            cosine2.abs() < 0.1,
            "Cosine similarity with v2 is too high: {}",
            cosine2
        );
    }

    #[test]
    fn test_bind_reduce() {
        let v = gen(5, 10000);
        let bound = bind(&v, None).unwrap();
        for i in 0..5 {
            let vi = v.get(i);
            let cosine = cosine_similarity(&bound.get(0), &vi);
            assert!(
                cosine.abs() < 0.1,
                "Cosine similarity for row {} is too high: {}",
                i,
                cosine
            );
        }
    }

    #[test]
    fn test_bundle_similarity() {
        let v = gen(1, 10000);
        let bundled = bundle(&v, Some(&v)).unwrap();
        let cosine = cosine_similarity(&v.get(0), &bundled.get(0));
        // Since bundling identical hypervectors doubles the value, cosine similarity should be 1.
        assert!(
            (cosine - 1.0).abs() < 1e-6,
            "Cosine similarity is not 1: {}",
            cosine
        );
    }

    #[test]
    fn test_bundle_reduce() {
        let v = gen(1, 10000);
        // Stack the same vector twice.
        let v_stack = Tensor::cat(&[&v, &v], 0);
        let bundled = bundle(&v_stack, None).unwrap();
        let cosine = cosine_similarity(&v.get(0), &bundled.get(0));
        assert!(
            (cosine - 1.0).abs() < 1e-6,
            "Cosine similarity is not 1: {}",
            cosine
        );
    }

    #[test]
    fn test_normalize() {
        let v = gen(5, 10000);
        let mag = 10.0;
        let normalized = normalize(&v, mag);
        let pow = Tensor::from_slice(&[2]);
        for i in 0..5 {
            // Compute the norm of each row manually.
            let norm_val = normalized
                .get(i)
                .pow(&pow)
                .sum(Kind::Float)
                .sqrt()
                .double_value(&[]);
            assert!(
                (norm_val - mag).abs() < 1e-5,
                "Row {} norm not equal to {}: {}",
                i,
                mag,
                norm_val
            );
        }
    }

    #[test]
    fn test_cosine_sim_pairwise() {
        let v1 = gen(3, 100);
        let v2 = gen(3, 100);
        let cs = cosine_sim(&v1, &v2);
        assert_eq!(cs.size(), [3]);
        for i in 0..3 {
            let manual = cosine_similarity(&v1.get(i), &v2.get(i));
            let cs_val = cs.get(i).double_value(&[]);
            assert!(
                (manual - cs_val).abs() < 1e-6,
                "Row {}: expected {}, got {}",
                i,
                manual,
                cs_val
            );
        }
    }

    #[test]
    fn test_cosine_sim_broadcast() {
        let v1 = gen(1, 100);
        let v2 = gen(3, 100);
        let cs = cosine_sim(&v1, &v2);
        assert_eq!(cs.size(), [3]);
        for i in 0..3 {
            let manual = cosine_similarity(&v1.get(0), &v2.get(i));
            let cs_val = cs.get(i).double_value(&[]);
            assert!(
                (manual - cs_val).abs() < 1e-6,
                "Row {}: expected {}, got {}",
                i,
                manual,
                cs_val
            );
        }
    }

    #[test]
    fn test_dot_sim_pairwise() {
        let v1 = gen(3, 100);
        let v2 = gen(3, 100);
        let ds = dot_sim(&v1, &v2);
        assert_eq!(ds.size(), [3]);
        for i in 0..3 {
            let manual = v1.get(i).mul(&v2.get(i)).sum(Kind::Float).double_value(&[]);
            let ds_val = ds.get(i).double_value(&[]);
            assert!(
                (manual - ds_val).abs() < 1e-6,
                "Row {}: expected {}, got {}",
                i,
                manual,
                ds_val
            );
        }
    }

    #[test]
    fn test_dot_sim_broadcast() {
        let v1 = gen(1, 100);
        let v2 = gen(3, 100);
        let ds = dot_sim(&v1, &v2);
        assert_eq!(ds.size(), [3]);
        for i in 0..3 {
            let manual = v1.get(0).mul(&v2.get(i)).sum(Kind::Float).double_value(&[]);
            let ds_val = ds.get(i).double_value(&[]);
            assert!(
                (manual - ds_val).abs() < 1e-6,
                "Row {}: expected {}, got {}",
                i,
                manual,
                ds_val
            );
        }
    }

    #[test]
    fn test_cosine_sim_single() {
        let v1 = gen(1, 100);
        let v2 = gen(1, 100);
        let cs = cosine_sim(&v1, &v2);
        assert_eq!(cs.size(), [1]);
        let manual = cosine_similarity(&v1.get(0), &v2.get(0));
        let cs_val = cs.get(0).double_value(&[]);
        assert!(
            (manual - cs_val).abs() < 1e-6,
            "Expected {}, got {}",
            manual,
            cs_val
        );
    }

    #[test]
    fn test_dot_sim_single() {
        let v1 = gen(1, 100);
        let v2 = gen(1, 100);
        let ds = dot_sim(&v1, &v2);
        assert_eq!(ds.size(), [1]);
        let manual = v1.get(0).mul(&v2.get(0)).sum(Kind::Float).double_value(&[]);
        let ds_val = ds.get(0).double_value(&[]);
        assert!(
            (manual - ds_val).abs() < 1e-6,
            "Expected {}, got {}",
            manual,
            ds_val
        );
    }
}
