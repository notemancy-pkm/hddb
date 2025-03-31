// src/vsa.rs

//! Vector Symbolic Architecture (VSA) operations using the Tch crate (v0.17.0).
//!
//! This module focuses on the MAP-I family of VSA, which uses:
//! - Bi-polar vectors (+1, -1).
//! - Element-wise multiplication for binding.
//! - Element-wise addition for bundling.
//! - Cosine similarity or dot product for similarity measures.

use tch::{Device, Kind, TchError, Tensor}; // Added IndexOp for convenience in tests

/// Sets the seed for the Tch random number generator.
///
/// This function should be called to ensure reproducibility of random
/// operations like vector generation within this module.
///
/// # Arguments
///
/// * `seed` - The seed value (i64).
pub fn set_seed(seed: i64) {
    tch::manual_seed(seed);
}

/// Generates random bi-polar vectors.
///
/// Creates a tensor of shape `[num_vectors, dim]` where each element
/// is either +1 or -1, chosen with equal probability (0.5).
/// This follows the formula: 2 * Bernoulli(0.5) - 1.
///
/// # Arguments
///
/// * `num_vectors` - The number of vectors to generate (rows).
/// * `dim` - The dimensionality of each vector (columns).
///
/// # Returns
///
/// A `Result` containing the tensor of bi-polar vectors or a `TchError`.
pub fn gen_vectors(num_vectors: i64, dim: i64) -> Result<Tensor, TchError> {
    if num_vectors <= 0 || dim <= 0 {
        return Err(TchError::Kind(format!(
            "Number of vectors ({}) and dimension ({}) must be positive.",
            num_vectors, dim
        )));
    }

    // Create an empty tensor first for the in-place bernoulli operation
    let mut bernoulli_tensor = Tensor::empty(&[num_vectors, dim], (Kind::Float, Device::Cpu));
    // Apply bernoulli in-place. Note: f_bernoulli_float_ requires &mut self
    // and returns the modified tensor (self).
    // In tch 0.17.0, bernoulli_float_ is the correct method name for in-place operation with float probability.
    let _ = bernoulli_tensor.f_bernoulli_float_(0.5)?; // Use f_bernoulli_float_ for Result return

    // Transform {0, 1} to {-1, 1} using 2*x - 1
    let bipolar_tensor = bernoulli_tensor.f_mul_scalar(2.0)?.f_sub_scalar(1.0)?;
    let vectors = bipolar_tensor.set_requires_grad(false);
    Ok(vectors)
}

/// Binds two tensors of vectors element-wise.
///
/// Performs element-wise multiplication (`*`). The tensors must have
/// broadcastable shapes (e.g., `[N, D]` and `[N, D]`, or `[N, D]` and `[D]`).
///
/// # Arguments
///
/// * `vecs1` - The first tensor of vectors.
/// * `vecs2` - The second tensor of vectors.
///
/// # Returns
///
/// A `Result` containing the resulting bound tensor or a `TchError`.
pub fn bind(vecs1: &Tensor, vecs2: &Tensor) -> Result<Tensor, TchError> {
    // Tch handles broadcasting directly with the '*' operator or f_mul
    vecs1.f_mul(vecs2)
}

/// Bundles two tensors of vectors element-wise.
///
/// Performs element-wise addition (`+`). The tensors must have
/// broadcastable shapes (e.g., `[N, D]` and `[N, D]`, or `[N, D]` and `[D]`).
///
/// # Arguments
///
/// * `vecs1` - The first tensor of vectors.
/// * `vecs2` - The second tensor of vectors.
///
/// # Returns
///
/// A `Result` containing the resulting bundled tensor or a `TchError`.
pub fn bundle(vecs1: &Tensor, vecs2: &Tensor) -> Result<Tensor, TchError> {
    // Tch handles broadcasting directly with the '+' operator or f_add
    vecs1.f_add(vecs2)
}

/// Binds all vectors in a tensor into a single vector.
///
/// Performs element-wise multiplication across all vectors along the 0th dimension.
/// Input shape `[N, D]` results in output shape `[D]`.
///
/// # Arguments
///
/// * `vecs` - A tensor of vectors (e.g., shape `[N, D]`).
///
/// # Returns
///
/// A `Result` containing the single bound vector or a `TchError`.
/// Returns an error if the input tensor is empty.
pub fn bind_all(vecs: &Tensor) -> Result<Tensor, TchError> {
    let size = vecs.size();
    if size.is_empty() || size.first() == Some(&0) {
        // More robust check for first dimension = 0
        return Err(TchError::Kind(format!(
            "Input tensor for bind_all cannot be empty or have zero vectors on axis 0. Shape: {:?}",
            size
        )));
    }
    if size[0] == 1 {
        // If only one vector, return it squeezed. squeeze_dim returns Tensor, wrap in Ok.
        return Ok(vecs.squeeze_dim(0));
    }
    // Perform element-wise product along the 0th dimension (the vector index)
    // keepdim=false to reduce the dimension
    // prod_dim_int returns Tensor, wrap in Ok.
    Ok(vecs.prod_dim_int(0, false, None)) // None uses the default dtype
}

/// Bundles all vectors in a tensor into a single vector.
///
/// Performs element-wise sum across all vectors along the 0th dimension.
/// Input shape `[N, D]` results in output shape `[D]`.
///
/// # Arguments
///
/// * `vecs` - A tensor of vectors (e.g., shape `[N, D]`).
///
/// # Returns
///
/// A `Result` containing the single bundled vector or a `TchError`.
/// Returns an error if the input tensor is empty.
pub fn bundle_all(vecs: &Tensor) -> Result<Tensor, TchError> {
    let size = vecs.size();
    if size.is_empty() || size.first() == Some(&0) {
        // More robust check for first dimension = 0
        return Err(TchError::Kind(format!(
            "Input tensor for bundle_all cannot be empty or have zero vectors on axis 0. Shape: {:?}",
            size
        )));
    }
    if size[0] == 1 {
        // If only one vector, return it squeezed. squeeze_dim returns Tensor, wrap in Ok.
        return Ok(vecs.squeeze_dim(0));
    }
    // Perform element-wise sum along the 0th dimension (the vector index)
    // keepdim=false to reduce the dimension
    // sum_dim_intlist returns Tensor, wrap in Ok. Ensure dim slice is &[i64].
    Ok(vecs.sum_dim_intlist(&[0i64][..], false, None)) // None uses the default dtype
}

/// Computes the pair-wise dot product similarity between vectors in two tensors.
///
/// Calculates `sum(vecs1 * vecs2)` along the feature dimension (dim 1).
/// Handles shapes like `[N, D]` vs `[N, D]` -> `[N]`, or `[D]` vs `[D]` -> `[]` (scalar tensor),
/// or broadcasting cases like `[N, D]` vs `[D]` -> `[N]`.
///
/// # Arguments
///
/// * `vecs1` - The first tensor of vectors.
/// * `vecs2` - The second tensor of vectors.
///
/// # Returns
///
/// A `Result` containing a tensor of dot product scores or a `TchError`.
pub fn dot_sim(vecs1: &Tensor, vecs2: &Tensor) -> Result<Tensor, TchError> {
    let size1 = vecs1.size();
    let size2 = vecs2.size();

    if size1.is_empty() || size2.is_empty() {
        return Err(TchError::Kind(format!(
            "Input tensors for dot_sim cannot be empty. Shapes: {:?} and {:?}",
            size1, size2
        )));
    }

    // Perform element-wise multiplication (handles broadcasting)
    let product = vecs1.f_mul(vecs2)?;

    // Sum along the feature dimension.
    // If original shapes were [N, D] vs [N, D] or [N, D] vs [D], product is [N, D]. Sum along dim 1.
    // If original shapes were [D] vs [D], product is [D]. Sum along dim 0.
    // Check product dim count, not original dims, as broadcasting might change things.
    let feature_dim_index = if product.dim() > 1 {
        product.dim() - 1
    } else {
        0
    }; // Usually the last dim

    // Ensure feature_dim index is valid before trying to get size.
    let feature_dim_size = product.size().get(feature_dim_index).copied().unwrap_or(0);

    if feature_dim_size == 0 {
        // Handle case where the feature dimension is 0 after multiplication (e.g., empty input dimension)
        // Return a tensor of zeros with the appropriate leading dimensions.
        // Tensor::zeros returns Tensor, wrap in Ok.
        let output_shape: Vec<i64> = product
            .size()
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != feature_dim_index)
            .map(|(_, &s)| s)
            .collect();
        Ok(Tensor::zeros(
            &output_shape,
            (product.kind(), product.device()),
        ))
    } else {
        // sum_dim_intlist returns Tensor, wrap in Ok. Ensure dim slice is &[i64].
        Ok(product.sum_dim_intlist(&[feature_dim_index as i64][..], false, None))
    }
}

/// Computes the pair-wise cosine similarity between vectors in two tensors.
///
/// Calculates `dot(vecs1, vecs2) / (norm(vecs1) * norm(vecs2))`.
/// Uses `tch::Tensor::cosine_similarity` for efficient computation.
/// Handles shapes like `[N, D]` vs `[N, D]` -> `[N]`, or `[D]` vs `[D]` -> `[]` (scalar tensor).
/// Also handles broadcasting `[N, D]` vs `[D]` -> `[N]` by expanding the `[D]` vector.
///
/// # Arguments
///
/// * `vecs1` - The first tensor of vectors.
/// * `vecs2` - The second tensor of vectors.
///
/// # Returns
///
/// A `Result` containing a tensor of cosine similarity scores or a `TchError`.
pub fn cos_sim(vecs1: &Tensor, vecs2: &Tensor) -> Result<Tensor, TchError> {
    let size1 = vecs1.size();
    let size2 = vecs2.size();
    let dim1 = size1.len();
    let dim2 = size2.len();

    if dim1 == 0 || dim2 == 0 {
        return Err(TchError::Kind(format!(
            "Input tensors for cos_sim cannot be empty. Shapes: {:?} and {:?}",
            size1, size2
        )));
    }

    let eps = 1e-8; // Epsilon for numerical stability

    // Use associated function syntax: Tensor::f_cosine_similarity
    // Case 1: Both tensors have the same shape (e.g., [N, D] vs [N, D] or [D] vs [D])
    if size1 == size2 {
        if dim1 == 1 {
            // Shape [D] vs [D]
            Tensor::f_cosine_similarity(vecs1, vecs2, 0, eps) // Compare along dimension 0
        } else if dim1 == 2 {
            // Shape [N, D] vs [N, D]
            Tensor::f_cosine_similarity(vecs1, vecs2, 1, eps) // Compare along dimension 1
        } else {
            Err(TchError::Kind(format!(
                "cos_sim not implemented for tensor dims > 2. Shapes: {:?} and {:?}",
                size1, size2
            )))
        }
    }
    // Case 2: Broadcasting [N, D] vs [D]
    else if dim1 == 2 && dim2 == 1 && size1.get(1) == size2.get(0) {
        // Expand vecs2 from [D] to [1, D] then broadcast to [N, D]
        // expand returns Tensor, no `?` needed.
        let vecs2_expanded = vecs2.unsqueeze(0).expand(&size1, true);
        Tensor::f_cosine_similarity(vecs1, &vecs2_expanded, 1, eps) // Compare along dimension 1
    }
    // Case 3: Broadcasting [D] vs [N, D]
    else if dim1 == 1 && dim2 == 2 && size1.get(0) == size2.get(1) {
        // Expand vecs1 from [D] to [1, D] then broadcast to [N, D]
        // expand returns Tensor, no `?` needed.
        let vecs1_expanded = vecs1.unsqueeze(0).expand(&size2, true);
        Tensor::f_cosine_similarity(&vecs1_expanded, vecs2, 1, eps) // Compare along dimension 1
    }
    // Unhandled/incompatible shapes
    else {
        Err(TchError::Kind(format!(
            "Incompatible or unhandled shapes for cos_sim: {:?} and {:?}",
            size1, size2
        )))
    }
}

// -------------------- Tests --------------------

// IMPORTANT: Add `approx` to your Cargo.toml under [dev-dependencies]
// `cargo add approx --dev`
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq; // For float comparisons
    use std::convert::TryFrom;
    use tch::IndexOp;

    const DIM: i64 = 2048;
    const NUM_VECTORS: i64 = 100;
    const SEED: i64 = 42; // For reproducible tests
    const EPSILON: f64 = 1e-6; // Tolerance for float comparisons
    const ORTHOGONALITY_THRESHOLD: f64 = 0.1; // Cosine sim magnitude for "orthogonal enough" random vectors
    const SIMILARITY_THRESHOLD: f64 = 0.1; // Cosine sim magnitude for "similar enough" after bundling

    fn assert_tensor_bipolar(tensor: &Tensor) -> Result<(), TchError> {
        let is_plus_one = tensor.eq(1.0);
        let is_minus_one = tensor.eq(-1.0);
        // Use logical_or instead of | operator
        let is_bipolar = Tensor::logical_or(&is_plus_one, &is_minus_one);
        let all_bipolar = bool::try_from(is_bipolar.all())?; // .all() returns a 0-dim tensor
        assert!(all_bipolar, "Tensor elements are not all +1 or -1");
        Ok(())
    }

    #[test]
    fn test_gen_vectors_invalid_dims() {
        assert!(gen_vectors(0, DIM).is_err());
        assert!(gen_vectors(NUM_VECTORS, 0).is_err());
        assert!(gen_vectors(0, 0).is_err());
        assert!(gen_vectors(-5, DIM).is_err());
        assert!(gen_vectors(NUM_VECTORS, -10).is_err());
    }

    #[test]
    fn test_orthogonality() -> Result<(), TchError> {
        set_seed(SEED);
        let vecs = gen_vectors(NUM_VECTORS, DIM)?;
        // Use IndexOp trait for cleaner indexing: vecs.i(0) instead of vecs.get(0)
        let v0 = vecs.i(0); // Shape [DIM]
        let v1 = vecs.i(1); // Shape [DIM]

        // Dot product of random bipolar vectors should be small relative to dim
        let dot_similarity = dot_sim(&v0, &v1)?;
        let dot_val = f64::try_from(dot_similarity)?;
        println!("Dot sim(v0, v1): {}", dot_val);
        assert_abs_diff_eq!(dot_val / DIM as f64, 0.0, epsilon = ORTHOGONALITY_THRESHOLD); // Normalized dot product close to 0

        // Cosine similarity should also be close to 0
        let cos_similarity = cos_sim(&v0, &v1)?;
        let cos_val = f64::try_from(cos_similarity)?;
        println!("Cos sim(v0, v1): {}", cos_val);
        assert_abs_diff_eq!(cos_val, 0.0, epsilon = ORTHOGONALITY_THRESHOLD);
        Ok(())
    }

    #[test]
    fn test_binding_orthogonality() -> Result<(), TchError> {
        set_seed(SEED);
        let vecs = gen_vectors(3, DIM)?; // Generate 3 vectors
        let v1 = vecs.i(0);
        let v2 = vecs.i(1);
        let v3 = vecs.i(2); // An unrelated vector

        // Bind v1 and v2
        let bound_v1_v2 = bind(&v1, &v2)?;
        assert_eq!(bound_v1_v2.size(), &[DIM]);
        assert_tensor_bipolar(&bound_v1_v2)?; // Binding bipolar vectors yields bipolar vectors

        // The bound vector should be dissimilar (orthogonal) to its components
        let cos_sim_bound_v1 = cos_sim(&bound_v1_v2, &v1)?;
        let cos_val_v1 = f64::try_from(cos_sim_bound_v1)?;
        println!("Cos sim(bound, v1): {}", cos_val_v1);
        assert_abs_diff_eq!(cos_val_v1, 0.0, epsilon = ORTHOGONALITY_THRESHOLD);

        let cos_sim_bound_v2 = cos_sim(&bound_v1_v2, &v2)?;
        let cos_val_v2 = f64::try_from(cos_sim_bound_v2)?;
        println!("Cos sim(bound, v2): {}", cos_val_v2);
        assert_abs_diff_eq!(cos_val_v2, 0.0, epsilon = ORTHOGONALITY_THRESHOLD);

        // The bound vector should also be dissimilar to other random vectors
        let cos_sim_bound_v3 = cos_sim(&bound_v1_v2, &v3)?;
        let cos_val_v3 = f64::try_from(cos_sim_bound_v3)?;
        println!("Cos sim(bound, v3): {}", cos_val_v3);
        assert_abs_diff_eq!(cos_val_v3, 0.0, epsilon = ORTHOGONALITY_THRESHOLD);

        Ok(())
    }

    #[test]
    fn test_bundling_similarity() -> Result<(), TchError> {
        set_seed(SEED);
        let vecs = gen_vectors(3, DIM)?; // Generate 3 vectors
        let v1 = vecs.i(0);
        let v2 = vecs.i(1);
        let v3 = vecs.i(2); // An unrelated vector

        // Bundle v1 and v2
        let bundled_v1_v2 = bundle(&v1, &v2)?;
        assert_eq!(bundled_v1_v2.size(), &[DIM]);
        // Bundled vectors are not necessarily bipolar (-2, 0, 2 possible)

        // The bundled vector should be similar to its components
        let cos_sim_bundled_v1 = cos_sim(&bundled_v1_v2, &v1)?;
        let cos_val_v1 = f64::try_from(cos_sim_bundled_v1)?;
        println!("Cos sim(bundled, v1): {}", cos_val_v1);
        assert!(
            cos_val_v1 > SIMILARITY_THRESHOLD,
            "Bundled vector should be similar to v1"
        );

        let cos_sim_bundled_v2 = cos_sim(&bundled_v1_v2, &v2)?;
        let cos_val_v2 = f64::try_from(cos_sim_bundled_v2)?;
        println!("Cos sim(bundled, v2): {}", cos_val_v2);
        assert!(
            cos_val_v2 > SIMILARITY_THRESHOLD,
            "Bundled vector should be similar to v2"
        );

        // The bundled vector should be dissimilar to unrelated vectors
        let cos_sim_bundled_v3 = cos_sim(&bundled_v1_v2, &v3)?;
        let cos_val_v3 = f64::try_from(cos_sim_bundled_v3)?;
        println!("Cos sim(bundled, v3): {}", cos_val_v3);
        assert_abs_diff_eq!(cos_val_v3, 0.0, epsilon = ORTHOGONALITY_THRESHOLD); // Should be near 0

        Ok(())
    }

    #[test]
    fn test_bind_all() -> Result<(), TchError> {
        set_seed(SEED);
        let num_vecs_to_bind = 5;
        let vecs = gen_vectors(num_vecs_to_bind, DIM)?;

        let bound_all = bind_all(&vecs)?;
        assert_eq!(bound_all.size(), &[DIM]);
        assert_tensor_bipolar(&bound_all)?; // Binding all bipolar vectors yields a bipolar vector

        // Optional: Verify against iterative binding (for small N)
        let v0 = vecs.i(0);
        let v1 = vecs.i(1);
        let v2 = vecs.i(2);
        let v3 = vecs.i(3);
        let v4 = vecs.i(4);
        let manual_bind = bind(&bind(&bind(&bind(&v0, &v1)?, &v2)?, &v3)?, &v4)?;

        // Compare sums to check approximate equality (element-wise might be too strict due to potential float issues)
        assert_abs_diff_eq!(
            f64::try_from(bound_all.sum(Kind::Float))?,
            f64::try_from(manual_bind.sum(Kind::Float))?,
            epsilon = EPSILON
        );

        // Stricter check: element-wise comparison
        let diff = (bound_all - manual_bind).abs().sum(Kind::Float);
        assert_abs_diff_eq!(f64::try_from(diff)?, 0.0, epsilon = EPSILON);

        Ok(())
    }

    #[test]
    fn test_bundle_all() -> Result<(), TchError> {
        set_seed(SEED);
        let num_vecs_to_bundle = 5;
        let vecs = gen_vectors(num_vecs_to_bundle, DIM)?;

        let bundled_all = bundle_all(&vecs)?;
        assert_eq!(bundled_all.size(), &[DIM]);
        // Note: bundled_all elements will likely not be just -1 or 1.

        // Optional: Verify against iterative bundling (for small N)
        let v0 = vecs.i(0);
        let v1 = vecs.i(1);
        let v2 = vecs.i(2);
        let v3 = vecs.i(3);
        let v4 = vecs.i(4);
        let manual_bundle = bundle(&bundle(&bundle(&bundle(&v0, &v1)?, &v2)?, &v3)?, &v4)?;

        // Compare sums
        assert_abs_diff_eq!(
            f64::try_from(bundled_all.sum(Kind::Float))?,
            f64::try_from(manual_bundle.sum(Kind::Float))?,
            epsilon = EPSILON
        );

        // Stricter check: element-wise comparison
        let diff = (bundled_all - manual_bundle).abs().sum(Kind::Float);
        assert_abs_diff_eq!(f64::try_from(diff)?, 0.0, epsilon = EPSILON);

        Ok(())
    }

    #[test]
    fn test_bind_all_single_vector() -> Result<(), TchError> {
        set_seed(SEED);
        let vecs = gen_vectors(1, DIM)?;
        let bound_all = bind_all(&vecs)?;
        assert_eq!(bound_all.size(), &[DIM]);
        // Compare element-wise
        let diff = (vecs.i(0) - bound_all).abs().sum(Kind::Float);
        assert_abs_diff_eq!(f64::try_from(diff)?, 0.0, epsilon = EPSILON);
        Ok(())
    }

    #[test]
    fn test_bundle_all_single_vector() -> Result<(), TchError> {
        set_seed(SEED);
        let vecs = gen_vectors(1, DIM)?;
        let bundled_all = bundle_all(&vecs)?;
        assert_eq!(bundled_all.size(), &[DIM]);
        // Compare element-wise
        let diff = (vecs.i(0) - bundled_all).abs().sum(Kind::Float);
        assert_abs_diff_eq!(f64::try_from(diff)?, 0.0, epsilon = EPSILON);
        Ok(())
    }

    #[test]
    fn test_bind_all_empty() {
        let empty_tensor_result = Tensor::f_zeros(&[0, DIM], (Kind::Float, Device::Cpu));
        assert!(empty_tensor_result.is_ok());
        let empty_tensor = empty_tensor_result.unwrap();
        assert!(bind_all(&empty_tensor).is_err());
    }

    #[test]
    fn test_bundle_all_empty() {
        let empty_tensor_result = Tensor::f_zeros(&[0, DIM], (Kind::Float, Device::Cpu));
        assert!(empty_tensor_result.is_ok());
        let empty_tensor = empty_tensor_result.unwrap();
        assert!(bundle_all(&empty_tensor).is_err());
    }

    #[test]
    fn test_sim_broadcasting() -> Result<(), TchError> {
        set_seed(SEED);
        let vecs_n_d = gen_vectors(5, DIM)?; // [5, D]
        let vec_d = gen_vectors(1, DIM)?.squeeze_dim(0); // [D]

        // Dot sim: [5, D] vs [D] -> [5]
        let dot_sims = dot_sim(&vecs_n_d, &vec_d)?;
        assert_eq!(dot_sims.size(), &[5]);

        // Cos sim: [5, D] vs [D] -> [5]
        let cos_sims = cos_sim(&vecs_n_d, &vec_d)?;
        assert_eq!(cos_sims.size(), &[5]);

        // Check symmetry ( [D] vs [N, D] -> [5] )
        let dot_sims_rev = dot_sim(&vec_d, &vecs_n_d)?;
        assert_eq!(dot_sims_rev.size(), &[5]);
        let cos_sims_rev = cos_sim(&vec_d, &vecs_n_d)?;
        assert_eq!(cos_sims_rev.size(), &[5]);

        // Compare results (should be identical)
        assert_abs_diff_eq!(
            f64::try_from(dot_sims.sum(Kind::Float))?,
            f64::try_from(dot_sims_rev.sum(Kind::Float))?,
            epsilon = EPSILON
        );
        assert_abs_diff_eq!(
            f64::try_from(cos_sims.sum(Kind::Float))?,
            f64::try_from(cos_sims_rev.sum(Kind::Float))?,
            epsilon = EPSILON
        );

        Ok(())
    }

    #[test]
    fn test_dot_sim_empty_dim() -> Result<(), TchError> {
        let t1 = Tensor::f_zeros(&[5, 0], (Kind::Float, Device::Cpu))?;
        let t2 = Tensor::f_zeros(&[5, 0], (Kind::Float, Device::Cpu))?;
        let dot = dot_sim(&t1, &t2)?;
        assert_eq!(dot.size(), &[5]); // Summing over dim 0 gives shape [5]
        assert_abs_diff_eq!(f64::try_from(dot.sum(Kind::Float))?, 0.0, epsilon = EPSILON); // Should be all zeros
        Ok(())
    }

    #[test]
    fn test_cos_sim_requires_grad() -> Result<(), TchError> {
        // Ensure cos_sim works even if inputs require grad (tch might use different kernels)
        set_seed(SEED);
        let v1 = gen_vectors(1, DIM)?.squeeze_dim(0);
        let v2 = gen_vectors(1, DIM)?.squeeze_dim(0);

        let cos_sim_val = cos_sim(&v1, &v2)?;
        let _ = f64::try_from(cos_sim_val)?; // Check it computed a value
        Ok(())
    }
}
