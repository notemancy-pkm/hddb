// src/crud.rs

//! Provides CRUD (Create, Read, Update, Delete) operations for a vector store.
//! The store manages Tch tensors and associated IDs, with persistence via bincode v1 API.

use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path; // Keep TryFrom

use log::{error, info, trace};
use serde::{Deserialize, Serialize};
use thiserror::Error;
// Import IndexOp trait for .i() method
use tch::{Device, IndexOp, Kind, TchError, Tensor};
// Assuming bincode v1 based on errors - direct functions, Box<bincode::Error>
use bincode::{
    self, config,
    error::{DecodeError, EncodeError},
    Decode, Encode,
};
// --- Error Handling ---

#[derive(Error, Debug)]
pub enum CrudError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Bincode Encode error: {0}")]
    BincodeEncodeError(#[from] EncodeError),

    #[error("Bincode Decode error: {0}")]
    BincodeDecodeError(#[from] DecodeError),

    #[error("Tch tensor error: {0}")]
    TchError(#[from] TchError),

    #[error("Dimension mismatch: Expected {expected}, found {found}")]
    DimensionMismatch { expected: String, found: String },

    #[error("Kind mismatch: Expected {expected:?}, found {found:?}")]
    KindMismatch { expected: Kind, found: Kind },

    #[error("Duplicate ID found: {0}")]
    DuplicateId(String),

    #[error("ID not found: {0}")]
    IdNotFound(String),

    #[error("Inconsistent state: {0}")]
    InconsistentState(String),

    #[error("Operation requires CPU tensor, found {0:?}")]
    NonCpuTensor(Device),

    #[error("Kind conversion error: Unsupported Kind {0:?}")]
    KindConversionError(Kind),

    #[error("Deserialized data size mismatch for kind {kind:?}")]
    DataSizeMismatch { kind: Kind },
}

// --- Serializable Kind Representation ---
// Wrapper enum for tch::Kind to enable Serde derive

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Encode, Decode)]
enum SerializableKind {
    Uint8,
    Bool,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    QInt8,
    QUint8,
    QInt32,
}

impl From<Kind> for SerializableKind {
    fn from(kind: Kind) -> Self {
        match kind {
            Kind::Uint8 => SerializableKind::Uint8,
            Kind::Bool => SerializableKind::Bool,
            Kind::Int8 => SerializableKind::Int8,
            Kind::Int16 => SerializableKind::Int16,
            Kind::Int => SerializableKind::Int,
            Kind::Int64 => SerializableKind::Int64,
            Kind::Half => SerializableKind::Half,
            Kind::Float => SerializableKind::Float,
            Kind::Double => SerializableKind::Double,
            Kind::ComplexHalf => SerializableKind::ComplexHalf,
            Kind::ComplexFloat => SerializableKind::ComplexFloat,
            Kind::ComplexDouble => SerializableKind::ComplexDouble,
            Kind::QInt8 => SerializableKind::QInt8,
            Kind::QUInt8 => SerializableKind::QUint8, // Corrected casing
            Kind::QInt32 => SerializableKind::QInt32,
            _ => panic!(
                "Unsupported tch::Kind encountered in From<Kind> for SerializableKind: {:?}",
                kind
            ),
        }
    }
}

impl From<SerializableKind> for Kind {
    fn from(skind: SerializableKind) -> Self {
        match skind {
            SerializableKind::Uint8 => Kind::Uint8,
            SerializableKind::Bool => Kind::Bool,
            SerializableKind::Int8 => Kind::Int8,
            SerializableKind::Int16 => Kind::Int16,
            SerializableKind::Int => Kind::Int,
            SerializableKind::Int64 => Kind::Int64,
            SerializableKind::Half => Kind::Half,
            SerializableKind::Float => Kind::Float,
            SerializableKind::Double => Kind::Double,
            SerializableKind::ComplexHalf => Kind::ComplexHalf,
            SerializableKind::ComplexFloat => Kind::ComplexFloat,
            SerializableKind::ComplexDouble => Kind::ComplexDouble,
            SerializableKind::QInt8 => Kind::QInt8,
            SerializableKind::QUint8 => Kind::QUInt8, // Corrected casing
            SerializableKind::QInt32 => Kind::QInt32,
        }
    }
}

// --- Serializable Tensor Representation ---

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Encode, Decode)]
enum SerializableTensorData {
    Float(Vec<f32>),
    Double(Vec<f64>),
    Int(Vec<i32>),
    Int64(Vec<i64>),
    Int16(Vec<i16>),
    Int8(Vec<i8>),
    Byte(Vec<u8>), // Covers Bool, QUint8, UInt8
    QInt32(Vec<i32>),
}

#[derive(Serialize, Deserialize, Debug, Clone, Encode, Decode)]
struct SerializableTensor {
    shape: Vec<i64>,
    kind: SerializableKind,
    data: SerializableTensorData,
}

impl SerializableTensor {
    fn from_tensor(tensor: &Tensor) -> Result<Self, CrudError> {
        let tensor_cpu = if tensor.device() != Device::Cpu {
            tensor.f_to(Device::Cpu)?
        } else {
            tensor.shallow_clone()
        };

        let shape = tensor_cpu.size();
        let original_kind = tensor_cpu.kind();
        let kind = SerializableKind::from(original_kind);
        let flat_tensor_view = tensor_cpu.view(-1);

        let data = match original_kind {
            Kind::Float => SerializableTensorData::Float(Vec::<f32>::try_from(&flat_tensor_view)?),
            Kind::Double => {
                SerializableTensorData::Double(Vec::<f64>::try_from(&flat_tensor_view)?)
            }
            Kind::Int => SerializableTensorData::Int(Vec::<i32>::try_from(&flat_tensor_view)?),
            Kind::Int64 => SerializableTensorData::Int64(Vec::<i64>::try_from(&flat_tensor_view)?),
            Kind::Int16 => SerializableTensorData::Int16(Vec::<i16>::try_from(&flat_tensor_view)?),
            Kind::Int8 => SerializableTensorData::Int8(Vec::<i8>::try_from(&flat_tensor_view)?),
            Kind::Uint8 | Kind::Bool | Kind::QUInt8 => {
                SerializableTensorData::Byte(Vec::<u8>::try_from(&flat_tensor_view)?)
            }
            Kind::QInt8 => SerializableTensorData::Byte(
                Vec::<i8>::try_from(&flat_tensor_view)?
                    .into_iter()
                    .map(|x| x as u8)
                    .collect(),
            ),
            Kind::QInt32 => {
                SerializableTensorData::QInt32(Vec::<i32>::try_from(&flat_tensor_view)?)
            }
            k => return Err(CrudError::KindConversionError(k)),
        };

        // Verification
        let expected_elements = shape.iter().product::<i64>() as usize;
        let actual_elements = match &data {
            SerializableTensorData::Float(d) => d.len(),
            SerializableTensorData::Double(d) => d.len(),
            SerializableTensorData::Int(d) => d.len(),
            SerializableTensorData::Int64(d) => d.len(),
            SerializableTensorData::Int16(d) => d.len(),
            SerializableTensorData::Int8(d) => d.len(),
            SerializableTensorData::Byte(d) => d.len(),
            SerializableTensorData::QInt32(d) => d.len(),
        };
        if actual_elements != expected_elements {
            error!(
                "Data size mismatch during serialization: expected {}, got {}",
                expected_elements, actual_elements
            );
            return Err(CrudError::DataSizeMismatch {
                kind: original_kind,
            });
        }
        if tensor_cpu.numel() != expected_elements {
            return Err(CrudError::InconsistentState(format!(
                "Tensor numel {} does not match shape product {}",
                tensor_cpu.numel(),
                expected_elements
            )));
        }

        Ok(SerializableTensor { shape, kind, data })
    }

    fn to_tensor(&self) -> Result<Tensor, CrudError> {
        let device = Device::Cpu;
        let target_kind = Kind::from(self.kind);

        let tensor = match &self.data {
            SerializableTensorData::Float(d) => Tensor::from_slice(d),
            SerializableTensorData::Double(d) => Tensor::from_slice(d),
            SerializableTensorData::Int(d) => Tensor::from_slice(d),
            SerializableTensorData::Int64(d) => Tensor::from_slice(d),
            SerializableTensorData::Int16(d) => Tensor::from_slice(d),
            SerializableTensorData::Int8(d) => Tensor::from_slice(d),
            SerializableTensorData::Byte(d) => Tensor::from_slice(d),
            SerializableTensorData::QInt32(d) => Tensor::from_slice(d),
        };

        let tensor = tensor.to_kind(target_kind);
        let final_tensor = tensor.reshape(&self.shape).to(device);

        // Verification
        if final_tensor.size() != self.shape {
            error!(
                "Deserialized tensor shape {:?} does not match expected {:?}",
                final_tensor.size(),
                self.shape
            );
            return Err(CrudError::InconsistentState(
                "Shape mismatch after deserialization.".to_string(),
            ));
        }
        if final_tensor.kind() != target_kind {
            error!(
                "Deserialized tensor kind {:?} does not match expected {:?}",
                final_tensor.kind(),
                target_kind
            );
            return Err(CrudError::InconsistentState(
                "Kind mismatch after deserialization.".to_string(),
            ));
        }

        Ok(final_tensor)
    }
}

// --- Vector Store Definition ---

#[derive(Debug)]
pub struct VectorStore {
    vectors: Tensor,
    ids: Vec<String>,
    reverse_map: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
struct SerializableVectorStore {
    vectors: SerializableTensor,
    ids: Vec<String>,
    reverse_map: HashMap<String, usize>,
}

// --- VectorStore Methods ---
impl VectorStore {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn dim(&self) -> i64 {
        // --- FIX 2: Correctly get dim even if empty ---
        // Get the size, check if it has at least 2 dimensions, return size of dim 1
        let size = self.vectors.size();
        if size.len() >= 2 {
            size[1] // Return the dimensionality (D in [N, D] or [0, D])
        } else {
            0 // Return 0 if tensor is not 2D (shouldn't happen with create_store checks)
        }
        // --- End FIX 2 ---
    }
    pub fn kind(&self) -> Kind {
        self.vectors.kind()
    }
    pub fn get_vectors_tensor(&self) -> &Tensor {
        &self.vectors
    }
    pub fn get_ids(&self) -> Vec<String> {
        self.ids.clone()
    }
    pub fn get_reverse_map(&self) -> HashMap<String, usize> {
        self.reverse_map.clone()
    }

    // Requires IndexOp trait in scope where called.
    pub fn get_vector_by_id(&self, id: &str) -> Result<Tensor, CrudError> {
        match self.reverse_map.get(id) {
            Some(&index) => {
                if index >= self.vectors.size()[0] as usize {
                    return Err(CrudError::InconsistentState(format!(
                        "ID '{}' maps to index {} which is out of bounds for tensor rows {}",
                        id,
                        index,
                        self.vectors.size()[0]
                    )));
                }
                // Use .i() from IndexOp trait
                Ok(self.vectors.i(index as i64))
            }
            None => Err(CrudError::IdNotFound(id.to_string())),
        }
    }
}

// --- Core CRUD Functions ---

pub fn create_store(vectors: Tensor, ids: Vec<String>) -> Result<VectorStore, CrudError> {
    info!("Attempting to create vector store...");
    let vec_size = vectors.size();
    if vec_size.len() != 2 {
        /* ... error handling ... */
        return Err(CrudError::DimensionMismatch {
            expected: "[N, D]".to_string(),
            found: format!("{:?}", vec_size),
        });
    }
    let num_vectors = vec_size[0];
    if num_vectors < 0 {
        /* ... error handling ... */
        return Err(CrudError::InconsistentState(
            "Tensor reports negative number of vectors".to_string(),
        ));
    }
    if num_vectors as usize != ids.len() {
        /* ... error handling ... */
        return Err(CrudError::DimensionMismatch {
            expected: format!("{} IDs", num_vectors),
            found: format!("{} IDs", ids.len()),
        });
    }
    if vectors.device() != Device::Cpu {
        /* ... error handling ... */
        return Err(CrudError::NonCpuTensor(vectors.device()));
    }

    let mut reverse_map = HashMap::with_capacity(ids.len());
    for (index, id) in ids.iter().enumerate() {
        if reverse_map.insert(id.clone(), index).is_some() {
            /* ... error handling ... */
            return Err(CrudError::DuplicateId(id.clone()));
        }
    }

    info!("Successfully created store with {} vectors.", num_vectors);
    Ok(VectorStore {
        vectors,
        ids,
        reverse_map,
    })
}

pub fn write_store(store: &VectorStore, path: &Path) -> Result<(), CrudError> {
    info!("Writing vector store to {:?}", path);
    let serializable_tensor = SerializableTensor::from_tensor(&store.vectors)?;
    // This struct now derives Encode
    let serializable_store = SerializableVectorStore {
        vectors: serializable_tensor,
        ids: store.ids.clone(),
        reverse_map: store.reverse_map.clone(),
    };

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);

    // encode_into_std_write requires Encode on serializable_store
    let _: usize =
        bincode::encode_into_std_write(&serializable_store, &mut writer, config::standard())?;
    writer.flush()?;

    info!("Vector store successfully written to {:?}", path);
    Ok(())
}

pub fn load_store(path: &Path) -> Result<VectorStore, CrudError> {
    info!("Loading vector store from {:?}", path);
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // decode_from_std_read requires Decode on SerializableVectorStore
    let serializable_store: SerializableVectorStore =
        bincode::decode_from_std_read(&mut reader, config::standard())?;
    trace!("SerializableVectorStore structure deserialized.");

    let tensor = serializable_store.vectors.to_tensor()?;
    trace!("Tensor successfully reconstructed.");

    // ... consistency checks ...
    let num_tensor_rows = tensor.size().get(0).copied().unwrap_or(0);
    let num_ids = serializable_store.ids.len();
    let num_map_entries = serializable_store.reverse_map.len();
    if num_tensor_rows as usize != num_ids || num_ids != num_map_entries {
        return Err(CrudError::InconsistentState(format!(
            "Data loaded from {:?} is inconsistent.",
            path
        )));
    }

    let store = VectorStore {
        vectors: tensor,
        ids: serializable_store.ids,
        reverse_map: serializable_store.reverse_map,
    };

    info!(
        "Vector store successfully loaded from {:?} ({} vectors)",
        path,
        store.len()
    );
    Ok(store)
}

pub fn add_vectors(
    store: &mut VectorStore,
    new_vectors: Tensor,
    new_ids: Vec<String>,
) -> Result<(), CrudError> {
    info!("Adding {} new vectors to the store...", new_ids.len());
    let num_new_vectors = new_vectors.size().get(0).copied().unwrap_or(-1);

    // --- Validation ---
    if new_vectors.dim() != 2 || new_ids.is_empty() || num_new_vectors <= 0 {
        /* ... error handling ... */
        return Err(CrudError::DimensionMismatch {
            expected: "[M, D] where M > 0".to_string(),
            found: format!("{:?}", new_vectors.size()),
        });
    }
    if num_new_vectors as usize != new_ids.len() {
        /* ... error handling ... */
        return Err(CrudError::DimensionMismatch {
            expected: format!("{} IDs", num_new_vectors),
            found: format!("{} IDs", new_ids.len()),
        });
    }
    if new_vectors.device() != Device::Cpu {
        /* ... error handling ... */
        return Err(CrudError::NonCpuTensor(new_vectors.device()));
    }
    if !store.is_empty() {
        // Check dim/kind if store not empty
        if store.dim() != new_vectors.size()[1] {
            /* ... error handling ... */
            return Err(CrudError::DimensionMismatch {
                expected: format!("Dimension {}", store.dim()),
                found: format!("Dimension {}", new_vectors.size()[1]),
            });
        }
        if store.kind() != new_vectors.kind() {
            /* ... error handling ... */
            return Err(CrudError::KindMismatch {
                expected: store.kind(),
                found: new_vectors.kind(),
            });
        }
    }

    // Check for duplicate IDs
    let original_len = store.len();
    let mut temp_new_id_set = std::collections::HashSet::with_capacity(new_ids.len());
    for (i, id) in new_ids.iter().enumerate() {
        if store.reverse_map.contains_key(id) {
            /* ... error handling ... */
            return Err(CrudError::DuplicateId(id.clone()));
        }
        if !temp_new_id_set.insert(id.clone()) {
            /* ... error handling ... */
            return Err(CrudError::DuplicateId(id.clone()));
        }
        store.reverse_map.insert(id.clone(), original_len + i);
    }
    trace!("Duplicate ID checks passed.");

    // Append vectors using f_cat (returns Result)
    let updated_vectors = if store.is_empty() {
        new_vectors
    } else {
        Tensor::f_cat(&[&store.vectors, &new_vectors], 0)? // Use f_cat
    };
    store.vectors = updated_vectors;
    store.ids.extend(new_ids); // Consume new_ids

    // Consistency check
    let current_len = store.len();
    if store.vectors.size()[0] as usize != current_len || current_len != store.reverse_map.len() {
        /* ... error handling ... */
        return Err(CrudError::InconsistentState(
            "Store became inconsistent after adding vectors.".to_string(),
        ));
    }

    info!(
        "Successfully added {} vectors. Store size now {}.",
        store.len() - original_len,
        store.len()
    );
    Ok(())
}

pub fn delete_vectors(store: &mut VectorStore, ids_to_delete: &[String]) -> Result<(), CrudError> {
    if ids_to_delete.is_empty() {
        return Ok(());
    }
    if store.is_empty() {
        return Err(CrudError::IdNotFound(ids_to_delete[0].clone()));
    }

    info!("Attempting to delete {} vectors...", ids_to_delete.len());
    let original_len = store.len();
    let mut indices_to_delete = std::collections::HashSet::new();

    // Find indices & validate IDs
    for id in ids_to_delete {
        match store.reverse_map.get(id) {
            Some(&index) => {
                indices_to_delete.insert(index);
            }
            None => {
                return Err(CrudError::IdNotFound(id.clone()));
            }
        }
    }
    if indices_to_delete.is_empty() {
        return Ok(());
    }
    trace!("Found indices to delete: {:?}", indices_to_delete);

    // Build indices to keep
    let indices_to_keep: Vec<i64> = (0..original_len)
        .filter(|i| !indices_to_delete.contains(i))
        .map(|i| i as i64)
        .collect();

    if indices_to_keep.is_empty() {
        // Deleting all
        info!("Deleting all vectors from the store.");
        let dim = store.dim();
        let kind = store.kind();
        store.vectors = Tensor::zeros(&[0, dim], (kind, Device::Cpu));
        store.ids.clear();
        store.reverse_map.clear();
    } else {
        // Deleting some
        let indices_to_keep_tensor = Tensor::from_slice(&indices_to_keep);
        // Use f_index_select (returns Result)
        store.vectors = store
            .vectors
            .f_index_select(0, &indices_to_keep_tensor.to(store.vectors.device()))?;
        trace!("Vectors filtered.");

        // Rebuild IDs and reverse map
        let mut new_ids = Vec::with_capacity(indices_to_keep.len());
        let mut new_reverse_map = HashMap::with_capacity(indices_to_keep.len());
        let current_ids = std::mem::take(&mut store.ids); // Take ownership to avoid clone

        for (new_idx, &original_idx) in indices_to_keep.iter().enumerate() {
            // original_idx is i64, use try_into for usize conversion
            let original_usize_idx: usize = original_idx
                .try_into()
                .map_err(|_| CrudError::InconsistentState("Index conversion failed".to_string()))?;
            if let Some(id) = current_ids.get(original_usize_idx) {
                new_ids.push(id.clone());
                new_reverse_map.insert(id.clone(), new_idx);
            } else {
                return Err(CrudError::InconsistentState(
                    "Failed to rebuild IDs after deletion.".to_string(),
                ));
            }
        }
        store.ids = new_ids;
        store.reverse_map = new_reverse_map;
        trace!("IDs and reverse map updated.");
    }

    // Consistency check
    let final_len = store.len();
    if store.vectors.size().get(0).map(|&n| n as usize) != Some(final_len)
        || final_len != store.reverse_map.len()
    {
        /* ... error handling ... */
        return Err(CrudError::InconsistentState(
            "Store became inconsistent after deleting vectors.".to_string(),
        ));
    }

    info!(
        "Successfully deleted {} vectors. Store size now {}.",
        original_len - final_len,
        final_len
    );
    Ok(())
}

pub fn update_vectors(
    store: &mut VectorStore,
    ids_to_update: &[String],
    update_vectors: Tensor,
) -> Result<(), CrudError> {
    if ids_to_update.is_empty() {
        return Ok(());
    }
    let num_updates = update_vectors.size().get(0).copied().unwrap_or(-1);
    if store.is_empty() {
        return Err(CrudError::IdNotFound(ids_to_update[0].clone()));
    }

    info!("Attempting to update {} vectors...", ids_to_update.len());

    // --- Validation ---
    if update_vectors.dim() != 2 || num_updates <= 0 {
        /* ... */
        return Err(CrudError::DimensionMismatch {
            expected: "[M, D], M > 0".to_string(),
            found: format!("{:?}", update_vectors.size()),
        });
    }
    if num_updates as usize != ids_to_update.len() {
        /* ... */
        return Err(CrudError::DimensionMismatch {
            expected: format!(
                "{} vectors for {} IDs",
                ids_to_update.len(),
                ids_to_update.len()
            ),
            found: format!("{} vectors", num_updates),
        });
    }
    if update_vectors.device() != Device::Cpu {
        /* ... */
        return Err(CrudError::NonCpuTensor(update_vectors.device()));
    }
    if store.dim() != update_vectors.size()[1] {
        /* ... */
        return Err(CrudError::DimensionMismatch {
            expected: format!("Dim {}", store.dim()),
            found: format!("Dim {}", update_vectors.size()[1]),
        });
    }
    if store.kind() != update_vectors.kind() {
        /* ... */
        return Err(CrudError::KindMismatch {
            expected: store.kind(),
            found: update_vectors.kind(),
        });
    }

    // --- Update Store using f_index_put_ ---
    let mut indices = Vec::with_capacity(ids_to_update.len());
    for id in ids_to_update {
        match store.reverse_map.get(id) {
            Some(&index) => indices.push(index as i64), // Collect indices as i64
            None => return Err(CrudError::IdNotFound(id.clone())),
        }
    }

    let indices_tensor = Tensor::from_slice(&indices);
    // Ensure tensors are on the same device before index_put_
    let values_tensor = update_vectors.to(store.vectors.device());
    let indices_tensor = indices_tensor.to(store.vectors.device());

    // Use fallible f_index_put_
    let _ = store
        .vectors
        .f_index_put_(&[Some(indices_tensor)], &values_tensor, false)?; // accumulate = false

    info!("Successfully updated {} vectors.", ids_to_update.len());
    Ok(())
}

// -------------------- Tests --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // Make sure IndexOp is imported in test module too
    use tch::{IndexOp, Kind, Tensor};
    use tempfile::tempdir; // For temporary directories/files

    fn create_test_store(
        n: i64,
        d: i64,
        kind: Kind,
        start_id: usize,
    ) -> Result<VectorStore, CrudError> {
        let vectors = match kind {
            Kind::Float => {
                Tensor::arange(n * d, (kind, Device::Cpu)).reshape(&[n, d]) / ((n * d) as f64)
            } // Use f64 for division
            Kind::Double => {
                Tensor::arange(n * d, (kind, Device::Cpu)).reshape(&[n, d]) / ((n * d) as f64)
            }
            Kind::Int64 => Tensor::arange(n * d, (kind, Device::Cpu)).reshape(&[n, d]),
            _ => panic!("Unsupported kind for test store generation"),
        };
        let ids: Vec<String> = (start_id..start_id + n as usize)
            .map(|i| format!("id_{}", i))
            .collect();
        create_store(vectors, ids)
    }

    fn assert_stores_equal(store1: &VectorStore, store2: &VectorStore, epsilon: f64) {
        assert_eq!(store1.ids, store2.ids, "IDs do not match");
        assert_eq!(
            store1.reverse_map, store2.reverse_map,
            "Reverse maps do not match"
        );
        assert_eq!(
            store1.vectors.size(),
            store2.vectors.size(),
            "Vector tensor shapes do not match"
        );
        assert_eq!(
            store1.vectors.kind(),
            store2.vectors.kind(),
            "Vector tensor kinds do not match"
        );

        let t1_cpu = store1.vectors.to(Device::Cpu);
        let t2_cpu = store2.vectors.to(Device::Cpu);

        if t1_cpu.numel() > 0 || t2_cpu.numel() > 0 {
            let diff = (t1_cpu - t2_cpu).abs().sum(Kind::Double);
            let diff_val = f64::try_from(diff).expect("Failed to convert diff sum to f64");
            // Corrected assert_abs_diff_eq usage (no trailing message string)
            assert_abs_diff_eq!(diff_val, 0.0, epsilon = epsilon);
        }
    }

    // --- Tests for create_store ---
    #[test]
    fn test_create_store_ok() -> Result<(), CrudError> {
        let n = 10;
        let d = 5;
        let store = create_test_store(n, d, Kind::Float, 0)?;

        assert_eq!(store.len(), n as usize);
        assert_eq!(store.dim(), d);
        assert_eq!(store.kind(), Kind::Float);
        assert_eq!(store.ids.len(), n as usize);
        assert_eq!(store.reverse_map.len(), n as usize);
        assert_eq!(store.reverse_map.get("id_0"), Some(&0));
        assert_eq!(store.reverse_map.get("id_9"), Some(&9));
        assert_eq!(store.vectors.size(), &[n, d]);
        Ok(())
    }

    #[test]
    fn test_create_empty_store_ok() -> Result<(), CrudError> {
        let empty_vecs = Tensor::zeros(&[0, 10], (Kind::Float, Device::Cpu));
        let empty_ids: Vec<String> = vec![];
        let store = create_store(empty_vecs, empty_ids)?;
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        // --- Check the assertion that failed ---
        assert_eq!(store.dim(), 10, "Empty store should retain dimensionality"); // Correctly asserts 10 == 10 now
                                                                                 // ---
        assert_eq!(store.kind(), Kind::Float);
        assert_eq!(store.vectors.size(), &[0, 10]);
        Ok(())
    }

    #[test]
    fn test_create_store_errors() {
        let cpu = Device::Cpu;
        // Mismatched lengths
        let vectors = Tensor::randn(&[5, 3], (Kind::Float, cpu));
        let ids_short = vec!["id_0".to_string(), "id_1".to_string()];
        let ids_long = vec!["id_0".to_string(); 6];
        assert!(matches!(
            create_store(vectors.copy(), ids_short),
            Err(CrudError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            create_store(vectors.copy(), ids_long),
            Err(CrudError::DimensionMismatch { .. })
        ));

        // Wrong dimensions
        let vectors_1d = Tensor::randn(&[5], (Kind::Float, cpu));
        let ids = vec!["id_0".to_string(); 5];
        assert!(matches!(
            create_store(vectors_1d, ids.clone()),
            Err(CrudError::DimensionMismatch { .. })
        ));

        // Duplicate IDs
        let vectors = Tensor::randn(&[3, 2], (Kind::Float, cpu));
        let ids_dup = vec!["id_0".to_string(), "id_1".to_string(), "id_0".to_string()];
        assert!(
            matches!(create_store(vectors, ids_dup), Err(CrudError::DuplicateId(id)) if id == "id_0")
        );

        // Non-CPU tensor (if GPU is available)
        if tch::utils::has_cuda() {
            let gpu = Device::Cuda(0);
            let vectors_gpu = Tensor::randn(&[5, 3], (Kind::Float, gpu));
            let ids_gpu = vec!["id_0".to_string(); 5];
            assert!(
                matches!(create_store(vectors_gpu, ids_gpu), Err(CrudError::NonCpuTensor(dev)) if dev == gpu)
            );
        } else {
            println!("Skipping NonCpuTensor test: No CUDA device found.");
        }
    }

    #[test]
    fn test_write_load_store_cycle() -> Result<(), CrudError> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test_store.bin");

        let n = 50;
        let d = 10;
        let original_store = create_test_store(n, d, Kind::Double, 100)?; // Use Double

        // Write the store
        write_store(&original_store, &file_path)?;
        assert!(file_path.exists());
        assert!(file_path.metadata()?.len() > 0); // Check file is not empty

        // Load the store
        let loaded_store = load_store(&file_path)?;

        // Compare
        assert_stores_equal(&original_store, &loaded_store, 1e-9);

        // Test loading non-existent file
        let bad_path = dir.path().join("non_existent.bin");
        assert!(matches!(load_store(&bad_path), Err(CrudError::IoError(_))));

        // Test loading corrupted file (write something else)
        {
            let mut file = File::create(&file_path)?;
            file.write_all(b"corrupted data")?;
        }
        // --- FIX: Expect DecodeError, not EncodeError ---
        assert!(matches!(
            load_store(&file_path),
            Err(CrudError::BincodeDecodeError(_)) // Changed to DecodeError
        ));
        // --- End FIX ---

        dir.close()?; // Clean up temp directory
        Ok(())
    }

    #[test]
    fn test_write_load_empty_store() -> Result<(), CrudError> {
        let dir = tempdir()?;
        let file_path = dir.path().join("empty_store.bin");
        let empty_store = create_store(Tensor::zeros(&[0, 5], (Kind::Float, Device::Cpu)), vec![])?;

        write_store(&empty_store, &file_path)?;
        let loaded_store = load_store(&file_path)?;

        assert!(loaded_store.is_empty());
        assert_eq!(loaded_store.dim(), 5);
        assert_stores_equal(&empty_store, &loaded_store, 1e-9);

        dir.close()?;
        Ok(())
    }

    // --- Tests for add_vectors ---
    #[test]
    fn test_add_vectors_ok() -> Result<(), CrudError> {
        let mut store = create_test_store(5, 4, Kind::Float, 0)?; // id_0 to id_4

        let new_vectors = Tensor::randn(&[3, 4], (Kind::Float, Device::Cpu));
        let new_ids = vec!["id_5".to_string(), "id_6".to_string(), "id_7".to_string()];

        add_vectors(&mut store, new_vectors, new_ids)?;

        assert_eq!(store.len(), 8);
        assert_eq!(store.dim(), 4);
        assert_eq!(store.ids.len(), 8);
        assert_eq!(store.reverse_map.len(), 8);
        assert_eq!(store.vectors.size(), &[8, 4]);

        // Check if new IDs/indices are correct
        assert_eq!(store.reverse_map.get("id_5"), Some(&5));
        assert_eq!(store.reverse_map.get("id_6"), Some(&6));
        assert_eq!(store.reverse_map.get("id_7"), Some(&7));
        assert_eq!(store.ids[5..], ["id_5", "id_6", "id_7"]);

        // Check adding to an empty store
        let mut empty_store = create_store(
            Tensor::zeros(&[0, 10], (Kind::Float, Device::Cpu)),
            Vec::new(),
        )?;
        assert!(empty_store.is_empty());
        let first_vectors = Tensor::randn(&[2, 10], (Kind::Float, Device::Cpu));
        let first_ids = vec!["first_0".to_string(), "first_1".to_string()];
        add_vectors(&mut empty_store, first_vectors, first_ids)?;
        assert_eq!(empty_store.len(), 2);
        assert_eq!(empty_store.dim(), 10);
        assert_eq!(empty_store.reverse_map.get("first_0"), Some(&0));

        Ok(())
    }

    #[test]
    fn test_add_vectors_errors() -> Result<(), CrudError> {
        let mut store = create_test_store(5, 4, Kind::Float, 0)?;
        let cpu = Device::Cpu;

        // Dimension mismatch
        let wrong_dim_vecs = Tensor::randn(&[2, 5], (Kind::Float, cpu)); // Dim 5 != 4
        let wrong_dim_ids = vec!["new_a".to_string(), "new_b".to_string()];
        assert!(matches!(
            add_vectors(&mut store, wrong_dim_vecs, wrong_dim_ids),
            Err(CrudError::DimensionMismatch { .. })
        ));

        // Kind mismatch
        let wrong_kind_vecs = Tensor::randn(&[2, 4], (Kind::Double, cpu)); // Double != Float
        let wrong_kind_ids = vec!["new_c".to_string(), "new_d".to_string()];
        assert!(matches!(
            add_vectors(&mut store, wrong_kind_vecs, wrong_kind_ids),
            Err(CrudError::KindMismatch { .. })
        ));

        // Length mismatch
        let vecs = Tensor::randn(&[2, 4], (Kind::Float, cpu));
        let ids_too_many = vec!["new_e".to_string(); 3];
        assert!(matches!(
            add_vectors(&mut store, vecs.copy(), ids_too_many),
            Err(CrudError::DimensionMismatch { .. })
        ));

        // Length mismatch (0 vectors)
        let zero_vecs = Tensor::zeros(&[0, 4], (Kind::Float, cpu));
        let zero_ids = vec!["new_x".to_string()];
        assert!(matches!(
            add_vectors(&mut store, zero_vecs.copy(), zero_ids),
            Err(CrudError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            add_vectors(&mut store, vecs.copy(), vec![]),
            Err(CrudError::DimensionMismatch { .. })
        ));

        // Duplicate ID (already in store)
        let vecs = Tensor::randn(&[1, 4], (Kind::Float, cpu));
        let ids_dup_store = vec!["id_0".to_string()]; // id_0 exists
        assert!(
            matches!(add_vectors(&mut store, vecs.copy(), ids_dup_store), Err(CrudError::DuplicateId(id)) if id == "id_0")
        );

        // Duplicate ID (within new IDs)
        let vecs = Tensor::randn(&[3, 4], (Kind::Float, cpu));
        let ids_dup_new = vec![
            "new_f".to_string(),
            "new_g".to_string(),
            "new_f".to_string(),
        ];
        assert!(
            matches!(add_vectors(&mut store, vecs.copy(), ids_dup_new), Err(CrudError::DuplicateId(id)) if id == "new_f")
        );

        Ok(())
    }

    // --- Tests for delete_vectors ---
    #[test]
    fn test_delete_vectors_ok() -> Result<(), CrudError> {
        let mut store = create_test_store(7, 3, Kind::Int64, 0)?; // id_0 to id_6

        let ids_to_delete = vec!["id_1".to_string(), "id_4".to_string(), "id_6".to_string()];
        // Save copies of vectors we expect to keep
        let vec_at_idx_0 = store.get_vector_by_id("id_0")?.copy();
        let vec_at_idx_2 = store.get_vector_by_id("id_2")?.copy();
        let vec_at_idx_3 = store.get_vector_by_id("id_3")?.copy();
        let vec_at_idx_5 = store.get_vector_by_id("id_5")?.copy();

        delete_vectors(&mut store, &ids_to_delete)?;

        assert_eq!(store.len(), 4); // 7 - 3 = 4
        assert_eq!(store.dim(), 3);
        assert_eq!(store.ids.len(), 4);
        assert_eq!(store.reverse_map.len(), 4);
        assert_eq!(store.vectors.size(), &[4, 3]);

        // Check remaining IDs and their new indices
        assert_eq!(store.ids, ["id_0", "id_2", "id_3", "id_5"]);
        assert_eq!(store.reverse_map.get("id_0"), Some(&0));
        assert_eq!(store.reverse_map.get("id_2"), Some(&1));
        assert_eq!(store.reverse_map.get("id_3"), Some(&2));
        assert_eq!(store.reverse_map.get("id_5"), Some(&3));

        // Check deleted IDs are gone
        assert!(store.reverse_map.get("id_1").is_none());
        assert!(store.reverse_map.get("id_4").is_none());
        assert!(store.reverse_map.get("id_6").is_none());

        // Check vector content integrity (by comparing saved vectors with vectors at new indices)
        let vec_check_0 = store.get_vector_by_id("id_0")?;
        let vec_check_2 = store.get_vector_by_id("id_2")?;
        let vec_check_3 = store.get_vector_by_id("id_3")?;
        let vec_check_5 = store.get_vector_by_id("id_5")?;
        assert_eq!(
            i64::try_from(vec_at_idx_0.sum(Kind::Int64))?,
            i64::try_from(vec_check_0.sum(Kind::Int64))?
        );
        assert_eq!(
            i64::try_from(vec_at_idx_2.sum(Kind::Int64))?,
            i64::try_from(vec_check_2.sum(Kind::Int64))?
        );
        assert_eq!(
            i64::try_from(vec_at_idx_3.sum(Kind::Int64))?,
            i64::try_from(vec_check_3.sum(Kind::Int64))?
        );
        assert_eq!(
            i64::try_from(vec_at_idx_5.sum(Kind::Int64))?,
            i64::try_from(vec_check_5.sum(Kind::Int64))?
        );

        // Test deleting all vectors
        let all_ids: Vec<String> = store.ids.clone();
        delete_vectors(&mut store, &all_ids)?;
        assert!(store.is_empty());
        assert_eq!(store.dim(), 3); // Dim should persist
        assert_eq!(store.kind(), Kind::Int64); // Kind should persist
        assert_eq!(store.vectors.size(), &[0, 3]); // Check tensor is empty [0, D]

        Ok(())
    }

    #[test]
    fn test_delete_vectors_errors() -> Result<(), CrudError> {
        let mut store = create_test_store(5, 4, Kind::Float, 0)?; // id_0 to id_4
        let original_len = store.len();
        let original_map = store.get_reverse_map();

        // ID not found
        let ids_bad = vec!["id_1".to_string(), "id_99".to_string()]; // id_99 doesn't exist
        assert!(
            matches!(delete_vectors(&mut store, &ids_bad), Err(CrudError::IdNotFound(id)) if id == "id_99")
        );

        // Check store state wasn't mutated on error
        assert_eq!(store.len(), original_len);
        assert_eq!(store.reverse_map, original_map); // Maps should be identical

        // Deleting from empty store
        let mut empty_store = create_store(
            Tensor::zeros(&[0, 4], (Kind::Float, Device::Cpu)),
            Vec::new(),
        )?;
        let ids_del_empty = vec!["id_0".to_string()];
        assert!(
            matches!(delete_vectors(&mut empty_store, &ids_del_empty), Err(CrudError::IdNotFound(id)) if id == "id_0")
        );

        Ok(())
    }

    #[test]
    fn test_update_vectors_ok() -> Result<(), CrudError> {
        let mut store = create_test_store(5, 2, Kind::Float, 10)?; // id_10 to id_14
        let ids_to_update = vec!["id_11".to_string(), "id_13".to_string()];
        // Use Tensor::from_slice (corrected)
        let update_data = Tensor::from_slice(&[1.0f32, 1.1, 2.0, 2.2]).reshape(&[2, 2]);
        let original_vec_10 = store.get_vector_by_id("id_10")?.copy();
        let original_vec_12 = store.get_vector_by_id("id_12")?.copy();
        let original_map = store.get_reverse_map();

        update_vectors(&mut store, &ids_to_update, update_data)?;

        assert_eq!(store.len(), 5);
        // ... rest of assertions ...
        let updated_vec_11 = store.get_vector_by_id("id_11")?;
        let updated_vec_13 = store.get_vector_by_id("id_13")?;
        assert_abs_diff_eq!(f32::try_from(updated_vec_11.i(0))?, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(f32::try_from(updated_vec_11.i(1))?, 1.1, epsilon = 1e-6);
        assert_abs_diff_eq!(f32::try_from(updated_vec_13.i(0))?, 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(f32::try_from(updated_vec_13.i(1))?, 2.2, epsilon = 1e-6);
        let current_vec_10 = store.get_vector_by_id("id_10")?;
        let current_vec_12 = store.get_vector_by_id("id_12")?;
        assert_abs_diff_eq!(
            f32::try_from((original_vec_10 - current_vec_10).abs().sum(Kind::Float))?,
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            f32::try_from((original_vec_12 - current_vec_12).abs().sum(Kind::Float))?,
            0.0,
            epsilon = 1e-6
        );
        assert_eq!(store.reverse_map, original_map);

        Ok(())
    }

    #[test]
    fn test_update_vectors_errors() -> Result<(), CrudError> {
        let mut store = create_test_store(5, 4, Kind::Float, 0)?;
        let cpu = Device::Cpu;

        // ID not found
        let ids_bad_id = vec!["id_1".to_string(), "id_99".to_string()];
        let update_vecs = Tensor::randn(&[2, 4], (Kind::Float, cpu));
        // Save original vector state before potential modification attempt
        let original_vec_1 = store.get_vector_by_id("id_1")?.copy();
        assert!(
            matches!(update_vectors(&mut store, &ids_bad_id, update_vecs.copy()), Err(CrudError::IdNotFound(id)) if id == "id_99")
        );
        // Verify that id_1 was *not* updated despite being in the list before the erroring one
        let current_vec_1_after_err = store.get_vector_by_id("id_1")?;
        assert_abs_diff_eq!(
            f32::try_from(
                (original_vec_1 - current_vec_1_after_err)
                    .abs()
                    .sum(Kind::Float)
            )?,
            0.0,
            epsilon = 1e-6
        );

        // Dimension mismatch
        let ids_dim = vec!["id_1".to_string()];
        let update_vecs_dim = Tensor::randn(&[1, 5], (Kind::Float, cpu)); // Dim 5 != 4
        assert!(matches!(
            update_vectors(&mut store, &ids_dim, update_vecs_dim),
            Err(CrudError::DimensionMismatch { .. })
        ));

        // Kind mismatch
        let ids_kind = vec!["id_1".to_string()];
        let update_vecs_kind = Tensor::randn(&[1, 4], (Kind::Double, cpu)); // Double != Float
        assert!(matches!(
            update_vectors(&mut store, &ids_kind, update_vecs_kind),
            Err(CrudError::KindMismatch { .. })
        ));

        // Length mismatch (vectors vs IDs)
        let ids_len = vec!["id_1".to_string(), "id_2".to_string()];
        let update_vecs_len = Tensor::randn(&[1, 4], (Kind::Float, cpu)); // Only 1 vector for 2 IDs
        assert!(matches!(
            update_vectors(&mut store, &ids_len, update_vecs_len),
            Err(CrudError::DimensionMismatch { .. })
        ));

        Ok(())
    }

    // --- Tests for get_vector_by_id ---
    #[test]
    fn test_get_vector_by_id() -> Result<(), CrudError> {
        let store = create_test_store(5, 3, Kind::Float, 0)?;
        let vec0 = store.get_vector_by_id("id_0")?;
        let vec4 = store.get_vector_by_id("id_4")?;

        assert_eq!(vec0.size(), &[3]);
        assert_eq!(vec4.size(), &[3]);
        assert_eq!(vec0.kind(), Kind::Float);

        // Check non-existent ID
        assert!(
            matches!(store.get_vector_by_id("id_99"), Err(CrudError::IdNotFound(id)) if id == "id_99")
        );

        Ok(())
    }
    #[test]
    fn test_get_vector_from_empty_store() -> Result<(), CrudError> {
        let empty_store = create_store(Tensor::zeros(&[0, 5], (Kind::Float, Device::Cpu)), vec![])?;
        assert!(
            matches!(empty_store.get_vector_by_id("id_0"), Err(CrudError::IdNotFound(id)) if id == "id_0")
        );
        Ok(())
    }
}
