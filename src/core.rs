use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::ops::Mul;
use std::path::Path;
use tch::Tensor;
use tokio::task;

/// Our in‑memory store.
pub struct Store {
    /// The hypervector array as a tch::Tensor.
    pub vectors: Tensor,
    /// Maps row index to an ID string.
    pub index_to_id: HashMap<usize, String>,
    /// Reverse mapping.
    pub id_to_index: HashMap<String, usize>,
}

/// Enum returned by the search functions.
#[derive(Debug)]
pub enum SearchResult {
    /// Returned when no threshold is provided: for every vector in the store, returns (index, id, similarity score).
    WithScores(Vec<(usize, String, f64)>),
    /// Returned when a threshold is provided: returns only (index, id) pairs with similarity ≥ threshold.
    WithoutScores(Vec<(usize, String)>),
}

/// Helper structure for serialization.
#[derive(Serialize, Deserialize)]
struct SerializableStore {
    data: Vec<f32>,
    shape: Vec<i64>,
    index_to_id: HashMap<usize, String>,
    id_to_index: HashMap<String, usize>,
}

impl Store {
    /// Converts a Store into a serializable representation.
    fn to_serializable(&self) -> Result<SerializableStore, Box<dyn Error + Send + Sync>> {
        let shape = self.vectors.size();
        let numel: i64 = shape.iter().product();
        let mut data = vec![0.0f32; numel as usize];
        self.vectors.copy_data(&mut data, numel as usize);
        Ok(SerializableStore {
            data,
            shape,
            index_to_id: self.index_to_id.clone(),
            id_to_index: self.id_to_index.clone(),
        })
    }

    /// Reconstructs a Store from a SerializableStore.
    fn from_serializable(s: SerializableStore) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let tensor = Tensor::f_from_slice(&s.data)?.reshape(&s.shape);
        Ok(Store {
            vectors: tensor,
            index_to_id: s.index_to_id,
            id_to_index: s.id_to_index,
        })
    }

    /// Adds new vectors given as (id, vector) pairs.
    /// If an id already exists, a warning is printed and that pair is skipped.
    pub fn add_vectors(
        &mut self,
        data: &[(String, Tensor)],
        store_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let current_rows = self.vectors.size()[0];
        let mut new_vecs: Vec<Tensor> = Vec::new();
        let mut new_ids: Vec<String> = Vec::new();
        for (id, vec) in data {
            if self.id_to_index.contains_key(id) {
                eprintln!(
                    "Warning: id '{}' already exists in store '{}'. Skipping.",
                    id, store_name
                );
            } else {
                // Ensure the vector is 2D: if it's 1D, unsqueeze it.
                let vec_2d = if vec.dim() == 1 {
                    vec.unsqueeze(0)
                } else {
                    vec.shallow_clone()
                };
                new_vecs.push(vec_2d);
                new_ids.push(id.clone());
            }
        }
        if !new_vecs.is_empty() {
            let new_tensor = Tensor::cat(&new_vecs, 0);
            self.vectors = Tensor::cat(&[&self.vectors, &new_tensor], 0);
            for (i, id) in new_ids.into_iter().enumerate() {
                let new_index = (current_rows + i as i64) as usize;
                self.index_to_id.insert(new_index, id.clone());
                self.id_to_index.insert(id, new_index);
            }
        }
        Ok(())
    }

    /// Updates existing vectors in the store.
    /// For each (id, vector) pair, if the id exists, update that row in the tensor.
    pub fn update_vectors(
        &mut self,
        data: &[(String, Tensor)],
        store_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let total = self.vectors.size()[0];
        let mut rows: Vec<Tensor> = Vec::with_capacity(total as usize);
        for i in 0..total {
            let current_index = i as usize;
            let id = self.index_to_id.get(&current_index).unwrap();
            // Get existing row as 2D.
            let existing_row = self.vectors.get(i).unsqueeze(0);
            if let Some((_, new_vec)) = data.iter().find(|(upd_id, _)| upd_id == id) {
                let candidate = if new_vec.dim() == 1 {
                    new_vec.unsqueeze(0)
                } else {
                    new_vec.shallow_clone()
                };
                // Compare dimensions in 2D form.
                if candidate.size() != existing_row.size() {
                    eprintln!("Warning: Dimension mismatch for id '{}' in store '{}'. Skipping update for this vector.", id, store_name);
                    rows.push(existing_row);
                } else {
                    rows.push(candidate);
                }
            } else {
                rows.push(existing_row);
            }
        }
        self.vectors = Tensor::cat(&rows, 0);
        Ok(())
    }

    /// Deletes vectors corresponding to the provided ids.
    /// Updates the in‑memory tensor and the hash maps.
    pub fn delete_vectors(
        &mut self,
        ids: &[String],
        store_name: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use std::collections::HashSet;
        let delete_set: HashSet<_> = ids.iter().cloned().collect();
        let total = self.vectors.size()[0];
        let mut new_rows: Vec<Tensor> = Vec::new();
        let mut new_index_to_id: HashMap<usize, String> = HashMap::new();
        let mut new_id_to_index: HashMap<String, usize> = HashMap::new();
        let mut new_index = 0;
        for i in 0..total {
            let idx = i as usize;
            if let Some(id) = self.index_to_id.get(&idx) {
                if delete_set.contains(id) {
                    eprintln!("Deleting id '{}' from store '{}'.", id, store_name);
                    continue;
                }
                // Ensure we keep a 2D row.
                new_rows.push(self.vectors.get(i).unsqueeze(0));
                new_index_to_id.insert(new_index, id.clone());
                new_id_to_index.insert(id.clone(), new_index);
                new_index += 1;
            }
        }
        if new_rows.is_empty() {
            let dim = self.vectors.size();
            self.vectors = Tensor::zeros(&[0, dim[1]], tch::kind::FLOAT_CPU);
        } else {
            self.vectors = Tensor::cat(&new_rows, 0);
        }
        self.index_to_id = new_index_to_id;
        self.id_to_index = new_id_to_index;
        Ok(())
    }

    /// Gets a vector from the store.
    /// Either an id (as &str) or an index (as usize) must be supplied—but not both.
    pub fn get_vector(
        &self,
        id: Option<&str>,
        index: Option<usize>,
    ) -> Result<Tensor, Box<dyn std::error::Error + Send + Sync>> {
        match (id, index) {
            (Some(id_str), None) => {
                if let Some(&idx) = self.id_to_index.get(id_str) {
                    Ok(self.vectors.get(idx as i64))
                } else {
                    Err(format!("ID '{}' not found", id_str).into())
                }
            }
            (None, Some(idx)) => {
                if idx < self.vectors.size()[0] as usize {
                    Ok(self.vectors.get(idx as i64))
                } else {
                    Err(format!("Index {} out of bounds", idx).into())
                }
            }
            _ => Err("Either id or index must be supplied, but not both".into()),
        }
    }

    pub fn cosine_search(
        &self,
        v1: &Tensor,
        thresh: Option<f64>,
    ) -> Result<SearchResult, Box<dyn std::error::Error + Send + Sync>> {
        // Ensure v1 is a 1D tensor.
        let v1 = if v1.dim() == 1 {
            v1.shallow_clone()
        } else {
            v1.squeeze()
        };
        let v1_norm = v1.norm();
        let batch_size = 32;
        let total = self.vectors.size()[0] as usize;
        let mut results_with_scores = Vec::new();
        let mut results_filtered = Vec::new();

        for start in (0..total).step_by(batch_size) {
            let end = std::cmp::min(start + batch_size, total);
            // Extract a batch of vectors: shape [batch, dim]
            let batch = self.vectors.narrow(0, start as i64, (end - start) as i64);
            // Compute dot product between each row and v1.
            let dot = batch.shallow_clone().mul(&v1.unsqueeze(0)).sum_dim_intlist(
                1,
                false,
                tch::Kind::Float,
            );
            // Unsqueeze dot to shape [batch,1] so that division is elementwise.
            let dot = dot.unsqueeze(1);
            // Compute L2 norm along dim=1, keeping the dimension: shape [batch,1]
            let batch_norm = batch.linalg_norm(2, 1, true, tch::Kind::Float);
            // Compute similarity: sim = dot / (norm * v1_norm), resulting in [batch,1]
            let sim = (dot / (batch_norm * &v1_norm)).flatten(0, -1);
            let sim_vec: Vec<f64> = sim.iter::<f64>()?.collect();

            for (i, &score) in sim_vec.iter().enumerate() {
                let global_idx = start + i;
                let id = self.index_to_id.get(&global_idx).unwrap().clone();
                results_with_scores.push((global_idx, id.clone(), score));
                if let Some(th) = thresh {
                    if score >= th {
                        results_filtered.push((global_idx, id));
                    }
                }
            }
        }
        if thresh.is_none() {
            Ok(SearchResult::WithScores(results_with_scores))
        } else {
            Ok(SearchResult::WithoutScores(results_filtered))
        }
    }

    /// Computes dot similarity between v1 and each vector in the store, processing in batches of 32.
    /// If no threshold is provided, returns (index, id, dot score) for every vector.
    /// Otherwise returns only (index, id) pairs for which the dot product is ≥ threshold.
    pub fn dot_search(
        &self,
        v1: &Tensor,
        thresh: Option<f64>,
    ) -> Result<SearchResult, Box<dyn std::error::Error + Send + Sync>> {
        let v1 = if v1.dim() == 1 {
            v1.shallow_clone()
        } else {
            v1.squeeze()
        };
        let batch_size = 32;
        let total = self.vectors.size()[0] as usize;
        let mut results_with_scores = Vec::new();
        let mut results_filtered = Vec::new();

        for start in (0..total).step_by(batch_size) {
            let end = std::cmp::min(start + batch_size, total);
            let batch = self.vectors.narrow(0, start as i64, (end - start) as i64);
            let dot = batch.shallow_clone().mul(&v1.unsqueeze(0)).sum_dim_intlist(
                1,
                false,
                tch::Kind::Float,
            );
            let dot_vec: Vec<f64> = dot.iter::<f64>()?.collect();
            for (i, &score) in dot_vec.iter().enumerate() {
                let global_idx = start + i;
                let id = self.index_to_id.get(&global_idx).unwrap().clone();
                results_with_scores.push((global_idx, id.clone(), score));
                if let Some(th) = thresh {
                    if score >= th {
                        results_filtered.push((global_idx, id));
                    }
                }
            }
        }
        if thresh.is_none() {
            Ok(SearchResult::WithScores(results_with_scores))
        } else {
            Ok(SearchResult::WithoutScores(results_filtered))
        }
    }
}

/// Creates a new store from the given tensor.
pub fn create_store(vectors: Tensor) -> Store {
    let num_rows = vectors.size()[0] as usize;
    let mut index_to_id = HashMap::with_capacity(num_rows);
    let mut id_to_index = HashMap::with_capacity(num_rows);
    for i in 0..num_rows {
        let id = format!("vec_{}", i);
        index_to_id.insert(i, id.clone());
        id_to_index.insert(id, i);
    }
    Store {
        vectors,
        index_to_id,
        id_to_index,
    }
}

/// Dumps the store to disk asynchronously using bincode.
/// The serialized data is written to "dir_path/name.bin".
pub async fn dump_store(
    dir_path: &str,
    name: &str,
    store: &Store,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let dir = Path::new(dir_path);
    let file_path = dir.join(format!("{}.bin", name));
    let serializable = store.to_serializable()?;
    task::spawn_blocking(move || -> Result<(), Box<dyn Error + Send + Sync>> {
        let encoded = bincode::serialize(&serializable)?;
        let mut file = File::create(&file_path)?;
        file.write_all(&encoded)?;
        Ok(())
    })
    .await??;
    Ok(())
}

/// Loads the store from disk by reading "dir_path/name.bin" and deserializing via bincode.
pub fn load_store(dir_path: &str, name: &str) -> Result<Store, Box<dyn Error + Send + Sync>> {
    let dir = Path::new(dir_path);
    let file_path = dir.join(format!("{}.bin", name));
    let mut file = File::open(&file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let serializable: SerializableStore = bincode::deserialize(&buffer)?;
    Store::from_serializable(serializable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use tch::Tensor;

    #[test]
    fn test_create_store() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::f_from_slice(&data).unwrap().reshape(&[2, 3]);
        let store = create_store(tensor);
        assert_eq!(store.index_to_id.len(), 2);
        assert_eq!(store.id_to_index.len(), 2);
        assert_eq!(store.index_to_id.get(&0).unwrap(), "vec_0");
        assert_eq!(store.index_to_id.get(&1).unwrap(), "vec_1");
        assert_eq!(*store.id_to_index.get("vec_0").unwrap(), 0);
        assert_eq!(*store.id_to_index.get("vec_1").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_dump_and_load_store() {
        let data = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let tensor = Tensor::f_from_slice(&data).unwrap().reshape(&[2, 3]);
        let store = create_store(tensor.shallow_clone());

        let mut temp_dir: PathBuf = env::temp_dir();
        temp_dir.push("vectordb_test");
        fs::create_dir_all(&temp_dir).expect("Failed to create temporary directory");
        let dir_path = temp_dir.to_str().unwrap();
        let name = "test_store";

        dump_store(dir_path, name, &store)
            .await
            .expect("Dump failed");

        let loaded_store = load_store(dir_path, name).expect("Load failed");

        let diff = &store.vectors - &loaded_store.vectors;
        let max_diff = diff.abs().max().double_value(&[]);
        assert!(
            max_diff < 1e-6,
            "Tensor values differ after dump and load: {}",
            max_diff
        );
        assert_eq!(store.index_to_id, loaded_store.index_to_id);
        assert_eq!(store.id_to_index, loaded_store.id_to_index);

        let file_path = temp_dir.join(format!("{}.bin", name));
        fs::remove_file(file_path).expect("Failed to remove file");
        fs::remove_dir(&temp_dir).expect("Failed to remove temporary directory");
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_add_vectors() {
        // Create an initial store with 2 vectors.
        let vec1 = Tensor::from_slice(&[0.1_f32, 0.2, 0.3]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[0.4_f32, 0.5, 0.6]).unsqueeze(0);
        let initial_tensor = Tensor::cat(&[&vec1, &vec2], 0);
        let mut store = create_store(initial_tensor);

        // Prepare new vectors to add.
        let new_vec1 = Tensor::from_slice(&[0.7_f32, 0.8, 0.9]);
        let new_vec2 = Tensor::from_slice(&[1.0_f32, 1.1, 1.2]);
        let data = vec![
            ("id_new1".to_string(), new_vec1),
            // Duplicate id; should be skipped.
            ("vec_0".to_string(), new_vec2.shallow_clone()),
        ];
        store.add_vectors(&data, "test_store").unwrap();

        // After adding, store should have 3 rows.
        assert_eq!(store.vectors.size()[0], 3);

        // The new vector should be at index 2.
        let row = store.vectors.get(2).unsqueeze(0);
        let diff = row - data[0].1.unsqueeze(0);
        let max_diff = diff.abs().max().double_value(&[]);
        assert!(max_diff < 1e-6);

        // Confirm that the duplicate was not added.
        assert!(store.id_to_index.get("vec_0").unwrap() < &3);
    }

    #[test]
    fn test_update_vectors() {
        // Create an initial store with 2 vectors.
        let vec1 = Tensor::from_slice(&[0.1_f32, 0.2, 0.3]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[0.4_f32, 0.5, 0.6]).unsqueeze(0);
        let initial_tensor = Tensor::cat(&[&vec1, &vec2], 0);
        let mut store = create_store(initial_tensor);

        // Update vector for id "vec_1".
        let updated = Tensor::from_slice(&[0.7_f32, 0.8, 0.9]);
        let updated_for_update = updated.shallow_clone(); // clone for update data
        let data = vec![("vec_1".to_string(), updated_for_update)];
        store.update_vectors(&data, "test_store").unwrap();

        // Verify that the vector at index 1 has been updated.
        let row = store.vectors.get(1).unsqueeze(0);
        let diff = row - updated.unsqueeze(0);
        let max_diff = diff.abs().max().double_value(&[]);
        assert!(max_diff < 1e-6);
    }

    #[test]
    fn test_delete_vectors() {
        // Create an initial store with 3 vectors.
        let vec1 = Tensor::from_slice(&[0.1_f32, 0.2, 0.3]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[0.4_f32, 0.5, 0.6]).unsqueeze(0);
        let vec3 = Tensor::from_slice(&[0.7_f32, 0.8, 0.9]).unsqueeze(0);
        let initial_tensor = Tensor::cat(&[&vec1, &vec2, &vec3], 0);
        let mut store = create_store(initial_tensor);

        // Delete vector with id "vec_1".
        let ids = vec!["vec_1".to_string()];
        store.delete_vectors(&ids, "test_store").unwrap();

        // Now the store should have 2 rows.
        assert_eq!(store.vectors.size()[0], 2);
        // Confirm that "vec_0" and "vec_2" remain and "vec_1" is gone.
        assert!(store.id_to_index.contains_key("vec_0"));
        assert!(store.id_to_index.contains_key("vec_2"));
        assert!(!store.id_to_index.contains_key("vec_1"));
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_get_vector_by_id() {
        let vec1 = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[4.0_f32, 5.0, 6.0]).unsqueeze(0);
        let tensor = Tensor::cat(&[&vec1, &vec2], 0);
        let store = create_store(tensor);
        let v = store.get_vector(Some("vec_1"), None).unwrap();
        let diff = v - vec2;
        let max_diff = diff.abs().max().double_value(&[]);
        assert!(max_diff < 1e-6);
    }

    #[test]
    fn test_get_vector_by_index() {
        let vec1 = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[4.0_f32, 5.0, 6.0]).unsqueeze(0);
        let tensor = Tensor::cat(&[&vec1, &vec2], 0);
        let store = create_store(tensor);
        let v = store.get_vector(None, Some(0)).unwrap();
        let diff = v - vec1;
        let max_diff = diff.abs().max().double_value(&[]);
        assert!(max_diff < 1e-6);
    }

    #[test]
    fn test_cosine_search_with_scores() {
        // Create store with three orthogonal vectors.
        let vec1 = Tensor::from_slice(&[1.0_f32, 0.0, 0.0]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[0.0_f32, 1.0, 0.0]).unsqueeze(0);
        let vec3 = Tensor::from_slice(&[0.0_f32, 0.0, 1.0]).unsqueeze(0);
        let tensor = Tensor::cat(&[&vec1, &vec2, &vec3], 0);
        let store = create_store(tensor);
        // Query with a vector similar to vec1.
        let query = Tensor::from_slice(&[0.9_f32, 0.1, 0.0]);
        let res = store.cosine_search(&query, None).unwrap();
        if let SearchResult::WithScores(results) = res {
            let (_idx, id, score) = results
                .iter()
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                .unwrap();
            assert_eq!(id, "vec_0");
            assert!(*score > 0.9);
        } else {
            panic!("Expected WithScores result");
        }
    }

    #[test]
    fn test_cosine_search_filtered() {
        let vec1 = Tensor::from_slice(&[1.0_f32, 0.0, 0.0]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[0.0_f32, 1.0, 0.0]).unsqueeze(0);
        let vec3 = Tensor::from_slice(&[0.0_f32, 0.0, 1.0]).unsqueeze(0);
        let tensor = Tensor::cat(&[&vec1, &vec2, &vec3], 0);
        let store = create_store(tensor);
        let query = Tensor::from_slice(&[0.9_f32, 0.1, 0.0]);
        let res = store.cosine_search(&query, Some(0.95)).unwrap();
        if let SearchResult::WithoutScores(results) = res {
            assert_eq!(results.len(), 1);
            let (_idx, id) = &results[0];
            assert_eq!(id, "vec_0");
        } else {
            panic!("Expected WithoutScores result");
        }
    }

    #[test]
    fn test_dot_search_with_scores() {
        let vec1 = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[4.0_f32, 5.0, 6.0]).unsqueeze(0);
        let vec3 = Tensor::from_slice(&[7.0_f32, 8.0, 9.0]).unsqueeze(0);
        let tensor = Tensor::cat(&[&vec1, &vec2, &vec3], 0);
        let store = create_store(tensor);
        let query = Tensor::from_slice(&[1.0_f32, 0.0, 0.0]);
        let res = store.dot_search(&query, None).unwrap();
        if let SearchResult::WithScores(results) = res {
            let (_idx, id, _score) = results
                .iter()
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                .unwrap();
            // Expect the best match to be the vector with the highest dot product (vec_2).
            assert_eq!(id, "vec_2");
        } else {
            panic!("Expected WithScores result");
        }
    }

    #[test]
    fn test_dot_search_filtered() {
        let vec1 = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]).unsqueeze(0);
        let vec2 = Tensor::from_slice(&[4.0_f32, 5.0, 6.0]).unsqueeze(0);
        let vec3 = Tensor::from_slice(&[7.0_f32, 8.0, 9.0]).unsqueeze(0);
        let tensor = Tensor::cat(&[&vec1, &vec2, &vec3], 0);
        let store = create_store(tensor);
        let query = Tensor::from_slice(&[1.0_f32, 0.0, 0.0]);
        // Set a threshold that only the best match qualifies. In this case, threshold = 5.0 should yield only vec_2.
        let res = store.dot_search(&query, Some(5.0)).unwrap();
        if let SearchResult::WithoutScores(results) = res {
            assert_eq!(results.len(), 1);
            let (_idx, id) = &results[0];
            assert_eq!(id, "vec_2");
        } else {
            panic!("Expected WithoutScores result");
        }
    }
}
