#include "replay_cache.hpp"

namespace ksptune {

double matrix_memory_mb(Mat matrix)
{
  MatInfo info;
  if(MatGetInfo(matrix, MAT_GLOBAL_SUM, &info)) return 0.0;
  if(info.memory > 0.0) return info.memory / (1024.0 * 1024.0);
  PetscInt rows = 0, cols = 0;
  if(MatGetSize(matrix, &rows, &cols)) return 0.0;
  return ((rows + 1) * double(sizeof(PetscInt)) +
          info.nz_allocated * (sizeof(PetscScalar) + sizeof(PetscInt))) / (1024.0 * 1024.0);
}

PetscErrorCode replay_cache::load_matrix(const snapshot& snap, const std::string& key,
                                        Mat& matrix, bool& hit)
{
  auto& entry = matrices[key];
  hit = entry.object.value != nullptr;
  if(!hit) {
    petsc_matrix loaded;
    PetscCall(load_matrix_binary(snap, loaded.ptr()));
    entry.object.value = loaded.release();
    entry.memory_mb = ksptune::matrix_memory_mb(entry.object);
  }
  entry.last_used = ++clock;
  matrix = entry.object;
  return 0;
}

PetscErrorCode replay_cache::load_vector_copy(const std::string& path,
                                             const std::vector<PetscInt>& ownership,
                                             PetscInt global_size, bool use_cache, Vec* vector)
{
  if(!use_cache) return load_vector_binary(path, ownership, global_size, vector);
  auto& entry = vectors[path + "\nglobal_size=" + std::to_string(global_size)];
  if(!entry.object.value) {
    petsc_vector loaded;
    PetscCall(load_vector_binary(path, ownership, global_size, loaded.ptr()));
    entry.object.value = loaded.release();
    PetscInt local_size = 0;
    PetscCall(VecGetLocalSize(entry.object, &local_size));
    double local_bytes = local_size * double(sizeof(PetscScalar));
    double global_bytes = 0.0;
    PetscCallMPI(MPI_Allreduce(&local_bytes, &global_bytes, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD));
    entry.memory_mb = global_bytes / (1024.0 * 1024.0);
  }
  entry.last_used = ++clock;
  PetscCall(VecDuplicate(entry.object, vector));
  PetscCall(VecCopy(entry.object, *vector));
  return 0;
}

double replay_cache::matrix_memory_mb() const
{
  double memory = 0.0;
  for(const auto& item : matrices) memory += item.second.memory_mb;
  return memory;
}

double replay_cache::vector_memory_mb() const
{
  double memory = 0.0;
  for(const auto& item : vectors) memory += item.second.memory_mb;
  return memory;
}

PetscErrorCode replay_cache::clear_ksps()
{
  for(auto& item : ksps) PetscCall(item.second.ksp.reset());
  ksps.clear();
  return 0;
}

PetscErrorCode replay_cache::clear()
{
  PetscCall(clear_ksps());
  for(auto& item : matrices) PetscCall(item.second.object.reset());
  matrices.clear();
  for(auto& item : vectors) PetscCall(item.second.object.reset());
  vectors.clear();
  return 0;
}

PetscErrorCode replay_cache::evict(double memory_limit_mb)
{
  if(memory_limit_mb <= 0.0) return 0;
  while(memory_mb() > memory_limit_mb && (!matrices.empty() || !vectors.empty())) {
    auto matrix = matrices.end();
    auto vector = vectors.end();
    unsigned long oldest = 0;
    for(auto item = matrices.begin(); item != matrices.end(); ++item) {
      if(matrix == matrices.end() || item->second.last_used < oldest) {
        matrix = item;
        oldest = item->second.last_used;
      }
    }
    for(auto item = vectors.begin(); item != vectors.end(); ++item) {
      if((matrix == matrices.end() && vector == vectors.end()) || item->second.last_used < oldest) {
        vector = item;
        oldest = item->second.last_used;
      }
    }
    if(vector != vectors.end()) {
      PetscCall(vector->second.object.reset());
      vectors.erase(vector);
    } else {
      for(auto solver = ksps.begin(); solver != ksps.end();) {
        if(solver->second.matrix_cache_key == matrix->first) {
          PetscCall(solver->second.ksp.reset());
          solver = ksps.erase(solver);
        } else {
          ++solver;
        }
      }
      PetscCall(matrix->second.object.reset());
      matrices.erase(matrix);
    }
    ++evictions;
  }
  return 0;
}

}  // namespace ksptune
