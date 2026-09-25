#pragma once

#include "snapshot_io.hpp"
#include <petscksp.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ksptune {

template<class T, PetscErrorCode (*Destroy)(T*)>
struct petsc_handle {
  T value = nullptr;
  petsc_handle() = default;
  petsc_handle(const petsc_handle&) = delete;
  petsc_handle& operator=(const petsc_handle&) = delete;
  ~petsc_handle() { reset(); }
  operator T() const { return value; }
  T* ptr() { return &value; }
  T release() { return std::exchange(value, nullptr); }
  PetscErrorCode reset() { return value ? Destroy(&value) : 0; }
};

using petsc_matrix = petsc_handle<Mat, MatDestroy>;
using petsc_vector = petsc_handle<Vec, VecDestroy>;
using petsc_ksp = petsc_handle<KSP, KSPDestroy>;
using petsc_nullspace = petsc_handle<MatNullSpace, MatNullSpaceDestroy>;

struct ksp_setup_context {
  double setup_time_sec = 0.0;
  std::string matrix_cache_key;
  std::vector<PetscReal> residual_history;
  petsc_ksp ksp;
};

template<class Handle>
struct cache_entry {
  Handle object;
  double memory_mb = 0.0;
  unsigned long last_used = 0;
};

double matrix_memory_mb(Mat matrix);

struct replay_cache {
  std::map<std::string, cache_entry<petsc_matrix>> matrices;
  std::map<std::string, cache_entry<petsc_vector>> vectors;
  // Destroy solvers before the matrices they reference.
  std::map<std::string, ksp_setup_context> ksps;
  unsigned long clock = 0;
  int evictions = 0;

  PetscErrorCode load_matrix(const snapshot& snap, const std::string& key, Mat& matrix, bool& hit);
  PetscErrorCode load_vector_copy(const std::string& path, const std::vector<PetscInt>& ownership,
                                  PetscInt global_size, bool use_cache, Vec* vector);
  double matrix_memory_mb() const;
  double vector_memory_mb() const;
  double memory_mb() const { return matrix_memory_mb() + vector_memory_mb(); }
  PetscErrorCode clear_ksps();
  PetscErrorCode clear();
  PetscErrorCode evict(double memory_limit_mb);
};

}  // namespace ksptune
