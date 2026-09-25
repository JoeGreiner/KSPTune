#pragma once

#include <petscksp.h>
#include <string>
#include <vector>

namespace ksptune {
struct snapshot {
  std::string snapshot_id;
  int solve_index = 0;
  std::string matrix_file_path;
  std::string right_hand_side_file_path;
  std::string initial_guess_file_path;
  std::string metadata_file_path;
  PetscInt rows = 0;
  PetscInt cols = 0;
  PetscInt right_hand_side_size = 0;
  int mpi_size = 1;
  bool symmetric = false;
  bool spd = false;
  bool nullspace = false;
  bool matrix_nullspace_attached = false;
  bool transpose_nullspace_attached = false;
  bool near_nullspace_attached = false;
  std::string nullspace_kind = "none";
  int nullspace_field_index = -1;
  int nullspace_block_size = 0;
  std::string rhs_nullspace_component_removed = "unknown";
  double relative_tolerance = -1.0;
  double absolute_tolerance = -1.0;
  double divergence_tolerance = -1.0;
  int max_iterations = -1;
  std::vector<PetscInt> row_ownership_ranges;
  std::vector<PetscInt> rhs_ownership_ranges;
  std::vector<PetscInt> x0_ownership_ranges;
};

std::string trim(const std::string& text);
bool file_exists(const std::string& path);
double parse_double_or_default(const std::string& text, double default_value);
std::vector<snapshot> read_snapshot_collection(const std::string& path);
PetscErrorCode select_snapshot(std::vector<snapshot>& snapshots, int solve_index,
                               const std::string& snapshot_id);
PetscErrorCode load_matrix_binary(const snapshot& snap, Mat* matrix);
PetscErrorCode load_vector_binary(const std::string& path,
                                  const std::vector<PetscInt>& ownership_ranges,
                                  PetscInt global_size, Vec* vector);
}  // namespace ksptune
