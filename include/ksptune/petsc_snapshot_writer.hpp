#pragma once

#include <petscmat.h>
#include <petscvec.h>

#include <string>
#include <utility>
#include <vector>

namespace ksptune {
namespace petsc_snapshot {

struct KspSnapshotNullspaceMetadata {
  std::string kind;
  int field_index = -1;
  int block_size = 0;
  std::string right_hand_side_component_removed = "unknown";
};

struct KspSnapshotRequest {
  MPI_Comm mpi_communicator = PETSC_COMM_WORLD;
  std::string output_directory = ".";
  std::string snapshot_file_prefix = "ksp";
  int snapshot_index = 0;
  Mat system_matrix = nullptr;
  Vec right_hand_side_vector = nullptr;
  Vec initial_guess_vector = nullptr;
  KspSnapshotNullspaceMetadata nullspace_metadata;
  std::vector<std::pair<std::string, std::string>> extra_metadata_entries;
};

struct PetscKspSnapshotExport {
  MPI_Comm mpi_communicator = PETSC_COMM_WORLD;
  std::string snapshot_directory;
  std::string solve_indices = "0";
  std::string snapshot_file_prefix = "ksp";
  Mat system_matrix = nullptr;
  Vec right_hand_side_vector = nullptr;
  Vec initial_guess_vector = nullptr;
  KspSnapshotNullspaceMetadata nullspace_metadata;
  std::vector<std::pair<std::string, std::string>> metadata_entries;
};

struct KspSnapshotResult {
  std::string matrix_file_path;
  std::string right_hand_side_file_path;
  std::string initial_guess_file_path;
  std::string metadata_file_path;
};

std::string SafeSnapshotFilePrefix(const std::string& descriptive_name);

std::string SafeSnapshotFilePrefixFromPath(const std::string& path);

bool ShouldWriteSnapshotForStep(const std::string& selected_steps, int snapshot_index);

PetscErrorCode WriteKspSnapshot(const KspSnapshotRequest& request, KspSnapshotResult* result);

PetscErrorCode WritePetscKspSnapshotIfRequested(
    const PetscKspSnapshotExport& snapshot_export,
    KspSnapshotResult* result);

}  // namespace petsc_snapshot
}  // namespace ksptune
