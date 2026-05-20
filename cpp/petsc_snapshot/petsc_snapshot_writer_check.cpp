#include "ksptune/petsc_snapshot_writer.hpp"

#include <petscksp.h>

#include <iostream>
#include <string>
#include <utility>

namespace {

PetscErrorCode check_step_selection()
{
  if(!ksptune::petsc_snapshot::ShouldWriteSnapshotForStep("", 0)) return PETSC_ERR_PLIB;
  if(!ksptune::petsc_snapshot::ShouldWriteSnapshotForStep("all", 7)) return PETSC_ERR_PLIB;
  if(!ksptune::petsc_snapshot::ShouldWriteSnapshotForStep("*", 3)) return PETSC_ERR_PLIB;
  if(!ksptune::petsc_snapshot::ShouldWriteSnapshotForStep("0, 2, 5", 2)) {
    return PETSC_ERR_PLIB;
  }
  if(ksptune::petsc_snapshot::ShouldWriteSnapshotForStep("0, 2, 5", 4)) return PETSC_ERR_PLIB;
  return 0;
}

PetscErrorCode check_safe_snapshot_file_prefix()
{
  using ksptune::petsc_snapshot::SafeSnapshotFilePrefix;

  if(SafeSnapshotFilePrefix("elliptic PDE (PETSc)") != "elliptic_pde_petsc") {
    return PETSC_ERR_PLIB;
  }
  if(SafeSnapshotFilePrefix("  parab//solver  ") != "parab_solver") {
    return PETSC_ERR_PLIB;
  }
  if(SafeSnapshotFilePrefix("***") != "ksp") return PETSC_ERR_PLIB;
  if(ksptune::petsc_snapshot::SafeSnapshotFilePrefixFromPath("meshes/rmtest-6x5x4-trimmed-veri-nosmall") !=
     "rmtest_6x5x4_trimmed_veri_nosmall") {
    return PETSC_ERR_PLIB;
  }
  return 0;
}

PetscErrorCode create_test_system(MPI_Comm mpi_communicator,
                                  Mat* system_matrix,
                                  Vec* right_hand_side_vector,
                                  Vec* initial_guess_vector)
{
  constexpr PetscInt global_size = 12;
  PetscCall(MatCreateAIJ(mpi_communicator,
                         PETSC_DECIDE,
                         PETSC_DECIDE,
                         global_size,
                         global_size,
                         3,
                         nullptr,
                         3,
                         nullptr,
                         system_matrix));
  PetscInt row_start = 0;
  PetscInt row_end = 0;
  PetscCall(MatGetOwnershipRange(*system_matrix, &row_start, &row_end));
  for(PetscInt row = row_start; row < row_end; ++row) {
    const PetscScalar diagonal = 4.0;
    PetscCall(MatSetValue(*system_matrix, row, row, diagonal, INSERT_VALUES));
    if(row > 0) {
      const PetscScalar off_diagonal = -1.0;
      PetscCall(MatSetValue(*system_matrix, row, row - 1, off_diagonal, INSERT_VALUES));
    }
    if(row + 1 < global_size) {
      const PetscScalar off_diagonal = -1.0;
      PetscCall(MatSetValue(*system_matrix, row, row + 1, off_diagonal, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(*system_matrix, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*system_matrix, MAT_FINAL_ASSEMBLY));

  PetscCall(VecCreateMPI(
      mpi_communicator,
      PETSC_DECIDE,
      global_size,
      right_hand_side_vector));
  PetscCall(VecDuplicate(*right_hand_side_vector, initial_guess_vector));
  PetscCall(VecSet(*right_hand_side_vector, 1.0));
  PetscCall(VecSet(*initial_guess_vector, 0.0));
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscCall(check_step_selection());
  PetscCall(check_safe_snapshot_file_prefix());

  char output_dir[PETSC_MAX_PATH_LEN] = ".";
  PetscBool found_output_dir = PETSC_FALSE;
  PetscCall(PetscOptionsGetString(nullptr,
                                  nullptr,
                                  "-snapshot_output_dir",
                                  output_dir,
                                  sizeof(output_dir),
                                  &found_output_dir));
  const std::string output_directory = found_output_dir ? output_dir : ".";

  Mat system_matrix = nullptr;
  Vec right_hand_side_vector = nullptr;
  Vec initial_guess_vector = nullptr;
  PetscCall(create_test_system(
      PETSC_COMM_WORLD,
      &system_matrix,
      &right_hand_side_vector,
      &initial_guess_vector));

  ksptune::petsc_snapshot::KspSnapshotRequest first;
  first.mpi_communicator = PETSC_COMM_WORLD;
  first.output_directory = output_directory;
  first.snapshot_file_prefix =
      ksptune::petsc_snapshot::SafeSnapshotFilePrefixFromPath("meshes/rmtest-6x5x4-trimmed-veri-nosmall");
  first.snapshot_index = 0;
  first.system_matrix = system_matrix;
  first.right_hand_side_vector = right_hand_side_vector;
  first.initial_guess_vector = initial_guess_vector;
  first.extra_metadata_entries = {{"label", "check"}, {"solver_name", "cg"}};

  ksptune::petsc_snapshot::KspSnapshotResult first_result;
  PetscCall(ksptune::petsc_snapshot::WriteKspSnapshot(first, &first_result));

  ksptune::petsc_snapshot::KspSnapshotRequest second = first;
  second.snapshot_index = 1;

  ksptune::petsc_snapshot::KspSnapshotResult second_result;
  PetscCall(ksptune::petsc_snapshot::WriteKspSnapshot(second, &second_result));

  ksptune::petsc_snapshot::PetscKspSnapshotExport skipped_export;
  skipped_export.mpi_communicator = PETSC_COMM_WORLD;
  skipped_export.solve_indices = "0";
  skipped_export.snapshot_file_prefix =
      ksptune::petsc_snapshot::SafeSnapshotFilePrefixFromPath("meshes/rmtest-6x5x4-trimmed-veri-nosmall");
  skipped_export.system_matrix = system_matrix;
  skipped_export.right_hand_side_vector = right_hand_side_vector;
  skipped_export.initial_guess_vector = initial_guess_vector;
  PetscCall(ksptune::petsc_snapshot::WritePetscKspSnapshotIfRequested(
      skipped_export,
      nullptr));

  ksptune::petsc_snapshot::KspSnapshotResult requested_result;
  ksptune::petsc_snapshot::PetscKspSnapshotExport requested_export = skipped_export;
  requested_export.snapshot_directory = output_directory;
  requested_export.metadata_entries.push_back(std::make_pair(
      "meshname",
      "meshes/rmtest-6x5x4-trimmed-veri-nosmall"));
  PetscCall(ksptune::petsc_snapshot::WritePetscKspSnapshotIfRequested(
      requested_export,
      &requested_result));

  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(rank == 0) {
    std::cout << second_result.metadata_file_path << '\n';
  }

  PetscCall(VecDestroy(&initial_guess_vector));
  PetscCall(VecDestroy(&right_hand_side_vector));
  PetscCall(MatDestroy(&system_matrix));
  PetscCall(PetscFinalize());
  return 0;
}
