#include "ksptune/petsc_snapshot.hpp"

#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <map>
#include <set>
#include <string>

namespace {

std::set<std::string> snapshot_files(const std::string& directory)
{
  std::set<std::string> files;
  DIR* entries = opendir(directory.c_str());
  if(!entries) return files;
  while(const auto* entry = readdir(entries)) {
    const std::string name = entry->d_name;
    if(name.compare(0, 2, "s_") == 0 &&
       name.size() > 4 && name.compare(name.size() - 4, 4, ".txt") == 0) {
      files.insert(directory + "/" + name);
    }
  }
  closedir(entries);
  return files;
}

PetscErrorCode check_snapshots(const std::string& directory,
                              const std::set<std::string>& previous_files)
{
  std::set<int> indices;
  std::set<std::string> matrices;
  int count = 0;
  for(const auto& file : snapshot_files(directory)) {
    if(previous_files.count(file)) continue;
    std::ifstream input(file);
    std::map<std::string, std::string> metadata;
    std::string line;
    while(std::getline(input, line)) {
      const auto separator = line.find('=');
      if(separator != std::string::npos) {
        metadata[line.substr(0, separator)] = line.substr(separator + 1);
      }
    }
    PetscCheck(metadata["solve_index"] == "0" || metadata["solve_index"] == "2",
               PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected exported solve index");
    indices.insert(std::stoi(metadata["solve_index"]));
    matrices.insert(metadata["A"]);
    PetscCheck(metadata["matrix_write"] == (metadata["solve_index"] == "0" ? "written" : "reused"),
               PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected reuse of the unchanged matrix");
    for(const char* key : {"A", "matrix_metadata", "b", "x0"}) {
      PetscCheck(!metadata[key].empty() && std::ifstream(directory + "/" + metadata[key]).good(),
                 PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing snapshot file: %s", key);
    }
    ++count;
  }
  PetscCheck(count == 2 && indices.size() == 2 && matrices.size() == 1,
             PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected solves 0 and 2 sharing one matrix");
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
  const char* output_directory = std::getenv("KSPTUNE_SNAPSHOT_DIR");
  PetscCheck(output_directory && *output_directory, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT,
             "Set KSPTUNE_SNAPSHOT_DIR and enable export for solves 0,2");
  int rank = 0;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  const auto previous_files = rank == 0 ? snapshot_files(output_directory) : std::set<std::string>();

  Mat matrix = nullptr;
  Vec b = nullptr, x = nullptr;
  PetscCall(create_test_system(PETSC_COMM_WORLD, &matrix, &b, &x));
  KSP ksp = nullptr;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, matrix, matrix));
  for(int solve = 0; solve < 3; ++solve) PetscCall(KSPTuneExport(ksp, b, x));

  PetscErrorCode check_error = rank == 0 ? check_snapshots(output_directory, previous_files) : 0;
  PetscCallMPI(MPI_Bcast(&check_error, 1, MPI_INT, 0, PETSC_COMM_WORLD));
  PetscCall(check_error);

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&matrix));
  PetscCall(PetscFinalize());
  return 0;
}
