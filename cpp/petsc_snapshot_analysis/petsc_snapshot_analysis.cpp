#include <petscksp.h>
#include "json_io.hpp"
#include "snapshot_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ksptune::json;
using ksptune::snapshot;
using ksptune::read_snapshot_collection;
using ksptune::load_matrix_binary;
using ksptune::load_vector_binary;

#define KSPTUNE_PETSC_CALL(call) \
  do { \
    PetscErrorCode ierr_ = (call); \
    if(ierr_) return ierr_; \
  } while(false)

struct analysis_args {
  std::string snapshot_collection_path;
  std::string json_output_path;
  int solve_target = -1;
  std::string snapshot_id;
  bool constant_nullspace = false;
  int field_nullspace = -1;
  int field_nullspace_block_size = 2;
  double rhs_compatibility_tolerance = 1.0e-10;
  double nullspace_residual_tolerance = 1.0e-10;
};

struct matrix_diagnostics {
  PetscInt row_count = 0;
  PetscInt column_count = 0;
  PetscInt local_row_count = 0;
  PetscInt local_column_count = 0;
  double nonzero_count = 0.0;
  double allocated_nonzero_count = 0.0;
  double petsc_matrix_memory_bytes = 0.0;
  double frobenius_norm = 0.0;
  bool symmetric = false;
};

struct nullspace_candidate {
  std::string label;
  std::string kind;
  bool has_constant = false;
  int field_index = -1;
  int block_size = 0;
  bool requested_by_user = false;
  bool declared_by_snapshot_metadata = false;
};

struct nullspace_diagnostics {
  nullspace_candidate candidate;
  bool right_is_nullspace_by_petsc_test = false;
  bool right_is_nullspace_by_residual = false;
  bool transpose_is_nullspace_by_residual = false;
  bool rhs_is_orthogonal_to_candidate = false;
  bool rhs_compatibility_check_applies = false;
  bool rhs_is_compatible_with_left_nullspace = false;
  bool rhs_has_nullspace_content = false;
  double right_residual_norm = 0.0;
  double right_residual_relative_to_matrix_norm = 0.0;
  double transpose_residual_norm = 0.0;
  double transpose_residual_relative_to_matrix_norm = 0.0;
  double rhs_norm = 0.0;
  double rhs_nullspace_component_norm = 0.0;
  double rhs_nullspace_component_relative_norm = 0.0;
  double rhs_norm_after_nullspace_removal = 0.0;
};

struct snapshot_analysis {
  int solve_index = 0;
  std::string snapshot_id;
  bool declared_petsc_nullspace = false;
  bool matrix_nullspace_attached = false;
  bool transpose_nullspace_attached = false;
  bool near_nullspace_attached = false;
  std::string nullspace_kind = "none";
  double rhs_norm = 0.0;
  matrix_diagnostics matrix;
  std::vector<nullspace_diagnostics> nullspaces;
};

struct analysis_result {
  std::string snapshot_collection_path;
  int snapshots = 0;
  std::string configured_nullspace = "none";
  int field_nullspace_index = -1;
  int field_nullspace_block_size = 0;
  double rhs_compatibility_tolerance = 1.0e-10;
  double nullspace_residual_tolerance = 1.0e-10;
  int declared_petsc_nullspace_count = 0;
  int nullspace_check_count = 0;
  bool all_checked_candidates_are_right_nullspaces = true;
  bool all_checked_candidates_are_transpose_nullspaces = true;
  bool all_rhs_orthogonal_to_checked_candidates = true;
  bool all_rhs_compatible_with_checked_left_nullspaces = true;
  double max_rhs_nullspace_component_relative_norm = 0.0;
  double max_right_residual_relative_to_matrix_norm = 0.0;
  double max_transpose_residual_relative_to_matrix_norm = 0.0;
  std::vector<snapshot_analysis> snapshot_results;
};

bool arg_value(int argc, char** argv, const std::string& key, std::string& value)
{
  for(int i = 1; i < argc; ++i) {
    const std::string current(argv[i]);
    if(current == key && i + 1 < argc) {
      value = argv[i + 1];
      return true;
    }
    const std::string prefix = key + "=";
    if(current.compare(0, prefix.size(), prefix) == 0) {
      value = current.substr(prefix.size());
      return true;
    }
  }
  return false;
}

int int_arg(int argc, char** argv, const std::string& key, int default_value)
{
  std::string value;
  if(!arg_value(argc, argv, key, value)) return default_value;
  return std::atoi(value.c_str());
}

double double_arg(int argc, char** argv, const std::string& key, double default_value)
{
  std::string value;
  if(!arg_value(argc, argv, key, value)) return default_value;
  return std::atof(value.c_str());
}

bool bool_arg(int argc, char** argv, const std::string& key, bool default_value)
{
  std::string value;
  if(!arg_value(argc, argv, key, value)) {
    for(int i = 1; i < argc; ++i) {
      if(std::string(argv[i]) == key) return true;
    }
    return default_value;
  }
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  return !(value == "0" || value == "false" || value == "off" || value == "no");
}

std::string nullspace_description(const analysis_args& args)
{
  if(args.field_nullspace >= 0) return "field";
  if(args.constant_nullspace) return "constant";
  return "none";
}

analysis_args parse_args(int argc, char** argv)
{
  analysis_args args;
  arg_value(argc, argv, "-snapshot_collection", args.snapshot_collection_path);
  arg_value(argc, argv, "-analysis_json_out", args.json_output_path);
  args.solve_target = int_arg(argc, argv, "-analysis_solve_target", args.solve_target);
  arg_value(argc, argv, "-analysis_snapshot_id", args.snapshot_id);
  args.constant_nullspace =
      bool_arg(argc, argv, "-analysis_constant_nullspace", false) ||
      bool_arg(argc, argv, "-replay_constant_nullspace", false);

  args.field_nullspace = int_arg(
      argc,
      argv,
      "-analysis_field_nullspace",
      args.field_nullspace);
  args.field_nullspace = int_arg(
      argc,
      argv,
      "-replay_field_nullspace",
      args.field_nullspace);
  args.field_nullspace_block_size = int_arg(
      argc,
      argv,
      "-analysis_field_nullspace_block_size",
      args.field_nullspace_block_size);
  args.field_nullspace_block_size = int_arg(
      argc,
      argv,
      "-replay_field_nullspace_block_size",
      args.field_nullspace_block_size);

  args.rhs_compatibility_tolerance = double_arg(
      argc,
      argv,
      "-analysis_rhs_compatibility_tolerance",
      args.rhs_compatibility_tolerance);
  args.nullspace_residual_tolerance = double_arg(
      argc,
      argv,
      "-analysis_nullspace_residual_tolerance",
      args.nullspace_residual_tolerance);
  return args;
}

void clear_analysis_options()
{
  const char* keys[] = {
    "-snapshot_collection",
    "-analysis_json_out",
    "-analysis_solve_target",
    "-analysis_snapshot_id",
    "-analysis_constant_nullspace",
    "-analysis_field_nullspace",
    "-analysis_field_nullspace_block_size",
    "-analysis_rhs_compatibility_tolerance",
    "-analysis_nullspace_residual_tolerance",
    "-replay_constant_nullspace",
    "-replay_field_nullspace",
    "-replay_field_nullspace_block_size",
  };
  for(const char* key : keys) {
    PetscOptionsClearValue(nullptr, key);
  }
}

PetscErrorCode collect_matrix_diagnostics(Mat matrix, matrix_diagnostics& diagnostics)
{
  KSPTUNE_PETSC_CALL(MatGetSize(matrix, &diagnostics.row_count, &diagnostics.column_count));
  KSPTUNE_PETSC_CALL(MatGetLocalSize(
      matrix,
      &diagnostics.local_row_count,
      &diagnostics.local_column_count));

  MatInfo info;
  KSPTUNE_PETSC_CALL(MatGetInfo(matrix, MAT_GLOBAL_SUM, &info));
  diagnostics.nonzero_count = double(info.nz_used);
  diagnostics.allocated_nonzero_count = double(info.nz_allocated);
  diagnostics.petsc_matrix_memory_bytes = double(info.memory);

  PetscReal frobenius_norm = 0.0;
  KSPTUNE_PETSC_CALL(MatNorm(matrix, NORM_FROBENIUS, &frobenius_norm));
  diagnostics.frobenius_norm = double(frobenius_norm);

  PetscBool symmetric = PETSC_FALSE;
  KSPTUNE_PETSC_CALL(MatIsSymmetric(matrix, 1.0e-12, &symmetric));
  diagnostics.symmetric = symmetric == PETSC_TRUE;
  return 0;
}

PetscErrorCode create_normalized_constant_basis(Vec reference_vector, Vec* basis)
{
  KSPTUNE_PETSC_CALL(VecDuplicate(reference_vector, basis));
  KSPTUNE_PETSC_CALL(VecSet(*basis, PetscScalar(1.0)));
  PetscReal norm = 0.0;
  KSPTUNE_PETSC_CALL(VecNormalize(*basis, &norm));
  if(norm == 0.0) {
    VecDestroy(basis);
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Constant nullspace basis is empty\n");
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  return 0;
}

PetscErrorCode create_normalized_field_basis(Vec reference_vector,
                                             PetscInt block_size,
                                             PetscInt field,
                                             Vec* basis)
{
  if(block_size <= 0) {
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace block size must be positive\n");
    return PETSC_ERR_ARG_OUTOFRANGE;
  }
  if(field < 0 || field >= block_size) {
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace index must be in [0, block_size)\n");
    return PETSC_ERR_ARG_OUTOFRANGE;
  }

  KSPTUNE_PETSC_CALL(VecDuplicate(reference_vector, basis));

  PetscInt ownership_start = 0;
  PetscInt ownership_end = 0;
  PetscInt local_size = 0;
  KSPTUNE_PETSC_CALL(VecGetOwnershipRange(reference_vector, &ownership_start, &ownership_end));
  KSPTUNE_PETSC_CALL(VecGetLocalSize(reference_vector, &local_size));

  PetscScalar* values = nullptr;
  KSPTUNE_PETSC_CALL(VecGetArray(*basis, &values));
  for(PetscInt local = 0; local < local_size; ++local) {
    const PetscInt global_index = ownership_start + local;
    values[local] =
        (global_index % block_size == field) ? PetscScalar(1.0) : PetscScalar(0.0);
  }
  KSPTUNE_PETSC_CALL(VecRestoreArray(*basis, &values));
  KSPTUNE_PETSC_CALL(VecAssemblyBegin(*basis));
  KSPTUNE_PETSC_CALL(VecAssemblyEnd(*basis));

  PetscReal norm = 0.0;
  KSPTUNE_PETSC_CALL(VecNormalize(*basis, &norm));
  if(norm == 0.0) {
    VecDestroy(basis);
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace basis is empty\n");
    return PETSC_ERR_ARG_WRONGSTATE;
  }
  return 0;
}

PetscErrorCode create_normalized_candidate_basis(const nullspace_candidate& candidate,
                                                 Vec reference_vector,
                                                 Vec* basis)
{
  if(candidate.has_constant) return create_normalized_constant_basis(reference_vector, basis);
  return create_normalized_field_basis(
      reference_vector,
      PetscInt(candidate.block_size),
      PetscInt(candidate.field_index),
      basis);
}

PetscErrorCode create_candidate_nullspace(const nullspace_candidate& candidate,
                                          Vec reference_vector,
                                          MatNullSpace* nullspace)
{
  MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(reference_vector));
  if(candidate.has_constant) {
    KSPTUNE_PETSC_CALL(MatNullSpaceCreate(comm, PETSC_TRUE, 0, nullptr, nullspace));
    return 0;
  }

  Vec basis = nullptr;
  KSPTUNE_PETSC_CALL(create_normalized_candidate_basis(candidate, reference_vector, &basis));
  KSPTUNE_PETSC_CALL(MatNullSpaceCreate(comm, PETSC_FALSE, 1, &basis, nullspace));
  KSPTUNE_PETSC_CALL(VecDestroy(&basis));
  return 0;
}

std::vector<nullspace_candidate> nullspace_candidates(const analysis_args& args,
                                                      const snapshot& current_snapshot)
{
  std::vector<nullspace_candidate> candidates;

  nullspace_candidate constant_candidate;
  constant_candidate.label = "constant";
  constant_candidate.kind = "constant";
  constant_candidate.has_constant = true;
  constant_candidate.requested_by_user = args.constant_nullspace;
  constant_candidate.declared_by_snapshot_metadata =
      current_snapshot.nullspace_kind == "constant" &&
      (current_snapshot.matrix_nullspace_attached ||
       current_snapshot.transpose_nullspace_attached ||
       current_snapshot.near_nullspace_attached);
  candidates.push_back(constant_candidate);

  if(args.field_nullspace >= 0) {
    nullspace_candidate field_candidate;
    field_candidate.label = "field:" + std::to_string(args.field_nullspace);
    field_candidate.kind = "field";
    field_candidate.field_index = args.field_nullspace;
    field_candidate.block_size = args.field_nullspace_block_size;
    field_candidate.requested_by_user = true;
    field_candidate.declared_by_snapshot_metadata =
        current_snapshot.nullspace_kind == "field" &&
        current_snapshot.nullspace_field_index == args.field_nullspace &&
        current_snapshot.nullspace_block_size == args.field_nullspace_block_size;
    candidates.push_back(field_candidate);
  }

  return candidates;
}

double relative_to_matrix_norm(double value, double matrix_norm)
{
  return matrix_norm > 0.0 ? value / matrix_norm : value;
}

PetscErrorCode compute_matrix_times_basis_norm(Mat matrix,
                                               Vec basis,
                                               double matrix_norm,
                                               bool transpose,
                                               double& residual_norm,
                                               double& residual_relative_to_matrix_norm)
{
  Vec product = nullptr;
  if(transpose) {
    KSPTUNE_PETSC_CALL(MatCreateVecs(matrix, &product, nullptr));
    KSPTUNE_PETSC_CALL(MatMultTranspose(matrix, basis, product));
  } else {
    KSPTUNE_PETSC_CALL(MatCreateVecs(matrix, nullptr, &product));
    KSPTUNE_PETSC_CALL(MatMult(matrix, basis, product));
  }

  PetscReal norm = 0.0;
  KSPTUNE_PETSC_CALL(VecNorm(product, NORM_2, &norm));
  residual_norm = double(norm);
  residual_relative_to_matrix_norm = relative_to_matrix_norm(residual_norm, matrix_norm);
  KSPTUNE_PETSC_CALL(VecDestroy(&product));
  return 0;
}

PetscErrorCode analyze_rhs_nullspace_component(const nullspace_candidate& candidate,
                                               Vec right_hand_side,
                                               double rhs_compatibility_tolerance,
                                               nullspace_diagnostics& diagnostics)
{
  MatNullSpace left_nullspace = nullptr;
  Vec projected_right_hand_side = nullptr;
  Vec removed_component = nullptr;

  KSPTUNE_PETSC_CALL(create_candidate_nullspace(candidate, right_hand_side, &left_nullspace));
  KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, &projected_right_hand_side));
  KSPTUNE_PETSC_CALL(VecCopy(right_hand_side, projected_right_hand_side));
  KSPTUNE_PETSC_CALL(MatNullSpaceRemove(left_nullspace, projected_right_hand_side));

  KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, &removed_component));
  KSPTUNE_PETSC_CALL(VecCopy(right_hand_side, removed_component));
  KSPTUNE_PETSC_CALL(VecAXPY(removed_component, PetscScalar(-1.0), projected_right_hand_side));

  PetscReal rhs_norm = 0.0;
  PetscReal projected_norm = 0.0;
  PetscReal component_norm = 0.0;
  KSPTUNE_PETSC_CALL(VecNorm(right_hand_side, NORM_2, &rhs_norm));
  KSPTUNE_PETSC_CALL(VecNorm(projected_right_hand_side, NORM_2, &projected_norm));
  KSPTUNE_PETSC_CALL(VecNorm(removed_component, NORM_2, &component_norm));

  diagnostics.rhs_norm = double(rhs_norm);
  diagnostics.rhs_norm_after_nullspace_removal = double(projected_norm);
  diagnostics.rhs_nullspace_component_norm = double(component_norm);
  diagnostics.rhs_nullspace_component_relative_norm = rhs_norm > 0.0
      ? double(component_norm / rhs_norm)
      : double(component_norm);
  diagnostics.rhs_is_orthogonal_to_candidate =
      diagnostics.rhs_nullspace_component_relative_norm <= rhs_compatibility_tolerance;
  diagnostics.rhs_has_nullspace_content = !diagnostics.rhs_is_orthogonal_to_candidate;

  KSPTUNE_PETSC_CALL(VecDestroy(&removed_component));
  KSPTUNE_PETSC_CALL(VecDestroy(&projected_right_hand_side));
  KSPTUNE_PETSC_CALL(MatNullSpaceDestroy(&left_nullspace));
  return 0;
}

PetscErrorCode analyze_nullspace_candidate(Mat matrix,
                                           Vec right_hand_side,
                                           const nullspace_candidate& candidate,
                                           double matrix_norm,
                                           double rhs_compatibility_tolerance,
                                           double nullspace_residual_tolerance,
                                           nullspace_diagnostics& diagnostics)
{
  Vec domain_reference = nullptr;
  Vec right_basis = nullptr;
  Vec left_basis = nullptr;
  MatNullSpace right_nullspace = nullptr;

  diagnostics.candidate = candidate;

  KSPTUNE_PETSC_CALL(MatCreateVecs(matrix, &domain_reference, nullptr));
  KSPTUNE_PETSC_CALL(create_normalized_candidate_basis(candidate, domain_reference, &right_basis));
  KSPTUNE_PETSC_CALL(compute_matrix_times_basis_norm(
      matrix,
      right_basis,
      matrix_norm,
      false,
      diagnostics.right_residual_norm,
      diagnostics.right_residual_relative_to_matrix_norm));
  diagnostics.right_is_nullspace_by_residual =
      diagnostics.right_residual_relative_to_matrix_norm <= nullspace_residual_tolerance;

  KSPTUNE_PETSC_CALL(create_candidate_nullspace(candidate, domain_reference, &right_nullspace));
  PetscBool petsc_test = PETSC_FALSE;
  KSPTUNE_PETSC_CALL(MatNullSpaceTest(right_nullspace, matrix, &petsc_test));
  diagnostics.right_is_nullspace_by_petsc_test = petsc_test == PETSC_TRUE;

  KSPTUNE_PETSC_CALL(create_normalized_candidate_basis(candidate, right_hand_side, &left_basis));
  KSPTUNE_PETSC_CALL(compute_matrix_times_basis_norm(
      matrix,
      left_basis,
      matrix_norm,
      true,
      diagnostics.transpose_residual_norm,
      diagnostics.transpose_residual_relative_to_matrix_norm));
  diagnostics.transpose_is_nullspace_by_residual =
      diagnostics.transpose_residual_relative_to_matrix_norm <= nullspace_residual_tolerance;

  KSPTUNE_PETSC_CALL(analyze_rhs_nullspace_component(
      candidate,
      right_hand_side,
      rhs_compatibility_tolerance,
      diagnostics));
  diagnostics.rhs_compatibility_check_applies = diagnostics.transpose_is_nullspace_by_residual;
  diagnostics.rhs_is_compatible_with_left_nullspace =
      diagnostics.rhs_compatibility_check_applies &&
      diagnostics.rhs_is_orthogonal_to_candidate;

  KSPTUNE_PETSC_CALL(MatNullSpaceDestroy(&right_nullspace));
  KSPTUNE_PETSC_CALL(VecDestroy(&left_basis));
  KSPTUNE_PETSC_CALL(VecDestroy(&right_basis));
  KSPTUNE_PETSC_CALL(VecDestroy(&domain_reference));
  return 0;
}

PetscErrorCode analyze_snapshot(const analysis_args& args,
                                const snapshot& current_snapshot,
                                snapshot_analysis& result)
{
  Mat matrix = nullptr;
  Vec right_hand_side = nullptr;

  result.solve_index = current_snapshot.solve_index;
  result.snapshot_id = current_snapshot.snapshot_id;
  result.declared_petsc_nullspace = current_snapshot.nullspace;
  result.matrix_nullspace_attached = current_snapshot.matrix_nullspace_attached;
  result.transpose_nullspace_attached = current_snapshot.transpose_nullspace_attached;
  result.near_nullspace_attached = current_snapshot.near_nullspace_attached;
  result.nullspace_kind = current_snapshot.nullspace_kind;

  KSPTUNE_PETSC_CALL(load_matrix_binary(current_snapshot, &matrix));
  KSPTUNE_PETSC_CALL(load_vector_binary(
      current_snapshot.right_hand_side_file_path,
      current_snapshot.rhs_ownership_ranges,
      current_snapshot.right_hand_side_size,
      &right_hand_side));
  KSPTUNE_PETSC_CALL(collect_matrix_diagnostics(matrix, result.matrix));

  PetscReal rhs_norm = 0.0;
  KSPTUNE_PETSC_CALL(VecNorm(right_hand_side, NORM_2, &rhs_norm));
  result.rhs_norm = double(rhs_norm);

  for(const nullspace_candidate& candidate : nullspace_candidates(args, current_snapshot)) {
    nullspace_diagnostics diagnostics;
    KSPTUNE_PETSC_CALL(analyze_nullspace_candidate(
        matrix,
        right_hand_side,
        candidate,
        result.matrix.frobenius_norm,
        args.rhs_compatibility_tolerance,
        args.nullspace_residual_tolerance,
        diagnostics));
    result.nullspaces.push_back(diagnostics);
  }

  KSPTUNE_PETSC_CALL(VecDestroy(&right_hand_side));
  KSPTUNE_PETSC_CALL(MatDestroy(&matrix));
  return 0;
}

void update_summary(analysis_result& result, const snapshot_analysis& snapshot_result)
{
  if(snapshot_result.declared_petsc_nullspace) ++result.declared_petsc_nullspace_count;

  for(const nullspace_diagnostics& diagnostics : snapshot_result.nullspaces) {
    ++result.nullspace_check_count;
    result.all_checked_candidates_are_right_nullspaces =
        result.all_checked_candidates_are_right_nullspaces &&
        diagnostics.right_is_nullspace_by_residual;
    result.all_checked_candidates_are_transpose_nullspaces =
        result.all_checked_candidates_are_transpose_nullspaces &&
        diagnostics.transpose_is_nullspace_by_residual;
    result.all_rhs_orthogonal_to_checked_candidates =
        result.all_rhs_orthogonal_to_checked_candidates &&
        diagnostics.rhs_is_orthogonal_to_candidate;
    result.all_rhs_compatible_with_checked_left_nullspaces =
        result.all_rhs_compatible_with_checked_left_nullspaces &&
        (!diagnostics.rhs_compatibility_check_applies ||
         diagnostics.rhs_is_compatible_with_left_nullspace);
    result.max_rhs_nullspace_component_relative_norm = std::max(
        result.max_rhs_nullspace_component_relative_norm,
        diagnostics.rhs_nullspace_component_relative_norm);
    result.max_right_residual_relative_to_matrix_norm = std::max(
        result.max_right_residual_relative_to_matrix_norm,
        diagnostics.right_residual_relative_to_matrix_norm);
    result.max_transpose_residual_relative_to_matrix_norm = std::max(
        result.max_transpose_residual_relative_to_matrix_norm,
        diagnostics.transpose_residual_relative_to_matrix_norm);
  }
}


PetscErrorCode write_json_result(const analysis_result& result, const std::string& path)
{
  json output = {
    {"snapshot_collection_path", result.snapshot_collection_path},
    {"snapshots", result.snapshots},
    {"configured_nullspace", result.configured_nullspace},
    {"field_nullspace_index", result.field_nullspace_index},
    {"field_nullspace_block_size", result.field_nullspace_block_size},
    {"rhs_compatibility_tolerance", result.rhs_compatibility_tolerance},
    {"nullspace_residual_tolerance", result.nullspace_residual_tolerance},
    {"declared_petsc_nullspace_count", result.declared_petsc_nullspace_count},
    {"nullspace_check_count", result.nullspace_check_count},
    {"all_checked_candidates_are_right_nullspaces", result.all_checked_candidates_are_right_nullspaces},
    {"all_checked_candidates_are_transpose_nullspaces", result.all_checked_candidates_are_transpose_nullspaces},
    {"all_rhs_orthogonal_to_checked_candidates", result.all_rhs_orthogonal_to_checked_candidates},
    {"all_rhs_compatible_with_checked_left_nullspaces", result.all_rhs_compatible_with_checked_left_nullspaces},
    {"max_rhs_nullspace_component_relative_norm", result.max_rhs_nullspace_component_relative_norm},
    {"max_right_residual_relative_to_matrix_norm", result.max_right_residual_relative_to_matrix_norm},
    {"max_transpose_residual_relative_to_matrix_norm", result.max_transpose_residual_relative_to_matrix_norm},
    {"schema_version", 1},
    {"note", "PETSc binary matrix dumps do not serialize MatNullSpace basis vectors. KSPTune records whether a snapshot declared a PETSc nullspace and tests explicit candidate nullspaces with PETSc."}
  };
  output["snapshot_results"] = json::array();
  for(const auto& snapshot_result : result.snapshot_results) {
    json entry = {
      {"solve_index", snapshot_result.solve_index},
      {"snapshot_id", snapshot_result.snapshot_id},
      {"declared_petsc_nullspace", snapshot_result.declared_petsc_nullspace},
      {"matrix_nullspace_attached", snapshot_result.matrix_nullspace_attached},
      {"transpose_nullspace_attached", snapshot_result.transpose_nullspace_attached},
      {"near_nullspace_attached", snapshot_result.near_nullspace_attached},
      {"nullspace_kind", snapshot_result.nullspace_kind},
      {"rhs_norm", snapshot_result.rhs_norm}
    };
    const auto& matrix = snapshot_result.matrix;
    json matrix_json = {
      {"rows", matrix.row_count},
      {"cols", matrix.column_count},
      {"local_rows", matrix.local_row_count},
      {"local_cols", matrix.local_column_count},
      {"nonzeros_used", matrix.nonzero_count},
      {"nonzeros_allocated", matrix.allocated_nonzero_count},
      {"memory_bytes", matrix.petsc_matrix_memory_bytes},
      {"frobenius_norm", matrix.frobenius_norm},
      {"symmetric", matrix.symmetric}
    };
    entry["matrix"] = std::move(matrix_json);
    entry["nullspaces"] = json::array();
    for(const auto& diagnostics : snapshot_result.nullspaces) {
      const auto& candidate = diagnostics.candidate;
      json check = {
        {"label", candidate.label},
        {"kind", candidate.kind},
        {"field_index", candidate.field_index},
        {"block_size", candidate.block_size},
        {"requested_by_user", candidate.requested_by_user},
        {"declared_by_snapshot_metadata", candidate.declared_by_snapshot_metadata},
        {"right_is_nullspace_by_petsc_test", diagnostics.right_is_nullspace_by_petsc_test},
        {"right_is_nullspace_by_residual", diagnostics.right_is_nullspace_by_residual},
        {"right_residual_norm", diagnostics.right_residual_norm},
        {"right_residual_relative_to_matrix_norm", diagnostics.right_residual_relative_to_matrix_norm},
        {"transpose_is_nullspace_by_residual", diagnostics.transpose_is_nullspace_by_residual},
        {"transpose_residual_norm", diagnostics.transpose_residual_norm},
        {"transpose_residual_relative_to_matrix_norm", diagnostics.transpose_residual_relative_to_matrix_norm},
        {"rhs_norm", diagnostics.rhs_norm},
        {"rhs_nullspace_component_norm", diagnostics.rhs_nullspace_component_norm},
        {"rhs_nullspace_component_relative_norm", diagnostics.rhs_nullspace_component_relative_norm},
        {"rhs_norm_after_nullspace_removal", diagnostics.rhs_norm_after_nullspace_removal},
        {"rhs_has_nullspace_content", diagnostics.rhs_has_nullspace_content},
        {"rhs_is_orthogonal_to_candidate", diagnostics.rhs_is_orthogonal_to_candidate},
        {"rhs_compatibility_check_applies", diagnostics.rhs_compatibility_check_applies},
        {"rhs_is_compatible_with_left_nullspace", diagnostics.rhs_is_compatible_with_left_nullspace}
      };
      entry["nullspaces"].push_back(std::move(check));
    }
    output["snapshot_results"].push_back(std::move(entry));
  }
  return ksptune::write_json_output(output, path);
}

PetscErrorCode run_analysis(const analysis_args& args)
{
  if(args.snapshot_collection_path.empty()) {
    PetscFPrintf(
        PETSC_COMM_WORLD,
        stderr,
        "Usage: ksptune-petsc-snapshot-analysis -snapshot_collection snapshot_collection.csv "
        "[-analysis_json_out result.json]\n");
    return PETSC_ERR_ARG_WRONG;
  }
  std::vector<snapshot> snapshots = read_snapshot_collection(args.snapshot_collection_path);
  if(snapshots.empty()) {
    PetscFPrintf(
        PETSC_COMM_WORLD,
        stderr,
        "No snapshots found in %s\n",
        args.snapshot_collection_path.c_str());
    return PETSC_ERR_FILE_OPEN;
  }
  KSPTUNE_PETSC_CALL(ksptune::select_snapshot(snapshots, args.solve_target, args.snapshot_id));

  analysis_result result;
  result.snapshot_collection_path = args.snapshot_collection_path;
  result.snapshots = int(snapshots.size());
  result.configured_nullspace = nullspace_description(args);
  result.field_nullspace_index = args.field_nullspace;
  result.field_nullspace_block_size = args.field_nullspace >= 0
      ? args.field_nullspace_block_size
      : 0;
  result.rhs_compatibility_tolerance = args.rhs_compatibility_tolerance;
  result.nullspace_residual_tolerance = args.nullspace_residual_tolerance;

  for(const snapshot& current_snapshot : snapshots) {
    snapshot_analysis snapshot_result;
    KSPTUNE_PETSC_CALL(analyze_snapshot(args, current_snapshot, snapshot_result));
    update_summary(result, snapshot_result);
    result.snapshot_results.push_back(snapshot_result);
  }

  KSPTUNE_PETSC_CALL(write_json_result(result, args.json_output_path));
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if(ierr) return int(ierr);
  const analysis_args args = parse_args(argc, argv);
  clear_analysis_options();
  ierr = run_analysis(args);
  const PetscErrorCode finalize_ierr = PetscFinalize();
  return int(ierr ? ierr : finalize_ierr);
}
