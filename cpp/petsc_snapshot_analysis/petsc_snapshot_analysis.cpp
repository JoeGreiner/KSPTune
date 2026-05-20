#include <petscksp.h>

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

#define KSPTUNE_PETSC_CALL(call) \
  do { \
    PetscErrorCode ierr_ = (call); \
    if(ierr_) return ierr_; \
  } while(false)

struct analysis_args {
  std::string snapshot_collection_path;
  std::string json_output_path;
  int solve_target = -1;
  bool constant_nullspace = false;
  int field_nullspace = -1;
  int field_nullspace_block_size = 2;
  double rhs_compatibility_tolerance = 1.0e-10;
  double nullspace_residual_tolerance = 1.0e-10;
};

struct snapshot {
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
  std::vector<PetscInt> row_ownership_ranges;
  std::vector<PetscInt> rhs_ownership_ranges;
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

std::string trim(const std::string& text)
{
  size_t first = 0;
  while(first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) ++first;
  size_t last = text.size();
  while(last > first && std::isspace(static_cast<unsigned char>(text[last - 1]))) --last;
  return text.substr(first, last - first);
}

std::vector<std::string> split_csv_line(const std::string& line)
{
  std::vector<std::string> fields;
  std::string field;
  bool in_quotes = false;
  for(size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if(c == '"') {
      if(in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if(c == ',' && !in_quotes) {
      fields.push_back(trim(field));
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  fields.push_back(trim(field));
  return fields;
}

bool file_exists(const std::string& path)
{
  std::ifstream input(path.c_str());
  return input.good();
}

bool is_absolute_path(const std::string& path)
{
  return !path.empty() && path[0] == '/';
}

std::string directory_name(const std::string& path)
{
  const size_t position = path.find_last_of('/');
  if(position == std::string::npos) return ".";
  if(position == 0) return "/";
  return path.substr(0, position);
}

std::string resolve_snapshot_path(const std::string& path, const std::string& base_directory)
{
  if(path.empty() || is_absolute_path(path)) return path;
  const std::string snapshot_collection_relative_path = base_directory + "/" + path;
  if(file_exists(snapshot_collection_relative_path) || !file_exists(path)) {
    return snapshot_collection_relative_path;
  }
  return path;
}

std::map<std::string, std::string> read_metadata(const std::string& path)
{
  std::map<std::string, std::string> metadata;
  if(path.empty() || !file_exists(path)) return metadata;

  std::ifstream input(path.c_str());
  std::string line;
  while(std::getline(input, line)) {
    const size_t position = line.find('=');
    if(position == std::string::npos) continue;
    metadata[trim(line.substr(0, position))] = trim(line.substr(position + 1));
  }
  return metadata;
}

bool parse_bool(const std::string& value)
{
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

PetscInt parse_petsc_int(const std::string& value, PetscInt default_value)
{
  if(value.empty()) return default_value;
  return PetscInt(std::atoll(value.c_str()));
}

int parse_int(const std::string& value, int default_value)
{
  if(value.empty()) return default_value;
  return std::atoi(value.c_str());
}

std::vector<PetscInt> parse_petsc_int_list(const std::string& text)
{
  std::vector<PetscInt> values;
  std::stringstream stream(text);
  std::string item;
  while(std::getline(stream, item, ',')) {
    item = trim(item);
    if(!item.empty()) values.push_back(PetscInt(std::atoll(item.c_str())));
  }
  return values;
}

std::string field_or_empty(const std::map<std::string, std::string>& row,
                           const std::string& key)
{
  const auto iterator = row.find(key);
  return iterator == row.end() ? "" : iterator->second;
}

snapshot snapshot_from_row(const std::map<std::string, std::string>& row,
                           const std::string& base_directory)
{
  snapshot current_snapshot;
  current_snapshot.solve_index = parse_int(field_or_empty(row, "solve_index"), 0);
  current_snapshot.matrix_file_path =
      resolve_snapshot_path(field_or_empty(row, "A"), base_directory);
  current_snapshot.right_hand_side_file_path =
      resolve_snapshot_path(field_or_empty(row, "b"), base_directory);
  current_snapshot.initial_guess_file_path =
      resolve_snapshot_path(field_or_empty(row, "x0"), base_directory);
  current_snapshot.metadata_file_path =
      resolve_snapshot_path(field_or_empty(row, "meta"), base_directory);
  current_snapshot.rows = parse_petsc_int(field_or_empty(row, "rows"), 0);
  current_snapshot.cols = parse_petsc_int(field_or_empty(row, "cols"), 0);
  current_snapshot.right_hand_side_size = parse_petsc_int(field_or_empty(row, "rhs_size"), 0);
  current_snapshot.mpi_size = parse_int(field_or_empty(row, "mpi_size"), 1);
  current_snapshot.matrix_nullspace_attached =
      parse_bool(field_or_empty(row, "matrix_nullspace_attached")) ||
      parse_bool(field_or_empty(row, "nullspace"));
  current_snapshot.transpose_nullspace_attached =
      parse_bool(field_or_empty(row, "transpose_nullspace_attached"));
  current_snapshot.near_nullspace_attached =
      parse_bool(field_or_empty(row, "near_nullspace_attached"));
  current_snapshot.nullspace_kind = field_or_empty(row, "nullspace_kind");
  if(current_snapshot.nullspace_kind.empty()) {
    current_snapshot.nullspace_kind =
        current_snapshot.matrix_nullspace_attached ? "unknown" : "none";
  }
  current_snapshot.nullspace_field_index =
      parse_int(field_or_empty(row, "nullspace_field_index"), -1);
  current_snapshot.nullspace_block_size =
      parse_int(field_or_empty(row, "nullspace_block_size"), 0);
  current_snapshot.rhs_nullspace_component_removed =
      field_or_empty(row, "rhs_nullspace_component_removed");
  if(current_snapshot.rhs_nullspace_component_removed.empty()) {
    current_snapshot.rhs_nullspace_component_removed = "unknown";
  }

  const std::map<std::string, std::string> metadata =
      read_metadata(current_snapshot.metadata_file_path);
  if(!metadata.empty()) {
    if(metadata.count("rows")) {
      current_snapshot.rows = parse_petsc_int(metadata.at("rows"), current_snapshot.rows);
    }
    if(metadata.count("cols")) {
      current_snapshot.cols = parse_petsc_int(metadata.at("cols"), current_snapshot.cols);
    }
    if(metadata.count("rhs_size")) {
      current_snapshot.right_hand_side_size =
          parse_petsc_int(metadata.at("rhs_size"), current_snapshot.right_hand_side_size);
    }
    if(metadata.count("mpi_size")) {
      current_snapshot.mpi_size = parse_int(metadata.at("mpi_size"), current_snapshot.mpi_size);
    }
    if(metadata.count("symmetric")) current_snapshot.symmetric = parse_bool(metadata.at("symmetric"));
    if(metadata.count("spd")) current_snapshot.spd = parse_bool(metadata.at("spd"));
    if(metadata.count("matrix_nullspace_attached")) {
      current_snapshot.matrix_nullspace_attached =
          parse_bool(metadata.at("matrix_nullspace_attached"));
    }
    if(metadata.count("transpose_nullspace_attached")) {
      current_snapshot.transpose_nullspace_attached =
          parse_bool(metadata.at("transpose_nullspace_attached"));
    }
    if(metadata.count("near_nullspace_attached")) {
      current_snapshot.near_nullspace_attached = parse_bool(metadata.at("near_nullspace_attached"));
    }
    if(metadata.count("nullspace_kind")) {
      current_snapshot.nullspace_kind = metadata.at("nullspace_kind");
    }
    if(metadata.count("nullspace_field_index")) {
      current_snapshot.nullspace_field_index =
          parse_int(metadata.at("nullspace_field_index"), current_snapshot.nullspace_field_index);
    }
    if(metadata.count("nullspace_block_size")) {
      current_snapshot.nullspace_block_size =
          parse_int(metadata.at("nullspace_block_size"), current_snapshot.nullspace_block_size);
    }
    if(metadata.count("rhs_nullspace_component_removed")) {
      current_snapshot.rhs_nullspace_component_removed =
          metadata.at("rhs_nullspace_component_removed");
    }
    if(metadata.count("nullspace")) {
      current_snapshot.matrix_nullspace_attached =
          parse_bool(metadata.at("nullspace"));
    }
    if(metadata.count("row_ownership_ranges")) {
      current_snapshot.row_ownership_ranges =
          parse_petsc_int_list(metadata.at("row_ownership_ranges"));
    }
    if(metadata.count("rhs_ownership_ranges")) {
      current_snapshot.rhs_ownership_ranges =
          parse_petsc_int_list(metadata.at("rhs_ownership_ranges"));
    }
  }
  current_snapshot.nullspace =
      current_snapshot.matrix_nullspace_attached ||
      current_snapshot.transpose_nullspace_attached ||
      current_snapshot.near_nullspace_attached;
  if(current_snapshot.nullspace &&
     (current_snapshot.nullspace_kind.empty() || current_snapshot.nullspace_kind == "none")) {
    current_snapshot.nullspace_kind = "unknown";
  }
  return current_snapshot;
}

std::vector<snapshot> read_snapshot_collection(const std::string& path)
{
  std::ifstream input(path.c_str());
  if(!input) return {};

  std::string line;
  if(!std::getline(input, line)) return {};

  const std::vector<std::string> header = split_csv_line(line);
  const std::string base_directory = directory_name(path);
  std::vector<snapshot> snapshots;
  while(std::getline(input, line)) {
    if(trim(line).empty()) continue;
    const std::vector<std::string> fields = split_csv_line(line);
    std::map<std::string, std::string> row;
    for(size_t i = 0; i < header.size() && i < fields.size(); ++i) {
      row[header[i]] = fields[i];
    }
    snapshots.push_back(snapshot_from_row(row, base_directory));
  }

  std::sort(snapshots.begin(), snapshots.end(), [](const snapshot& left, const snapshot& right) {
    return left.solve_index < right.solve_index;
  });
  return snapshots;
}

bool ownership_ranges_match_current(const std::vector<PetscInt>& ranges)
{
  int communicator_size = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &communicator_size);
  return ranges.size() == static_cast<size_t>(communicator_size) + 1;
}

PetscInt local_size_from_ranges(const std::vector<PetscInt>& ranges)
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  return ranges[static_cast<size_t>(rank) + 1] - ranges[static_cast<size_t>(rank)];
}

PetscErrorCode load_matrix_binary(const snapshot& current_snapshot, Mat* matrix)
{
  PetscViewer viewer = nullptr;
  KSPTUNE_PETSC_CALL(PetscViewerBinaryOpen(
      PETSC_COMM_WORLD,
      current_snapshot.matrix_file_path.c_str(),
      FILE_MODE_READ,
      &viewer));
  KSPTUNE_PETSC_CALL(MatCreate(PETSC_COMM_WORLD, matrix));
  if(ownership_ranges_match_current(current_snapshot.row_ownership_ranges)) {
    const PetscInt local_rows = local_size_from_ranges(current_snapshot.row_ownership_ranges);
    KSPTUNE_PETSC_CALL(MatSetSizes(
        *matrix,
        local_rows,
        local_rows,
        current_snapshot.rows,
        current_snapshot.cols));
    KSPTUNE_PETSC_CALL(MatSetType(*matrix, MATAIJ));
  }
  KSPTUNE_PETSC_CALL(MatLoad(*matrix, viewer));
  KSPTUNE_PETSC_CALL(PetscViewerDestroy(&viewer));
  return 0;
}

PetscErrorCode load_vector_binary(const std::string& path,
                                  const std::vector<PetscInt>& ownership_ranges,
                                  PetscInt global_size,
                                  Vec* vector)
{
  PetscViewer viewer = nullptr;
  KSPTUNE_PETSC_CALL(PetscViewerBinaryOpen(
      PETSC_COMM_WORLD,
      path.c_str(),
      FILE_MODE_READ,
      &viewer));
  KSPTUNE_PETSC_CALL(VecCreate(PETSC_COMM_WORLD, vector));
  if(ownership_ranges_match_current(ownership_ranges)) {
    KSPTUNE_PETSC_CALL(VecSetSizes(*vector, local_size_from_ranges(ownership_ranges), global_size));
    KSPTUNE_PETSC_CALL(VecSetType(*vector, VECMPI));
  }
  KSPTUNE_PETSC_CALL(VecLoad(*vector, viewer));
  KSPTUNE_PETSC_CALL(PetscViewerDestroy(&viewer));
  return 0;
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

int find_snapshot_by_solve_index(const std::vector<snapshot>& snapshots, int solve_index)
{
  for(size_t i = 0; i < snapshots.size(); ++i) {
    if(snapshots[i].solve_index == solve_index) return int(i);
  }
  return -1;
}

std::string json_escape(const std::string& input)
{
  std::ostringstream output;
  for(char c : input) {
    if(c == '\\' || c == '"') output << '\\' << c;
    else if(c == '\n') output << "\\n";
    else output << c;
  }
  return output.str();
}

void write_json_result(const analysis_result& result, const std::string& path)
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(rank != 0) return;

  std::ostringstream json;
  json << std::setprecision(17)
       << "{\n"
       << "  \"snapshot_collection_path\": \""
       << json_escape(result.snapshot_collection_path) << "\",\n"
       << "  \"snapshots\": " << result.snapshots << ",\n"
       << "  \"configured_nullspace\": \"" << json_escape(result.configured_nullspace) << "\",\n"
       << "  \"field_nullspace_index\": " << result.field_nullspace_index << ",\n"
       << "  \"field_nullspace_block_size\": " << result.field_nullspace_block_size << ",\n"
       << "  \"rhs_compatibility_tolerance\": "
       << result.rhs_compatibility_tolerance << ",\n"
       << "  \"nullspace_residual_tolerance\": "
       << result.nullspace_residual_tolerance << ",\n"
       << "  \"declared_petsc_nullspace_count\": "
       << result.declared_petsc_nullspace_count << ",\n"
       << "  \"nullspace_check_count\": " << result.nullspace_check_count << ",\n"
       << "  \"all_checked_candidates_are_right_nullspaces\": "
       << (result.all_checked_candidates_are_right_nullspaces ? "true" : "false") << ",\n"
       << "  \"all_checked_candidates_are_transpose_nullspaces\": "
       << (result.all_checked_candidates_are_transpose_nullspaces ? "true" : "false") << ",\n"
       << "  \"all_rhs_orthogonal_to_checked_candidates\": "
       << (result.all_rhs_orthogonal_to_checked_candidates ? "true" : "false") << ",\n"
       << "  \"all_rhs_compatible_with_checked_left_nullspaces\": "
       << (result.all_rhs_compatible_with_checked_left_nullspaces ? "true" : "false") << ",\n"
       << "  \"max_rhs_nullspace_component_relative_norm\": "
       << result.max_rhs_nullspace_component_relative_norm << ",\n"
       << "  \"max_right_residual_relative_to_matrix_norm\": "
       << result.max_right_residual_relative_to_matrix_norm << ",\n"
       << "  \"max_transpose_residual_relative_to_matrix_norm\": "
       << result.max_transpose_residual_relative_to_matrix_norm << ",\n"
       << "  \"note\": \"PETSc binary matrix dumps do not serialize MatNullSpace basis "
          "vectors. KSPTune records whether a snapshot declared a PETSc nullspace and "
          "tests explicit candidate nullspaces with PETSc.\",\n"
       << "  \"snapshot_results\": [\n";

  for(size_t snapshot_index = 0; snapshot_index < result.snapshot_results.size();
      ++snapshot_index) {
    const snapshot_analysis& snapshot_result = result.snapshot_results[snapshot_index];
    const matrix_diagnostics& matrix = snapshot_result.matrix;
    json << "    {\n"
         << "      \"solve_index\": " << snapshot_result.solve_index << ",\n"
         << "      \"declared_petsc_nullspace\": "
         << (snapshot_result.declared_petsc_nullspace ? "true" : "false") << ",\n"
         << "      \"matrix_nullspace_attached\": "
         << (snapshot_result.matrix_nullspace_attached ? "true" : "false") << ",\n"
         << "      \"transpose_nullspace_attached\": "
         << (snapshot_result.transpose_nullspace_attached ? "true" : "false") << ",\n"
         << "      \"near_nullspace_attached\": "
         << (snapshot_result.near_nullspace_attached ? "true" : "false") << ",\n"
         << "      \"nullspace_kind\": \""
         << json_escape(snapshot_result.nullspace_kind) << "\",\n"
         << "      \"rhs_norm\": " << snapshot_result.rhs_norm << ",\n"
         << "      \"matrix\": {"
         << "\"rows\": " << matrix.row_count << ", "
         << "\"cols\": " << matrix.column_count << ", "
         << "\"local_rows\": " << matrix.local_row_count << ", "
         << "\"local_cols\": " << matrix.local_column_count << ", "
         << "\"nonzeros_used\": " << matrix.nonzero_count << ", "
         << "\"nonzeros_allocated\": " << matrix.allocated_nonzero_count << ", "
         << "\"memory_bytes\": " << matrix.petsc_matrix_memory_bytes << ", "
         << "\"frobenius_norm\": " << matrix.frobenius_norm << ", "
         << "\"symmetric\": " << (matrix.symmetric ? "true" : "false") << "},\n"
         << "      \"nullspaces\": [\n";

    for(size_t check_index = 0; check_index < snapshot_result.nullspaces.size();
        ++check_index) {
      const nullspace_diagnostics& diagnostics = snapshot_result.nullspaces[check_index];
      const nullspace_candidate& candidate = diagnostics.candidate;
      json << "        {"
           << "\"label\": \"" << json_escape(candidate.label) << "\", "
           << "\"kind\": \"" << json_escape(candidate.kind) << "\", "
           << "\"field_index\": " << candidate.field_index << ", "
           << "\"block_size\": " << candidate.block_size << ", "
           << "\"requested_by_user\": "
           << (candidate.requested_by_user ? "true" : "false") << ", "
           << "\"declared_by_snapshot_metadata\": "
           << (candidate.declared_by_snapshot_metadata ? "true" : "false") << ", "
           << "\"right_is_nullspace_by_petsc_test\": "
           << (diagnostics.right_is_nullspace_by_petsc_test ? "true" : "false") << ", "
           << "\"right_is_nullspace_by_residual\": "
           << (diagnostics.right_is_nullspace_by_residual ? "true" : "false") << ", "
           << "\"right_residual_norm\": " << diagnostics.right_residual_norm << ", "
           << "\"right_residual_relative_to_matrix_norm\": "
           << diagnostics.right_residual_relative_to_matrix_norm << ", "
           << "\"transpose_is_nullspace_by_residual\": "
           << (diagnostics.transpose_is_nullspace_by_residual ? "true" : "false") << ", "
           << "\"transpose_residual_norm\": " << diagnostics.transpose_residual_norm << ", "
           << "\"transpose_residual_relative_to_matrix_norm\": "
           << diagnostics.transpose_residual_relative_to_matrix_norm << ", "
           << "\"rhs_norm\": " << diagnostics.rhs_norm << ", "
           << "\"rhs_nullspace_component_norm\": "
           << diagnostics.rhs_nullspace_component_norm << ", "
           << "\"rhs_nullspace_component_relative_norm\": "
           << diagnostics.rhs_nullspace_component_relative_norm << ", "
           << "\"rhs_norm_after_nullspace_removal\": "
           << diagnostics.rhs_norm_after_nullspace_removal << ", "
           << "\"rhs_has_nullspace_content\": "
           << (diagnostics.rhs_has_nullspace_content ? "true" : "false") << ", "
           << "\"rhs_is_orthogonal_to_candidate\": "
           << (diagnostics.rhs_is_orthogonal_to_candidate ? "true" : "false") << ", "
           << "\"rhs_compatibility_check_applies\": "
           << (diagnostics.rhs_compatibility_check_applies ? "true" : "false") << ", "
           << "\"rhs_is_compatible_with_left_nullspace\": "
           << (diagnostics.rhs_is_compatible_with_left_nullspace ? "true" : "false")
           << "}";
      if(check_index + 1 < snapshot_result.nullspaces.size()) json << ",";
      json << "\n";
    }

    json << "      ]\n"
         << "    }";
    if(snapshot_index + 1 < result.snapshot_results.size()) json << ",";
    json << "\n";
  }

  json << "  ]\n"
       << "}\n";

  if(path.empty()) {
    std::cout << json.str();
  } else {
    std::ofstream output(path.c_str());
    output << json.str();
  }
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
  if(args.solve_target >= 0) {
    const int position = find_snapshot_by_solve_index(snapshots, args.solve_target);
    if(position < 0) {
      PetscFPrintf(
          PETSC_COMM_WORLD,
          stderr,
          "No snapshot with solve_index %d\n",
          args.solve_target);
      return PETSC_ERR_ARG_WRONG;
    }
    snapshots = {snapshots[static_cast<size_t>(position)]};
  }

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

  write_json_result(result, args.json_output_path);
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
