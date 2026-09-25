#include "snapshot_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

namespace ksptune {
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

bool is_absolute_path(const std::string& path)
{
  return !path.empty() && (path[0] == '/' || (path.size() > 2 && path[1] == ':'));
}

bool file_exists(const std::string& path)
{
  std::ifstream input(path.c_str());
  return input.good();
}

std::string dirname(const std::string& path)
{
  const std::string::size_type pos = path.find_last_of("/\\");
  if(pos == std::string::npos) return ".";
  if(pos == 0) return path.substr(0, 1);
  return path.substr(0, pos);
}

std::string join_relative_to(const std::string& base_directory, const std::string& path)
{
  if(path.empty() || is_absolute_path(path)) return path;
  if(base_directory.empty() || base_directory == ".") return path;
  return base_directory + "/" + path;
}

std::vector<PetscInt> parse_petsc_int_list(const std::string& text)
{
  std::vector<PetscInt> values;
  std::stringstream stream(text);
  std::string token;
  while(std::getline(stream, token, ',')) {
    token = trim(token);
    if(!token.empty()) values.push_back(static_cast<PetscInt>(std::stoll(token)));
  }
  return values;
}

bool parse_bool(const std::string& text)
{
  std::string value = trim(text);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

int parse_int_or_default(const std::string& text, int default_value)
{
  if(trim(text).empty()) return default_value;
  return std::atoi(text.c_str());
}

PetscInt parse_petsc_int_or_default(const std::string& text, PetscInt default_value)
{
  if(trim(text).empty()) return default_value;
  return static_cast<PetscInt>(std::stoll(text));
}

double parse_double_or_default(const std::string& text, double default_value)
{
  if(trim(text).empty()) return default_value;
  return std::atof(text.c_str());
}

void read_snapshot_metadata(snapshot& snap)
{
  if(snap.metadata_file_path.empty()) return;

  std::ifstream input(snap.metadata_file_path.c_str());
  std::string line;
  while(std::getline(input, line)) {
    const size_t equals = line.find('=');
    if(equals == std::string::npos) continue;
    const std::string key = trim(line.substr(0, equals));
    const std::string value = trim(line.substr(equals + 1));
    if(key == "snapshot_id") snap.snapshot_id = value;
    else if(key == "row_ownership_ranges") snap.row_ownership_ranges = parse_petsc_int_list(value);
    else if(key == "rhs_ownership_ranges") snap.rhs_ownership_ranges = parse_petsc_int_list(value);
    else if(key == "x0_ownership_ranges") snap.x0_ownership_ranges = parse_petsc_int_list(value);
    else if(key == "relative_tolerance") snap.relative_tolerance = std::atof(value.c_str());
    else if(key == "absolute_tolerance") snap.absolute_tolerance = std::atof(value.c_str());
    else if(key == "divergence_tolerance") snap.divergence_tolerance = std::atof(value.c_str());
    else if(key == "max_iterations") snap.max_iterations = std::atoi(value.c_str());
    else if(key == "symmetric") snap.symmetric = parse_bool(value);
    else if(key == "spd") snap.spd = parse_bool(value);
    else if(key == "mpi_size") snap.mpi_size = parse_int_or_default(value, snap.mpi_size);
    else if(key == "matrix_nullspace_attached") snap.matrix_nullspace_attached = parse_bool(value);
    else if(key == "transpose_nullspace_attached") {
      snap.transpose_nullspace_attached = parse_bool(value);
    }
    else if(key == "near_nullspace_attached") snap.near_nullspace_attached = parse_bool(value);
    else if(key == "nullspace_kind") snap.nullspace_kind = value.empty() ? "none" : value;
    else if(key == "nullspace_field_index") snap.nullspace_field_index = std::atoi(value.c_str());
    else if(key == "nullspace_block_size") snap.nullspace_block_size = std::atoi(value.c_str());
    else if(key == "rhs_nullspace_component_removed") {
      snap.rhs_nullspace_component_removed = value.empty() ? "unknown" : value;
    }
    else if(key == "rows") snap.rows = parse_petsc_int_or_default(value, snap.rows);
    else if(key == "cols") snap.cols = parse_petsc_int_or_default(value, snap.cols);
    else if(key == "rhs_size") snap.right_hand_side_size =
        parse_petsc_int_or_default(value, snap.right_hand_side_size);
  }
}

std::string field_or_empty(const std::map<std::string, std::string>& row, const std::string& key)
{
  const auto found = row.find(key);
  return found == row.end() ? "" : found->second;
}

snapshot snapshot_from_named_row(const std::map<std::string, std::string>& row,
                                 const std::string& base_directory)
{
  snapshot snap;
  snap.snapshot_id = field_or_empty(row, "snapshot_id");
  snap.solve_index = parse_int_or_default(field_or_empty(row, "solve_index"), 0);
  snap.matrix_file_path = join_relative_to(
      base_directory,
      field_or_empty(row, "A"));
  snap.right_hand_side_file_path = join_relative_to(
      base_directory,
      field_or_empty(row, "b"));
  snap.initial_guess_file_path = join_relative_to(
      base_directory,
      field_or_empty(row, "x0"));
  snap.metadata_file_path = join_relative_to(
      base_directory,
      field_or_empty(row, "meta"));
  snap.rows = parse_petsc_int_or_default(field_or_empty(row, "rows"), 0);
  snap.cols = parse_petsc_int_or_default(field_or_empty(row, "cols"), snap.rows);
  snap.right_hand_side_size = parse_petsc_int_or_default(
      field_or_empty(row, "rhs_size"),
      0);
  snap.mpi_size = parse_int_or_default(field_or_empty(row, "mpi_size"), 1);
  snap.symmetric = parse_bool(field_or_empty(row, "symmetric"));
  snap.spd = parse_bool(field_or_empty(row, "spd"));
  snap.matrix_nullspace_attached =
      parse_bool(field_or_empty(row, "matrix_nullspace_attached"));
  snap.transpose_nullspace_attached = parse_bool(field_or_empty(row, "transpose_nullspace_attached"));
  snap.near_nullspace_attached = parse_bool(field_or_empty(row, "near_nullspace_attached"));
  snap.nullspace_kind = field_or_empty(row, "nullspace_kind");
  if(snap.nullspace_kind.empty()) snap.nullspace_kind = snap.matrix_nullspace_attached ? "unknown" : "none";
  snap.nullspace_field_index = parse_int_or_default(field_or_empty(row, "nullspace_field_index"), -1);
  snap.nullspace_block_size = parse_int_or_default(field_or_empty(row, "nullspace_block_size"), 0);
  snap.rhs_nullspace_component_removed =
      field_or_empty(row, "rhs_nullspace_component_removed");
  if(snap.rhs_nullspace_component_removed.empty()) snap.rhs_nullspace_component_removed = "unknown";
  read_snapshot_metadata(snap);
  snap.nullspace =
      snap.matrix_nullspace_attached || snap.transpose_nullspace_attached || snap.near_nullspace_attached;
  if(snap.nullspace_kind.empty() || (snap.nullspace && snap.nullspace_kind == "none")) {
    snap.nullspace_kind = snap.nullspace ? "unknown" : "none";
  }
  return snap;
}

std::vector<snapshot> read_snapshot_collection(const std::string& path)
{
  std::ifstream input(path.c_str());
  std::vector<snapshot> snapshots;
  const std::string base_directory = dirname(path);
  std::string line;

  if(!std::getline(input, line)) return snapshots;
  const std::vector<std::string> header = split_csv_line(line);
  for(const char* column : {"solve_index", "A", "matrix_metadata", "b", "x0", "meta"}) {
    if(std::find(header.begin(), header.end(), column) == header.end()) return snapshots;
  }

  while(std::getline(input, line)) {
    if(trim(line).empty()) continue;
    const std::vector<std::string> fields = split_csv_line(line);
    std::map<std::string, std::string> row;
    for(size_t i = 0; i < header.size() && i < fields.size(); ++i) {
      row[header[i]] = fields[i];
    }
    snapshots.push_back(snapshot_from_named_row(row, base_directory));
  }

  std::sort(snapshots.begin(), snapshots.end(), [](const snapshot& left, const snapshot& right) {
    return left.solve_index < right.solve_index;
  });
  return snapshots;
}

PetscErrorCode select_snapshot(std::vector<snapshot>& snapshots, int solve_index,
                               const std::string& snapshot_id)
{
  if(solve_index < 0 && snapshot_id.empty()) return 0;
  PetscCheck(solve_index < 0 || snapshot_id.empty(), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Select either snapshot_id or solve_index, not both");
  std::vector<snapshot> matches;
  for(const snapshot& snap : snapshots) {
    if(snapshot_id.empty() ? snap.solve_index == solve_index : snap.snapshot_id == snapshot_id) {
      matches.push_back(snap);
    }
  }
  PetscCheck(matches.size() == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Expected one snapshot for snapshot_id='%s', solve_index=%d; found %zu. Use a unique snapshot_id.",
             snapshot_id.c_str(), solve_index, matches.size());
  snapshots = std::move(matches);
  return 0;
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

PetscErrorCode load_matrix_binary(const snapshot& snap, Mat* matrix)
{
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(
      PETSC_COMM_WORLD,
      snap.matrix_file_path.c_str(),
      FILE_MODE_READ,
      &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, matrix));
  if(ownership_ranges_match_current(snap.row_ownership_ranges)) {
    const PetscInt local_rows = local_size_from_ranges(snap.row_ownership_ranges);
    PetscCall(MatSetSizes(*matrix, local_rows, local_rows, snap.rows, snap.cols));
    PetscCall(MatSetType(*matrix, MATAIJ));
  }
  PetscCall(MatLoad(*matrix, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  return 0;
}

PetscErrorCode load_vector_binary(const std::string& path,
                                  const std::vector<PetscInt>& ownership_ranges,
                                  PetscInt global_size,
                                  Vec* vector)
{
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(
      PETSC_COMM_WORLD,
      path.c_str(),
      FILE_MODE_READ,
      &viewer));
  PetscCall(VecCreate(PETSC_COMM_WORLD, vector));
  if(ownership_ranges_match_current(ownership_ranges)) {
    PetscCall(VecSetSizes(*vector, local_size_from_ranges(ownership_ranges), global_size));
    PetscCall(VecSetType(*vector, VECMPI));
  }
  PetscCall(VecLoad(*vector, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  return 0;
}

}  // namespace ksptune
