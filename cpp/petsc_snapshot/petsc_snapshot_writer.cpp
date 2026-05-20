#include "ksptune/petsc_snapshot_writer.hpp"

#include <petscsys.h>
#include <petscviewer.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ksptune {
namespace petsc_snapshot {
namespace {

std::string trim_whitespace(const std::string& text)
{
  const auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char value) {
    return std::isspace(value) != 0;
  });
  const auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char value) {
    return std::isspace(value) != 0;
  }).base();
  if(begin >= end) return "";
  return std::string(begin, end);
}

std::string lowercase_text(std::string text)
{
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char value) {
    return char(std::tolower(value));
  });
  return text;
}

std::string zero_padded_snapshot_index(int snapshot_index)
{
  std::ostringstream stream;
  stream << std::setw(6) << std::setfill('0') << snapshot_index;
  return stream.str();
}

bool path_is_absolute(const std::string& path)
{
  return !path.empty() && path[0] == '/';
}

std::string normalize_path_text(const std::string& path)
{
  const bool absolute_path = path_is_absolute(path);
  std::vector<std::string> parts;
  std::stringstream stream(path);
  std::string part;
  while(std::getline(stream, part, '/')) {
    if(part.empty() || part == ".") continue;
    if(part == "..") {
      if(!parts.empty() && parts.back() != "..") {
        parts.pop_back();
      } else if(!absolute_path) {
        parts.push_back(part);
      }
    } else {
      parts.push_back(part);
    }
  }

  std::ostringstream normalized_path;
  if(absolute_path) normalized_path << '/';
  for(size_t index = 0; index < parts.size(); ++index) {
    if(index > 0) normalized_path << '/';
    normalized_path << parts[index];
  }
  const std::string normalized = normalized_path.str();
  if(normalized.empty()) return absolute_path ? "/" : ".";
  return normalized;
}

PetscErrorCode absolute_path_from_user_path(const std::string& path, std::string* absolute_path)
{
  if(path_is_absolute(path)) {
    *absolute_path = normalize_path_text(path);
    return 0;
  }

  char current_working_directory[PATH_MAX];
  if(getcwd(current_working_directory, sizeof(current_working_directory)) == nullptr) {
    return PETSC_ERR_FILE_OPEN;
  }
  *absolute_path = normalize_path_text(
      std::string(current_working_directory) + "/" + path);
  return 0;
}

std::string join_paths(const std::string& directory, const std::string& file_name)
{
  if(directory.empty() || directory == ".") return file_name;
  if(directory == "/") return "/" + file_name;
  return directory + "/" + file_name;
}

std::string file_name_from_path(const std::string& path)
{
  std::string trimmed_path = path;
  while(trimmed_path.size() > 1 &&
        (trimmed_path.back() == '/' || trimmed_path.back() == '\\')) {
    trimmed_path.pop_back();
  }
  const std::string::size_type separator = trimmed_path.find_last_of("/\\");
  if(separator == std::string::npos) return trimmed_path;
  return trimmed_path.substr(separator + 1);
}

std::string snapshot_output_file_path(const std::string& output_directory,
                                      const std::string& snapshot_file_prefix,
                                      int snapshot_index,
                                      const char* suffix)
{
  return join_paths(
      output_directory,
      snapshot_file_prefix + "__solve_" + zero_padded_snapshot_index(snapshot_index) + suffix);
}

PetscErrorCode write_matrix_to_petsc_binary_file(MPI_Comm mpi_communicator,
                                                 Mat matrix,
                                                 const std::string& file_path)
{
  PetscViewer viewer = nullptr;
  PetscErrorCode error = PetscViewerBinaryOpen(
      mpi_communicator,
      file_path.c_str(),
      FILE_MODE_WRITE,
      &viewer);
  if(error != 0) return error;
  error = MatView(matrix, viewer);
  if(error == 0) error = PetscViewerDestroy(&viewer);
  else PetscViewerDestroy(&viewer);
  return error;
}

PetscErrorCode write_vector_to_petsc_binary_file(MPI_Comm mpi_communicator,
                                                 Vec vector,
                                                 const std::string& file_path)
{
  PetscViewer viewer = nullptr;
  PetscErrorCode error = PetscViewerBinaryOpen(
      mpi_communicator,
      file_path.c_str(),
      FILE_MODE_WRITE,
      &viewer);
  if(error != 0) return error;
  error = VecView(vector, viewer);
  if(error == 0) error = PetscViewerDestroy(&viewer);
  else PetscViewerDestroy(&viewer);
  return error;
}

PetscErrorCode gather_parallel_ownership_range_endpoints_on_rank_zero(
    MPI_Comm mpi_communicator,
    PetscInt local_start,
    PetscInt local_end,
    std::vector<long long>* ownership_range_endpoints)
{
  int mpi_rank = 0;
  int mpi_size = 1;
  int mpi_error = MPI_Comm_rank(mpi_communicator, &mpi_rank);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;
  mpi_error = MPI_Comm_size(mpi_communicator, &mpi_size);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;

  const long long local_start_as_integer = static_cast<long long>(local_start);
  const long long local_end_as_integer = static_cast<long long>(local_end);
  std::vector<long long> local_starts(mpi_rank == 0 ? static_cast<size_t>(mpi_size) : 0);
  std::vector<long long> local_ends(mpi_rank == 0 ? static_cast<size_t>(mpi_size) : 0);
  mpi_error = MPI_Gather(&local_start_as_integer,
                         1,
                         MPI_LONG_LONG,
                         mpi_rank == 0 ? local_starts.data() : nullptr,
                         1,
                         MPI_LONG_LONG,
                         0,
                         mpi_communicator);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;
  mpi_error = MPI_Gather(&local_end_as_integer,
                         1,
                         MPI_LONG_LONG,
                         mpi_rank == 0 ? local_ends.data() : nullptr,
                         1,
                         MPI_LONG_LONG,
                         0,
                         mpi_communicator);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;

  ownership_range_endpoints->clear();
  if(mpi_rank == 0) {
    ownership_range_endpoints->resize(static_cast<size_t>(mpi_size) + 1);
    (*ownership_range_endpoints)[0] = local_starts[0];
    for(int rank_index = 0; rank_index < mpi_size; ++rank_index) {
      (*ownership_range_endpoints)[static_cast<size_t>(rank_index) + 1] =
          local_ends[static_cast<size_t>(rank_index)];
    }
  }
  return 0;
}

void write_metadata_list_line(std::ofstream& output,
                              const std::string& key,
                              const std::vector<long long>& values)
{
  output << key << '=';
  for(size_t index = 0; index < values.size(); ++index) {
    if(index > 0) output << ',';
    output << values[index];
  }
  output << '\n';
}

bool directory_exists(const std::string& path)
{
  struct stat path_status;
  return stat(path.c_str(), &path_status) == 0 && S_ISDIR(path_status.st_mode);
}

PetscErrorCode create_directory_tree(const std::string& directory)
{
  const std::string normalized_directory = normalize_path_text(directory);
  if(normalized_directory.empty() || normalized_directory == ".") return 0;
  if(directory_exists(normalized_directory)) return 0;

  std::string current_directory = path_is_absolute(normalized_directory) ? "/" : "";
  std::stringstream stream(normalized_directory);
  std::string part;
  while(std::getline(stream, part, '/')) {
    if(part.empty()) continue;
    current_directory = current_directory.empty()
        ? part
        : join_paths(current_directory, part);
    if(directory_exists(current_directory)) continue;
    if(mkdir(current_directory.c_str(), 0777) != 0 && errno != EEXIST) {
      return PETSC_ERR_FILE_WRITE;
    }
  }
  return 0;
}

PetscErrorCode create_output_directory_on_rank_zero(MPI_Comm mpi_communicator,
                                                    const std::string& output_directory)
{
  int mpi_rank = 0;
  int mpi_error = MPI_Comm_rank(mpi_communicator, &mpi_rank);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;

  if(mpi_rank == 0) {
    PetscErrorCode error = create_directory_tree(output_directory);
    if(error != 0) return error;
  }
  mpi_error = MPI_Barrier(mpi_communicator);
  return mpi_error == MPI_SUCCESS ? 0 : PETSC_ERR_MPI;
}

PetscErrorCode validate_snapshot_request(const KspSnapshotRequest& request)
{
  if(request.system_matrix == nullptr) return PETSC_ERR_ARG_NULL;
  if(request.right_hand_side_vector == nullptr) return PETSC_ERR_ARG_NULL;
  if(request.initial_guess_vector == nullptr) return PETSC_ERR_ARG_NULL;
  return 0;
}

int next_requested_snapshot_solve_index()
{
  static int next_solve_index = 0;
  const int current_solve_index = next_solve_index;
  ++next_solve_index;
  return current_solve_index;
}

}  // namespace

static std::string safe_snapshot_file_prefix_or_empty(const std::string& descriptive_name)
{
  std::string safe_prefix;
  safe_prefix.reserve(descriptive_name.size());

  bool previous_character_was_separator = true;
  for(unsigned char character : descriptive_name) {
    if(std::isalnum(character) != 0) {
      safe_prefix += char(std::tolower(character));
      previous_character_was_separator = false;
    } else if(!previous_character_was_separator) {
      safe_prefix += '_';
      previous_character_was_separator = true;
    }
  }

  while(!safe_prefix.empty() && safe_prefix.back() == '_') {
    safe_prefix.pop_back();
  }

  return safe_prefix;
}

std::string SafeSnapshotFilePrefix(const std::string& descriptive_name)
{
  const std::string safe_prefix = safe_snapshot_file_prefix_or_empty(descriptive_name);
  return safe_prefix.empty() ? "ksp" : safe_prefix;
}

std::string SafeSnapshotFilePrefixFromPath(const std::string& path)
{
  return SafeSnapshotFilePrefix(file_name_from_path(path));
}

bool ShouldWriteSnapshotForStep(const std::string& selected_steps, int snapshot_index)
{
  const std::string normalized_selection = lowercase_text(trim_whitespace(selected_steps));
  if(normalized_selection.empty() || normalized_selection == "all" || normalized_selection == "*") {
    return true;
  }

  std::stringstream comma_separated_steps(selected_steps);
  std::string selected_step_text;
  while(std::getline(comma_separated_steps, selected_step_text, ',')) {
    std::stringstream selected_step_stream(trim_whitespace(selected_step_text));
    int selected_snapshot_index = -1;
    selected_step_stream >> selected_snapshot_index;
    if(!selected_step_stream.fail() && selected_snapshot_index == snapshot_index) {
      return true;
    }
  }
  return false;
}

PetscErrorCode WriteKspSnapshot(const KspSnapshotRequest& request, KspSnapshotResult* result)
{
  PetscErrorCode error = validate_snapshot_request(request);
  if(error != 0) return error;

  const MPI_Comm mpi_communicator = request.mpi_communicator;
  const std::string output_directory =
      request.output_directory.empty() ? "." : request.output_directory;
  std::string absolute_output_directory;
  error = absolute_path_from_user_path(output_directory, &absolute_output_directory);
  if(error != 0) return error;

  error = create_output_directory_on_rank_zero(mpi_communicator, absolute_output_directory);
  if(error != 0) return error;

  const std::string snapshot_file_prefix = SafeSnapshotFilePrefix(request.snapshot_file_prefix);

  const std::string matrix_file_path = snapshot_output_file_path(
      absolute_output_directory,
      snapshot_file_prefix,
      request.snapshot_index,
      "__A.bin");
  const std::string right_hand_side_file_path = snapshot_output_file_path(
      absolute_output_directory,
      snapshot_file_prefix,
      request.snapshot_index,
      "__b.bin");
  const std::string initial_guess_file_path = snapshot_output_file_path(
      absolute_output_directory,
      snapshot_file_prefix,
      request.snapshot_index,
      "__x0.bin");
  const std::string metadata_file_path = snapshot_output_file_path(
      absolute_output_directory,
      snapshot_file_prefix,
      request.snapshot_index,
      "__metadata.txt");

  error = write_matrix_to_petsc_binary_file(
      mpi_communicator,
      request.system_matrix,
      matrix_file_path);
  if(error != 0) return error;
  error = write_vector_to_petsc_binary_file(
      mpi_communicator,
      request.right_hand_side_vector,
      right_hand_side_file_path);
  if(error != 0) return error;
  error = write_vector_to_petsc_binary_file(
      mpi_communicator,
      request.initial_guess_vector,
      initial_guess_file_path);
  if(error != 0) return error;

  PetscInt matrix_global_rows = 0;
  PetscInt matrix_global_columns = 0;
  PetscInt right_hand_side_global_size = 0;
  PetscInt matrix_local_row_start = 0;
  PetscInt matrix_local_row_end = 0;
  PetscInt right_hand_side_local_start = 0;
  PetscInt right_hand_side_local_end = 0;
  PetscInt initial_guess_local_start = 0;
  PetscInt initial_guess_local_end = 0;
  error = MatGetSize(request.system_matrix, &matrix_global_rows, &matrix_global_columns);
  if(error != 0) return error;
  error = MatGetOwnershipRange(
      request.system_matrix,
      &matrix_local_row_start,
      &matrix_local_row_end);
  if(error != 0) return error;
  error = VecGetSize(request.right_hand_side_vector, &right_hand_side_global_size);
  if(error != 0) return error;
  error = VecGetOwnershipRange(
      request.right_hand_side_vector,
      &right_hand_side_local_start,
      &right_hand_side_local_end);
  if(error != 0) return error;
  error = VecGetOwnershipRange(
      request.initial_guess_vector,
      &initial_guess_local_start,
      &initial_guess_local_end);
  if(error != 0) return error;

  std::vector<long long> matrix_row_ownership_ranges;
  std::vector<long long> right_hand_side_ownership_ranges;
  std::vector<long long> initial_guess_ownership_ranges;
  error = gather_parallel_ownership_range_endpoints_on_rank_zero(
      mpi_communicator,
      matrix_local_row_start,
      matrix_local_row_end,
      &matrix_row_ownership_ranges);
  if(error != 0) return error;
  error = gather_parallel_ownership_range_endpoints_on_rank_zero(
      mpi_communicator,
      right_hand_side_local_start,
      right_hand_side_local_end,
      &right_hand_side_ownership_ranges);
  if(error != 0) return error;
  error = gather_parallel_ownership_range_endpoints_on_rank_zero(
      mpi_communicator,
      initial_guess_local_start,
      initial_guess_local_end,
      &initial_guess_ownership_ranges);
  if(error != 0) return error;

  int mpi_rank = 0;
  int mpi_size = 1;
  int mpi_error = MPI_Comm_rank(mpi_communicator, &mpi_rank);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;
  mpi_error = MPI_Comm_size(mpi_communicator, &mpi_size);
  if(mpi_error != MPI_SUCCESS) return PETSC_ERR_MPI;

  MatNullSpace matrix_nullspace = nullptr;
  error = MatGetNullSpace(request.system_matrix, &matrix_nullspace);
  if(error != 0) return error;
  MatNullSpace transpose_nullspace = nullptr;
  error = MatGetTransposeNullSpace(request.system_matrix, &transpose_nullspace);
  if(error != 0) return error;
  MatNullSpace near_nullspace = nullptr;
  error = MatGetNearNullSpace(request.system_matrix, &near_nullspace);
  if(error != 0) return error;

  const bool matrix_nullspace_attached = matrix_nullspace != nullptr;
  const bool transpose_nullspace_attached = transpose_nullspace != nullptr;
  const bool near_nullspace_attached = near_nullspace != nullptr;
  const bool any_nullspace_attached =
      matrix_nullspace_attached || transpose_nullspace_attached || near_nullspace_attached;
  const std::string nullspace_kind = request.nullspace_metadata.kind.empty()
      ? (any_nullspace_attached ? "unknown" : "none")
      : request.nullspace_metadata.kind;

  if(mpi_rank == 0) {
    std::ofstream metadata(metadata_file_path);
    if(!metadata.good()) return PETSC_ERR_FILE_WRITE;
    metadata << std::setprecision(17)
             << "solve_index=" << request.snapshot_index << '\n'
             << "snapshot_file_prefix=" << snapshot_file_prefix << '\n'
             << "A=" << file_name_from_path(matrix_file_path) << '\n'
             << "b=" << file_name_from_path(right_hand_side_file_path) << '\n'
             << "x0=" << file_name_from_path(initial_guess_file_path) << '\n'
             << "rows=" << matrix_global_rows << '\n'
             << "cols=" << matrix_global_columns << '\n'
             << "rhs_size=" << right_hand_side_global_size << '\n'
             << "mpi_size=" << mpi_size << '\n'
             << "matrix_nullspace_attached=" << (matrix_nullspace_attached ? 1 : 0) << '\n'
             << "transpose_nullspace_attached=" << (transpose_nullspace_attached ? 1 : 0) << '\n'
             << "near_nullspace_attached=" << (near_nullspace_attached ? 1 : 0) << '\n'
             << "nullspace_kind=" << nullspace_kind << '\n'
             << "nullspace_field_index=" << request.nullspace_metadata.field_index << '\n'
             << "nullspace_block_size=" << request.nullspace_metadata.block_size << '\n'
             << "rhs_nullspace_component_removed="
             << request.nullspace_metadata.right_hand_side_component_removed << '\n';
    write_metadata_list_line(metadata, "row_ownership_ranges", matrix_row_ownership_ranges);
    write_metadata_list_line(
        metadata,
        "rhs_ownership_ranges",
        right_hand_side_ownership_ranges);
    write_metadata_list_line(
        metadata,
        "x0_ownership_ranges",
        initial_guess_ownership_ranges);
    for(const auto& entry : request.extra_metadata_entries) {
      metadata << entry.first << '=' << entry.second << '\n';
    }
  }

  if(result != nullptr) {
    result->matrix_file_path = matrix_file_path;
    result->right_hand_side_file_path = right_hand_side_file_path;
    result->initial_guess_file_path = initial_guess_file_path;
    result->metadata_file_path = metadata_file_path;
  }
  return 0;
}

PetscErrorCode WritePetscKspSnapshotIfRequested(
    const PetscKspSnapshotExport& snapshot_export,
    KspSnapshotResult* result)
{
  if(snapshot_export.snapshot_directory.empty()) return 0;

  const int snapshot_solve_index = next_requested_snapshot_solve_index();
  const std::string selected_solve_indices_text = snapshot_export.solve_indices.empty()
      ? "0"
      : snapshot_export.solve_indices;
  if(!ShouldWriteSnapshotForStep(selected_solve_indices_text, snapshot_solve_index)) {
    return 0;
  }

  KspSnapshotRequest request;
  request.mpi_communicator = snapshot_export.mpi_communicator;
  request.output_directory = snapshot_export.snapshot_directory;
  request.snapshot_file_prefix = SafeSnapshotFilePrefix(snapshot_export.snapshot_file_prefix);
  request.snapshot_index = snapshot_solve_index;
  request.system_matrix = snapshot_export.system_matrix;
  request.right_hand_side_vector = snapshot_export.right_hand_side_vector;
  request.initial_guess_vector = snapshot_export.initial_guess_vector;
  request.nullspace_metadata = snapshot_export.nullspace_metadata;
  request.extra_metadata_entries = snapshot_export.metadata_entries;

  return WriteKspSnapshot(request, result);
}

}  // namespace petsc_snapshot
}  // namespace ksptune
