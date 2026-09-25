#pragma once

#include <petscksp.h>
#include <petscsys.h>
#include <petscviewer.h>
#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Call collectively immediately before KSPSolve(ksp, b, x).
 * Settings are read once per KSP from rank zero of its communicator:
 *   KSP_EXPORT_ENABLE: enable export with 1/true/on/yes (default: disabled).
 *   KSPTUNE_SNAPSHOT_DIR: output directory, required when enabled.
 *   KSPTUNE_SNAPSHOT_SOLVES: zero-based indices per KSP, e.g. 0,5,10 or all (default: 0).
 *   KSPTUNE_SNAPSHOT_PREFIX: optional filename prefix (default: ksp).
 * The operator and vectors are left unchanged. If KSP ignores x, x0 is exported as zero.
 * Relative output paths are resolved when settings are first read.
 * Constant nullspaces are recognized; other nullspaces are marked unknown.
 * Errors are returned using PETSc error codes. No separate initialization or cleanup is needed.
 */
inline PetscErrorCode KSPTuneExport(KSP ksp, Vec b, Vec x);

namespace ksptune {
namespace petsc_snapshot {
namespace detail {

inline std::string safe_snapshot_prefix(const std::string& descriptive_name)
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

  return safe_prefix.empty() ? "ksp" : safe_prefix;
}

inline std::string zero_padded_snapshot_index(int snapshot_index)
{
  std::ostringstream stream;
  stream << std::setw(6) << std::setfill('0') << snapshot_index;
  return stream.str();
}

inline bool path_is_absolute(const std::string& path)
{
  return !path.empty() && path[0] == '/';
}

inline std::string normalize_path_text(const std::string& path)
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

inline PetscErrorCode absolute_path_from_user_path(const std::string& path, std::string* absolute_path)
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

inline std::string join_paths(const std::string& directory, const std::string& file_name)
{
  if(directory.empty() || directory == ".") return file_name;
  if(directory == "/") return "/" + file_name;
  return directory + "/" + file_name;
}

inline bool directory_exists(const std::string& path)
{
  struct stat path_status;
  return stat(path.c_str(), &path_status) == 0 && S_ISDIR(path_status.st_mode);
}

inline PetscErrorCode create_directory_tree(const std::string& directory)
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
    if(mkdir(current_directory.c_str(), 0777) != 0 &&
       (errno != EEXIST || !directory_exists(current_directory))) {
      return PETSC_ERR_FILE_WRITE;
    }
  }
  return 0;
}

inline PetscErrorCode commit_temporary_file_on_rank_zero(MPI_Comm comm,
                                                       const std::string& temporary_path,
                                                       const std::string& final_path)
{
  int rank = 0;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  int error = 0;
  if(rank == 0 && std::rename(temporary_path.c_str(), final_path.c_str()) != 0) {
    error = PETSC_ERR_FILE_WRITE;
  }
  PetscCallMPI(MPI_Bcast(&error, 1, MPI_INT, 0, comm));
  PetscCheck(!error, comm, PETSC_ERR_FILE_WRITE, "Cannot publish snapshot file: %s", final_path.c_str());
  PetscCallMPI(MPI_Barrier(comm));
  return 0;
}

inline PetscErrorCode write_matrix_to_petsc_binary_file(MPI_Comm comm, Mat matrix,
                                                       const std::string& path)
{
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(comm, path.c_str(), FILE_MODE_WRITE, &viewer));
  PetscErrorCode error = PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE);
  if(!error) error = MatView(matrix, viewer);
  const PetscErrorCode cleanup_error = PetscViewerDestroy(&viewer);
  PetscCall(error);
  PetscCall(cleanup_error);
  return 0;
}

inline PetscErrorCode write_vector_to_petsc_binary_file(MPI_Comm comm, Vec vector,
                                                       const std::string& path)
{
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(comm, path.c_str(), FILE_MODE_WRITE, &viewer));
  PetscErrorCode error = PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE);
  if(!error) error = VecView(vector, viewer);
  const PetscErrorCode cleanup_error = PetscViewerDestroy(&viewer);
  PetscCall(error);
  PetscCall(cleanup_error);
  return 0;
}

inline void write_ownership_ranges(std::ostream& output, const char* key,
                                   const PetscInt* ranges, int mpi_size)
{
  output << key << '=';
  for(int rank = 0; rank <= mpi_size; ++rank) {
    if(rank) output << ',';
    output << ranges[rank];
  }
  output << '\n';
}

struct ExportContext {
  std::string directory;
  std::string prefix;
  std::string snapshot_namespace;
  std::vector<int> solve_indices;
  int next_solve_index = 0;
  int matrix_version = 0;
  PetscObjectId matrix_id = 0;
  PetscObjectState matrix_state = -1;
  PetscObjectId nullspace_ids[3] = {0, 0, 0};
};

inline PetscErrorCode broadcast_string(MPI_Comm comm, std::string& value)
{
  int length = static_cast<int>(value.size());
  PetscCallMPI(MPI_Bcast(&length, 1, MPI_INT, 0, comm));
  value.resize(static_cast<size_t>(length));
  if(length) PetscCallMPI(MPI_Bcast(&value[0], length, MPI_CHAR, 0, comm));
  return 0;
}

inline PetscErrorCode read_environment(MPI_Comm comm, const char* name, std::string& value)
{
  int rank = 0;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if(rank == 0) {
    const char* setting = std::getenv(name);
    value = setting ? setting : "";
  }
  return broadcast_string(comm, value);
}

inline PetscErrorCode initialize_export_context(KSP ksp, ExportContext& context)
{
  const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(ksp));
  std::string enable_setting;
  PetscCall(read_environment(comm, "KSP_EXPORT_ENABLE", enable_setting));
  PetscBool enabled = PETSC_FALSE;
  if(!enable_setting.empty()) PetscCall(PetscOptionsStringToBool(enable_setting.c_str(), &enabled));
  if(!enabled) return 0;
  PetscCall(read_environment(comm, "KSPTUNE_SNAPSHOT_DIR", context.directory));
  PetscCheck(!context.directory.empty(), comm, PETSC_ERR_ARG_WRONG,
             "KSPTUNE_SNAPSHOT_DIR is required when KSP_EXPORT_ENABLE is enabled");

  int rank = 0;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  std::string absolute_directory;
  int path_error = rank == 0 ? absolute_path_from_user_path(context.directory, &absolute_directory) : 0;
  PetscCallMPI(MPI_Bcast(&path_error, 1, MPI_INT, 0, comm));
  PetscCall(path_error);
  PetscCall(broadcast_string(comm, absolute_directory));
  context.directory = absolute_directory;

  std::string selection;
  PetscCall(read_environment(comm, "KSPTUNE_SNAPSHOT_SOLVES", selection));
  if(selection.empty()) selection = "0";
  if(selection != "all" && selection != "*") {
    std::stringstream selections(selection);
    std::string token;
    while(std::getline(selections, token, ',')) {
      std::stringstream input(token);
      int index = -1;
      std::string trailing;
      PetscCheck((input >> index) && index >= 0 && !(input >> trailing), comm,
                 PETSC_ERR_ARG_WRONG, "Invalid KSPTUNE_SNAPSHOT_SOLVES: %s", selection.c_str());
      context.solve_indices.push_back(index);
    }
    PetscCheck(!context.solve_indices.empty() && selection.back() != ',', comm,
               PETSC_ERR_ARG_WRONG, "Invalid KSPTUNE_SNAPSHOT_SOLVES: %s", selection.c_str());
  }
  PetscCall(read_environment(comm, "KSPTUNE_SNAPSHOT_PREFIX", context.prefix));
  context.prefix = safe_snapshot_prefix(context.prefix);

  if(rank == 0) {
    int world_rank = 0;
    PetscObjectId ksp_id = 0;
    PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    PetscCall(PetscObjectGetId(reinterpret_cast<PetscObject>(ksp), &ksp_id));
    const auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    context.snapshot_namespace = context.prefix + "_" + std::to_string(timestamp) + "_" +
        std::to_string(getpid()) + "_" + std::to_string(world_rank) + "_" + std::to_string(ksp_id);
  }
  return broadcast_string(comm, context.snapshot_namespace);
}

inline PetscErrorCode destroy_export_context(void* pointer)
{
  delete static_cast<ExportContext*>(pointer);
  return 0;
}

inline PetscErrorCode get_export_context(KSP ksp, ExportContext** context)
{
  const auto object = reinterpret_cast<PetscObject>(ksp);
  const MPI_Comm comm = PetscObjectComm(object);
  PetscContainer container = nullptr;
  PetscCall(PetscObjectQuery(object, "ksptune_snapshot_export", reinterpret_cast<PetscObject*>(&container)));
  if(container) {
    void* pointer = nullptr;
    PetscCall(PetscContainerGetPointer(container, &pointer));
    *context = static_cast<ExportContext*>(pointer);
    return 0;
  }

  std::unique_ptr<ExportContext> created(new ExportContext);
  PetscCall(initialize_export_context(ksp, *created));
  PetscCall(PetscContainerCreate(comm, &container));
  PetscErrorCode error = PetscContainerSetPointer(container, created.get());
  if(!error) error = PetscContainerSetUserDestroy(container, destroy_export_context);
  if(!error) {
    *context = created.release();
    error = PetscObjectCompose(object, "ksptune_snapshot_export", reinterpret_cast<PetscObject>(container));
  }
  const PetscErrorCode cleanup_error = PetscContainerDestroy(&container);
  return error ? error : cleanup_error;
}

struct SnapshotData {
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = 0, mpi_size = 1;
  Mat matrix = nullptr;
  Vec rhs = nullptr, initial_guess = nullptr;
  PetscInt rows = 0, cols = 0, rhs_size = 0;
  const PetscInt* row_ranges = nullptr;
  const PetscInt* rhs_ranges = nullptr;
  const PetscInt* x0_ranges = nullptr;
  PetscObjectId matrix_id = 0;
  PetscObjectState matrix_state = 0, matrix_nonzero_state = 0;
  MatNullSpace nullspaces[3] = {nullptr, nullptr, nullptr};
  PetscObjectId nullspace_ids[3] = {0, 0, 0};
  const char* nullspace_kind = "none";
  PetscReal rtol = 0, atol = 0, dtol = 0;
  PetscInt max_iterations = 0;
  PetscBool nonzero_guess = PETSC_FALSE;
  KSPType solver_name = nullptr;
  PetscBool symmetric_known = PETSC_FALSE, symmetric = PETSC_FALSE;
  PetscBool spd_known = PETSC_FALSE, spd = PETSC_FALSE;
};

inline PetscErrorCode read_snapshot_data(KSP ksp, Vec b, Vec x, SnapshotData& data)
{
  data.comm = PetscObjectComm(reinterpret_cast<PetscObject>(ksp));
  PetscCallMPI(MPI_Comm_rank(data.comm, &data.rank));
  PetscCallMPI(MPI_Comm_size(data.comm, &data.mpi_size));
  data.rhs = b;
  data.initial_guess = x;
  PetscCall(KSPGetOperators(ksp, &data.matrix, nullptr));
  PetscCall(MatGetSize(data.matrix, &data.rows, &data.cols));
  PetscCall(VecGetSize(b, &data.rhs_size));
  PetscCall(MatGetOwnershipRanges(data.matrix, &data.row_ranges));
  PetscCall(VecGetOwnershipRanges(b, &data.rhs_ranges));
  PetscCall(VecGetOwnershipRanges(x, &data.x0_ranges));
  PetscCall(PetscObjectGetId(reinterpret_cast<PetscObject>(data.matrix), &data.matrix_id));
  PetscCall(MatGetState(data.matrix, &data.matrix_state));
  PetscCall(MatGetNonzeroState(data.matrix, &data.matrix_nonzero_state));
  PetscCall(MatGetNullSpace(data.matrix, &data.nullspaces[0]));
  PetscCall(MatGetTransposeNullSpace(data.matrix, &data.nullspaces[1]));
  PetscCall(MatGetNearNullSpace(data.matrix, &data.nullspaces[2]));
  bool any_nullspace = false, constant_only = true;
  for(int index = 0; index < 3; ++index) {
    if(!data.nullspaces[index]) continue;
    any_nullspace = true;
    PetscBool has_constant = PETSC_FALSE;
    PetscInt vector_count = 0;
    PetscCall(MatNullSpaceGetVecs(data.nullspaces[index], &has_constant, &vector_count, nullptr));
    constant_only = constant_only && has_constant && vector_count == 0;
    PetscCall(PetscObjectGetId(reinterpret_cast<PetscObject>(data.nullspaces[index]), &data.nullspace_ids[index]));
  }
  data.nullspace_kind = any_nullspace ? (constant_only ? "constant" : "unknown") : "none";
  PetscCall(KSPGetTolerances(ksp, &data.rtol, &data.atol, &data.dtol, &data.max_iterations));
  PetscCall(KSPGetInitialGuessNonzero(ksp, &data.nonzero_guess));
  PetscCall(KSPGetType(ksp, &data.solver_name));
  PetscCall(MatIsSymmetricKnown(data.matrix, &data.symmetric_known, &data.symmetric));
  PetscCall(MatIsSPDKnown(data.matrix, &data.spd_known, &data.spd));
  return 0;
}

inline PetscErrorCode write_snapshot(const ExportContext& context, const SnapshotData& data,
                                     int solve_index, bool write_matrix)
{
  const int matrix_version = context.matrix_version + (write_matrix ? 1 : 0);
  const std::string matrix_id = context.snapshot_namespace + "_matrix_" + std::to_string(matrix_version);
  const std::string snapshot_id = context.snapshot_namespace + "_" + zero_padded_snapshot_index(solve_index);
  const std::string matrix_name = "A_" + matrix_id + ".bin";
  const std::string matrix_metadata_name = "A_" + matrix_id + ".txt";
  const std::string rhs_name = "b_" + snapshot_id + ".bin";
  const std::string x0_name = "x0_" + snapshot_id + ".bin";
  const std::string matrix_path = join_paths(context.directory, matrix_name);
  const std::string matrix_metadata_path = join_paths(context.directory, matrix_metadata_name);

  int directory_error = data.rank == 0 ? create_directory_tree(context.directory) : 0;
  PetscCallMPI(MPI_Bcast(&directory_error, 1, MPI_INT, 0, data.comm));
  PetscCheck(!directory_error, data.comm, PETSC_ERR_FILE_WRITE,
             "Cannot create snapshot directory: %s", context.directory.c_str());

  if(write_matrix) {
    // New matrix versions belong exclusively to this KSP; never adopt existing files.
    int collision = 0;
    if(data.rank == 0) {
      const std::string paths[] = {matrix_path, matrix_metadata_path,
                                   matrix_path + ".tmp", matrix_metadata_path + ".tmp"};
      for(const auto& path : paths) {
        struct stat status;
        if(lstat(path.c_str(), &status) == 0 || errno != ENOENT) {
          collision = 1;
          break;
        }
      }
    }
    PetscCallMPI(MPI_Bcast(&collision, 1, MPI_INT, 0, data.comm));
    PetscCheck(!collision, data.comm, PETSC_ERR_FILE_WRITE,
               "Snapshot matrix output already exists or cannot be inspected: %s", matrix_path.c_str());
    PetscCall(write_matrix_to_petsc_binary_file(data.comm, data.matrix, matrix_path + ".tmp"));
    PetscCall(commit_temporary_file_on_rank_zero(data.comm, matrix_path + ".tmp", matrix_path));

    int metadata_error = 0;
    if(data.rank == 0) {
      std::ofstream metadata(matrix_metadata_path + ".tmp");
      metadata << "matrix_key=" << matrix_id << '\n'
               << "matrix_id=" << matrix_id << '\n'
               << "A=" << matrix_name << '\n'
               << "rows=" << data.rows << '\n'
               << "cols=" << data.cols << '\n'
               << "mpi_size=" << data.mpi_size << '\n'
               << "matrix_nullspace_attached=" << (data.nullspaces[0] != nullptr) << '\n'
               << "transpose_nullspace_attached=" << (data.nullspaces[1] != nullptr) << '\n'
               << "near_nullspace_attached=" << (data.nullspaces[2] != nullptr) << '\n'
               << "nullspace_kind=" << data.nullspace_kind << '\n'
               << "nullspace_field_index=-1\n"
               << "nullspace_block_size=0\n"
               << "matrix_state=" << static_cast<long long>(data.matrix_state) << '\n'
               << "matrix_nonzero_state=" << static_cast<long long>(data.matrix_nonzero_state) << '\n'
               << "matrix_object_id=" << static_cast<unsigned long long>(data.matrix_id) << '\n';
      write_ownership_ranges(metadata, "row_ownership_ranges", data.row_ranges, data.mpi_size);
      metadata.close();
      if(!metadata) metadata_error = PETSC_ERR_FILE_WRITE;
    }
    PetscCallMPI(MPI_Bcast(&metadata_error, 1, MPI_INT, 0, data.comm));
    PetscCheck(!metadata_error, data.comm, PETSC_ERR_FILE_WRITE,
               "Cannot write matrix metadata: %s", matrix_metadata_path.c_str());
    PetscCall(commit_temporary_file_on_rank_zero(data.comm, matrix_metadata_path + ".tmp", matrix_metadata_path));
  }

  PetscCall(write_vector_to_petsc_binary_file(data.comm, data.rhs, join_paths(context.directory, rhs_name)));
  PetscCall(write_vector_to_petsc_binary_file(data.comm, data.initial_guess, join_paths(context.directory, x0_name)));
  const std::string metadata_path = join_paths(context.directory, "s_" + snapshot_id + ".txt");
  int metadata_error = 0;
  if(data.rank == 0) {
    std::ofstream metadata(metadata_path);
    metadata << std::setprecision(17)
             << "solve_index=" << solve_index << '\n'
             << "snapshot_file_prefix=" << context.prefix << '\n'
             << "snapshot_id=" << snapshot_id << '\n'
             << "matrix_key=" << matrix_id << '\n'
             << "matrix_id=" << matrix_id << '\n'
             << "matrix_write=" << (write_matrix ? "written" : "reused") << '\n'
             << "A=" << matrix_name << '\n'
             << "matrix_metadata=" << matrix_metadata_name << '\n'
             << "b=" << rhs_name << '\n'
             << "x0=" << x0_name << '\n'
             << "rows=" << data.rows << '\n'
             << "cols=" << data.cols << '\n'
             << "rhs_size=" << data.rhs_size << '\n'
             << "mpi_size=" << data.mpi_size << '\n'
             << "matrix_nullspace_attached=" << (data.nullspaces[0] != nullptr) << '\n'
             << "transpose_nullspace_attached=" << (data.nullspaces[1] != nullptr) << '\n'
             << "near_nullspace_attached=" << (data.nullspaces[2] != nullptr) << '\n'
             << "nullspace_kind=" << data.nullspace_kind << '\n'
             << "nullspace_field_index=-1\n"
             << "nullspace_block_size=0\n"
             << "rhs_nullspace_component_removed=unknown\n";
    write_ownership_ranges(metadata, "row_ownership_ranges", data.row_ranges, data.mpi_size);
    write_ownership_ranges(metadata, "rhs_ownership_ranges", data.rhs_ranges, data.mpi_size);
    write_ownership_ranges(metadata, "x0_ownership_ranges", data.x0_ranges, data.mpi_size);
    metadata << "relative_tolerance=" << data.rtol << '\n'
             << "absolute_tolerance=" << data.atol << '\n'
             << "divergence_tolerance=" << data.dtol << '\n'
             << "max_iterations=" << data.max_iterations << '\n'
             << "initial_guess_nonzero=" << data.nonzero_guess << '\n';
    if(data.solver_name) metadata << "solver_name=" << data.solver_name << '\n';
    if(data.symmetric_known) metadata << "symmetric=" << data.symmetric << '\n';
    if(data.spd_known) metadata << "spd=" << data.spd << '\n';
    metadata.close();
    if(!metadata) metadata_error = PETSC_ERR_FILE_WRITE;
  }
  PetscCallMPI(MPI_Bcast(&metadata_error, 1, MPI_INT, 0, data.comm));
  PetscCheck(!metadata_error, data.comm, PETSC_ERR_FILE_WRITE,
             "Cannot write snapshot metadata: %s", metadata_path.c_str());
  return 0;
}

}  // namespace detail
}  // namespace petsc_snapshot
}  // namespace ksptune

inline PetscErrorCode KSPTuneExport(KSP ksp, Vec b, Vec x)
{
  namespace detail = ksptune::petsc_snapshot::detail;
  if(!ksp) return PETSC_ERR_ARG_NULL;
  const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(ksp));
  detail::ExportContext* context = nullptr;
  PetscCall(detail::get_export_context(ksp, &context));
  if(context->directory.empty()) return 0;
  PetscCheck(context->next_solve_index < INT_MAX, comm, PETSC_ERR_ARG_OUTOFRANGE,
             "KSPTune solve counter exceeded its range");
  const int solve_index = context->next_solve_index++;
  if(!context->solve_indices.empty() &&
     std::find(context->solve_indices.begin(), context->solve_indices.end(), solve_index) ==
         context->solve_indices.end()) return 0;
  PetscCheck(b && x, comm, PETSC_ERR_ARG_NULL, "KSPTuneExport requires both b and x");
  PetscBool operator_set = PETSC_FALSE;
  PetscCall(KSPGetOperatorsSet(ksp, &operator_set, nullptr));
  PetscCheck(operator_set, comm, PETSC_ERR_ARG_WRONGSTATE, "Set the KSP operator before exporting");

  detail::SnapshotData data;
  PetscCall(detail::read_snapshot_data(ksp, b, x, data));
  // A change on any rank creates a new matrix version for the whole communicator.
  int matrix_changed = data.matrix_id != context->matrix_id || data.matrix_state != context->matrix_state ||
      !std::equal(data.nullspace_ids, data.nullspace_ids + 3, context->nullspace_ids);
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &matrix_changed, 1, MPI_INT, MPI_MAX, comm));

  Vec zero_guess = nullptr;
  PetscErrorCode error = 0;
  if(!data.nonzero_guess) {
    PetscCall(VecDuplicate(x, &zero_guess));
    error = VecSet(zero_guess, 0.0);
    data.initial_guess = zero_guess;
  }
  if(!error) error = detail::write_snapshot(*context, data, solve_index, matrix_changed != 0);
  const PetscErrorCode cleanup_error = VecDestroy(&zero_guess);
  PetscCall(error);
  PetscCall(cleanup_error);
  if(matrix_changed) ++context->matrix_version;
  context->matrix_id = data.matrix_id;
  context->matrix_state = data.matrix_state;
  std::copy(data.nullspace_ids, data.nullspace_ids + 3, context->nullspace_ids);
  return 0;
}
