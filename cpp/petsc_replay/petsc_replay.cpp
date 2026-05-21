#include <petscksp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

#define KSPTUNE_PETSC_CALL(call) \
  do { \
    PetscErrorCode ierr_ = (call); \
    if(ierr_) return ierr_; \
  } while(false)

struct replay_args {
  std::string snapshot_collection_path;
  std::string json_output_path;
  std::string initial_guess_vector_path;
  std::string nullspace_mode = "from-metadata";
  std::string nullspace_actions = "default";
  int repeat = 1;
  int warmup = 0;
  int solve_target = -1;
  bool constant_nullspace = false;
  int field_nullspace = -1;
  int field_nullspace_block_size = 2;
  bool diagnose_matrix = false;
};

struct nullspace_actions {
  bool use_metadata = false;
  bool matrix = false;
  bool transpose = false;
  bool near = false;
  bool remove_rhs = false;
  std::string label = "none";
};

struct replay_nullspace_choice {
  bool enabled = false;
  bool explicit_nullspace = false;
  std::string source = "none";
  std::string kind = "none";
  int field_index = -1;
  int block_size = 0;
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
  double relative_tolerance = -1.0;
  double absolute_tolerance = -1.0;
  double divergence_tolerance = -1.0;
  int max_iterations = -1;
  std::vector<PetscInt> row_ownership_ranges;
  std::vector<PetscInt> rhs_ownership_ranges;
  std::vector<PetscInt> x0_ownership_ranges;
};

struct matrix_diagnostics {
  PetscInt row_count = 0;
  PetscInt column_count = 0;
  PetscInt local_row_count = 0;
  PetscInt local_column_count = 0;
  double nonzero_count = 0.0;
  double allocated_nonzero_count = 0.0;
  double petsc_matrix_memory_bytes = 0.0;
  bool symmetric_tested = false;
  bool symmetric = false;
};

struct step_result {
  int repeat_index = 0;
  int solve_index = 0;
  double solve_time_sec = 0.0;
  double initial_true_residual_norm = 0.0;
  double rhs_norm = 0.0;
  double initial_true_relative_residual = 0.0;
  double final_true_residual_norm = 0.0;
  double final_true_relative_residual = 0.0;
  int iterations = 0;
  int reason_code = 0;
};

struct replay_result {
  double objective_time_sec_median = 1.0e30;
  double total_wall_time_sec = 0.0;
  double matrix_load_time_sec = 0.0;
  double solver_setup_time_sec = 0.0;
  double solve_time_sec_total = 0.0;
  double solve_time_sec_mean = 0.0;
  double solve_time_sec_median = 0.0;
  double solve_time_sec_min = 0.0;
  double solve_time_sec_max = 0.0;
  double solve_time_sec_stddev = 0.0;
  double solve_time_sec_range = 0.0;
  double initial_true_residual_norm_mean = 0.0;
  double initial_true_relative_residual_mean = 0.0;
  double final_true_residual_norm_mean = 0.0;
  double final_true_relative_residual_mean = 0.0;
  std::vector<double> peak_memory_mb_per_rank;
  double peak_memory_mb_max_per_rank = 0.0;
  double peak_memory_mb_mean_per_rank = 0.0;
  double peak_memory_mb_sum = 0.0;
  int peak_memory_rank_count = 0;
  double iterations_median = 0.0;
  double iterations_total = 0.0;
  bool converged = true;
  int reason_code = 0;
  int snapshots = 0;
  int repeat = 0;
  int warmup = 0;
  int solve_count = 0;
  std::string nullspace = "none";
  int field_nullspace_index = -1;
  int field_nullspace_block_size = 0;
  std::string nullspace_source = "none";
  std::string nullspace_kind = "none";
  std::string nullspace_actions = "none";
  int matrix_nullspace_attached_count = 0;
  int transpose_nullspace_attached_count = 0;
  int near_nullspace_attached_count = 0;
  int rhs_nullspace_removed_count = 0;
  double rhs_nullspace_removed_component_norm_max = 0.0;
  double rhs_nullspace_removed_component_relative_norm_max = 0.0;
  std::string ksp_type;
  std::string pc_type;
  std::vector<matrix_diagnostics> matrices;
  std::vector<step_result> steps;
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

replay_args parse_args(int argc, char** argv)
{
  replay_args args;
  arg_value(argc, argv, "-snapshot_collection", args.snapshot_collection_path);
  arg_value(argc, argv, "-replay_json_out", args.json_output_path);
  arg_value(argc, argv, "-replay_initial_guess_vec", args.initial_guess_vector_path);
  arg_value(argc, argv, "-replay_nullspace", args.nullspace_mode);
  arg_value(argc, argv, "-replay_nullspace_actions", args.nullspace_actions);
  args.repeat = std::max(1, int_arg(argc, argv, "-replay_repeat", args.repeat));
  args.warmup = std::max(0, int_arg(argc, argv, "-replay_warmup", args.warmup));
  args.solve_target = int_arg(argc, argv, "-replay_solve_target", args.solve_target);
  args.constant_nullspace = bool_arg(argc, argv, "-replay_constant_nullspace", false);
  if(args.constant_nullspace) args.nullspace_mode = "constant";
  if(args.nullspace_mode == "constant") args.constant_nullspace = true;
  args.field_nullspace = int_arg(argc, argv, "-replay_field_nullspace", args.field_nullspace);
  if(args.field_nullspace >= 0) args.nullspace_mode = "field";
  args.field_nullspace_block_size = int_arg(
      argc,
      argv,
      "-replay_field_nullspace_block_size",
      args.field_nullspace_block_size);
  args.diagnose_matrix = bool_arg(argc, argv, "-replay_diagnose_matrix", false);
  return args;
}

void clear_replay_options()
{
  const char* keys[] = {
    "-snapshot_collection",
    "-replay_json_out",
    "-replay_initial_guess_vec",
    "-replay_repeat",
    "-replay_warmup",
    "-replay_solve_target",
    "-replay_nullspace",
    "-replay_nullspace_actions",
    "-replay_constant_nullspace",
    "-replay_field_nullspace",
    "-replay_field_nullspace_block_size",
    "-replay_diagnose_matrix",
  };
  for(const char* key : keys) {
    PetscOptionsClearValue(nullptr, key);
  }
}

std::string nullspace_description(const replay_args& args)
{
  if(args.nullspace_mode == "from-metadata") return "from-metadata";
  if(args.nullspace_mode == "none") return "none";
  if(args.field_nullspace >= 0) return "field";
  if(args.constant_nullspace) return "constant";
  return "none";
}

std::string trim(const std::string& text);

std::string lower_text(std::string text)
{
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  return text;
}

std::string actions_label(const nullspace_actions& actions)
{
  if(actions.use_metadata) return "metadata";
  std::vector<std::string> names;
  if(actions.matrix) names.push_back("matrix");
  if(actions.transpose) names.push_back("transpose");
  if(actions.near) names.push_back("near");
  if(actions.remove_rhs) names.push_back("remove-rhs");
  if(names.empty()) return "none";
  std::ostringstream label;
  for(size_t i = 0; i < names.size(); ++i) {
    if(i > 0) label << ',';
    label << names[i];
  }
  return label.str();
}

bool rhs_component_removed_from_metadata(const snapshot& snap)
{
  const std::string text = lower_text(snap.rhs_nullspace_component_removed);
  return text == "1" || text == "true" || text == "yes" || text == "on";
}

PetscErrorCode resolve_nullspace_actions(const replay_args& args,
                                         const snapshot& snap,
                                         bool explicit_nullspace,
                                         nullspace_actions& actions)
{
  const std::string text = lower_text(trim(args.nullspace_actions));
  if(text.empty() || text == "default") {
    if(explicit_nullspace) {
      actions.matrix = true;
      actions.transpose = true;
    } else {
      actions.use_metadata = true;
    }
  } else if(text == "metadata") {
    actions.use_metadata = true;
  } else if(text == "none") {
    actions = nullspace_actions{};
  } else if(text == "all") {
    actions.matrix = true;
    actions.transpose = true;
    actions.near = true;
    actions.remove_rhs = true;
  } else {
    std::stringstream stream(text);
    std::string token;
    while(std::getline(stream, token, ',')) {
      token = lower_text(trim(token));
      if(token == "right") token = "matrix";
      if(token == "left") token = "transpose";
      if(token == "remove-right-component") token = "remove-rhs";

      if(token == "matrix") actions.matrix = true;
      else if(token == "transpose") actions.transpose = true;
      else if(token == "near") actions.near = true;
      else if(token == "remove-rhs") actions.remove_rhs = true;
      else if(token.empty()) continue;
      else {
        PetscFPrintf(PETSC_COMM_WORLD, stderr, "Invalid nullspace action: %s\n", token.c_str());
        return PETSC_ERR_ARG_WRONG;
      }
    }
  }

  if(actions.use_metadata) {
    actions.matrix = snap.matrix_nullspace_attached;
    actions.transpose = snap.transpose_nullspace_attached;
    actions.near = snap.near_nullspace_attached;
    actions.remove_rhs = rhs_component_removed_from_metadata(snap);
  }
  actions.label = actions_label(actions);
  return 0;
}

PetscErrorCode resolve_replay_nullspace_choice(const replay_args& args,
                                               const snapshot& snap,
                                               replay_nullspace_choice& choice)
{
  const std::string mode = lower_text(trim(args.nullspace_mode));
  if(mode.empty() || mode == "from-metadata") {
    choice.source = "from-metadata";
    choice.explicit_nullspace = false;
    if(!snap.nullspace && snap.nullspace_kind == "none") return 0;
    if(snap.nullspace_kind == "unknown") {
      PetscFPrintf(
          PETSC_COMM_WORLD,
          stderr,
          "Snapshot declares a PETSc nullspace, but nullspace_kind=unknown. "
          "Pass -replay_nullspace none or an explicit reconstructible nullspace.\n");
      return PETSC_ERR_ARG_WRONGSTATE;
    }
    if(snap.nullspace_kind == "constant") {
      choice.enabled = true;
      choice.kind = "constant";
      return 0;
    }
    if(snap.nullspace_kind == "field") {
      if(snap.nullspace_field_index < 0 || snap.nullspace_block_size <= 0) {
        PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace metadata is incomplete\n");
        return PETSC_ERR_ARG_WRONGSTATE;
      }
      choice.enabled = true;
      choice.kind = "field";
      choice.field_index = snap.nullspace_field_index;
      choice.block_size = snap.nullspace_block_size;
      return 0;
    }
    if(snap.nullspace_kind == "none") return 0;

    PetscFPrintf(
        PETSC_COMM_WORLD,
        stderr,
        "Unsupported nullspace_kind in metadata: %s\n",
        snap.nullspace_kind.c_str());
    return PETSC_ERR_ARG_WRONGSTATE;
  }

  choice.explicit_nullspace = true;
  choice.source = "explicit";
  if(mode == "none") {
    choice.source = "none";
    return 0;
  }
  if(mode == "constant") {
    choice.enabled = true;
    choice.kind = "constant";
    return 0;
  }
  if(mode == "field") {
    if(args.field_nullspace < 0 || args.field_nullspace_block_size <= 0) {
      PetscFPrintf(PETSC_COMM_WORLD, stderr, "Explicit field nullspace is incomplete\n");
      return PETSC_ERR_ARG_WRONG;
    }
    choice.enabled = true;
    choice.kind = "field";
    choice.field_index = args.field_nullspace;
    choice.block_size = args.field_nullspace_block_size;
    return 0;
  }

  PetscFPrintf(PETSC_COMM_WORLD, stderr, "Invalid replay nullspace mode: %s\n", mode.c_str());
  return PETSC_ERR_ARG_WRONG;
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
  const std::string snapshot_collection_relative_path = base_directory + "/" + path;
  if(file_exists(snapshot_collection_relative_path) || !file_exists(path)) {
    return snapshot_collection_relative_path;
  }
  return path;
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
    if(key == "row_ownership_ranges") snap.row_ownership_ranges = parse_petsc_int_list(value);
    else if(key == "rhs_ownership_ranges") snap.rhs_ownership_ranges = parse_petsc_int_list(value);
    else if(key == "x0_ownership_ranges") snap.x0_ownership_ranges = parse_petsc_int_list(value);
    else if(key == "relative_tolerance") snap.relative_tolerance = std::atof(value.c_str());
    else if(key == "absolute_tolerance") snap.absolute_tolerance = std::atof(value.c_str());
    else if(key == "divergence_tolerance") snap.divergence_tolerance = std::atof(value.c_str());
    else if(key == "max_iterations") snap.max_iterations = std::atoi(value.c_str());
    else if(key == "symmetric") snap.symmetric = parse_bool(value);
    else if(key == "spd") snap.spd = parse_bool(value);
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
    else if(key == "nullspace") snap.matrix_nullspace_attached = parse_bool(value);
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

std::string first_present(const std::map<std::string, std::string>& row,
                          const std::vector<std::string>& keys)
{
  for(const std::string& key : keys) {
    const std::string value = field_or_empty(row, key);
    if(!value.empty()) return value;
  }
  return "";
}

snapshot snapshot_from_named_row(const std::map<std::string, std::string>& row,
                                 const std::string& base_directory)
{
  snapshot snap;
  snap.solve_index = parse_int_or_default(field_or_empty(row, "solve_index"), 0);
  snap.matrix_file_path = join_relative_to(
      base_directory,
      first_present(row, {"A", "matrix_file_path", "mat_file"}));
  snap.right_hand_side_file_path = join_relative_to(
      base_directory,
      first_present(row, {"b", "right_hand_side_file_path", "rhs_file"}));
  snap.initial_guess_file_path = join_relative_to(
      base_directory,
      first_present(row, {"x0", "initial_guess_file_path"}));
  snap.metadata_file_path = join_relative_to(
      base_directory,
      first_present(row, {"meta", "metadata_file_path", "meta_file"}));
  snap.rows = parse_petsc_int_or_default(field_or_empty(row, "rows"), 0);
  snap.cols = parse_petsc_int_or_default(field_or_empty(row, "cols"), snap.rows);
  snap.right_hand_side_size = parse_petsc_int_or_default(
      first_present(row, {"rhs_size", "right_hand_side_size"}),
      0);
  snap.mpi_size = parse_int_or_default(field_or_empty(row, "mpi_size"), 1);
  snap.symmetric = parse_bool(field_or_empty(row, "symmetric"));
  snap.spd = parse_bool(field_or_empty(row, "spd"));
  snap.matrix_nullspace_attached =
      parse_bool(first_present(row, {"matrix_nullspace_attached", "nullspace"}));
  snap.transpose_nullspace_attached = parse_bool(field_or_empty(row, "transpose_nullspace_attached"));
  snap.near_nullspace_attached = parse_bool(field_or_empty(row, "near_nullspace_attached"));
  snap.nullspace_kind = first_present(row, {"nullspace_kind"});
  if(snap.nullspace_kind.empty()) snap.nullspace_kind = snap.matrix_nullspace_attached ? "unknown" : "none";
  snap.nullspace_field_index = parse_int_or_default(field_or_empty(row, "nullspace_field_index"), -1);
  snap.nullspace_block_size = parse_int_or_default(field_or_empty(row, "nullspace_block_size"), 0);
  snap.rhs_nullspace_component_removed =
      first_present(row, {"rhs_nullspace_component_removed"});
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
  const bool has_header = std::find(header.begin(), header.end(), "solve_index") != header.end();
  if(!has_header) return snapshots;

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
  KSPTUNE_PETSC_CALL(PetscViewerBinaryOpen(
      PETSC_COMM_WORLD,
      snap.matrix_file_path.c_str(),
      FILE_MODE_READ,
      &viewer));
  KSPTUNE_PETSC_CALL(MatCreate(PETSC_COMM_WORLD, matrix));
  if(ownership_ranges_match_current(snap.row_ownership_ranges)) {
    const PetscInt local_rows = local_size_from_ranges(snap.row_ownership_ranges);
    KSPTUNE_PETSC_CALL(MatSetSizes(*matrix, local_rows, local_rows, snap.rows, snap.cols));
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

PetscErrorCode collect_matrix_diagnostics(Mat matrix,
                                          bool diagnose_matrix,
                                          matrix_diagnostics& diagnostics)
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

  if(diagnose_matrix) {
    PetscBool symmetric = PETSC_FALSE;
    KSPTUNE_PETSC_CALL(MatIsSymmetric(matrix, 1.0e-12, &symmetric));
    diagnostics.symmetric_tested = true;
    diagnostics.symmetric = symmetric == PETSC_TRUE;
  }
  return 0;
}

double local_peak_memory_mb()
{
  rusage usage;
  if(getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
  return double(usage.ru_maxrss) / 1024.0;
}

void collect_memory_diagnostics(replay_result& result)
{
  const double local_peak_memory = local_peak_memory_mb();
  MPI_Allreduce(
      &local_peak_memory,
      &result.peak_memory_mb_max_per_rank,
      1,
      MPI_DOUBLE,
      MPI_MAX,
      PETSC_COMM_WORLD);
  MPI_Allreduce(
      &local_peak_memory,
      &result.peak_memory_mb_sum,
      1,
      MPI_DOUBLE,
      MPI_SUM,
      PETSC_COMM_WORLD);

  int rank = 0;
  int communicator_size = 1;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &communicator_size);
  result.peak_memory_rank_count = communicator_size;
  result.peak_memory_mb_mean_per_rank = communicator_size > 0
      ? result.peak_memory_mb_sum / double(communicator_size)
      : 0.0;
  if(rank == 0) result.peak_memory_mb_per_rank.resize(size_t(communicator_size));
  MPI_Gather(
      &local_peak_memory,
      1,
      MPI_DOUBLE,
      rank == 0 ? result.peak_memory_mb_per_rank.data() : nullptr,
      1,
      MPI_DOUBLE,
      0,
      PETSC_COMM_WORLD);
}

PetscErrorCode compute_residual_norm(Mat matrix,
                                     Vec right_hand_side,
                                     Vec solution,
                                     double& residual_norm,
                                     double& rhs_norm,
                                     double& relative_residual)
{
  Vec matrix_times_solution = nullptr;
  Vec residual_vector = nullptr;
  KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, &matrix_times_solution));
  KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, &residual_vector));
  KSPTUNE_PETSC_CALL(MatMult(matrix, solution, matrix_times_solution));
  KSPTUNE_PETSC_CALL(VecCopy(right_hand_side, residual_vector));
  KSPTUNE_PETSC_CALL(VecAXPY(residual_vector, -1.0, matrix_times_solution));

  PetscReal residual = 0.0;
  PetscReal rhs = 0.0;
  KSPTUNE_PETSC_CALL(VecNorm(residual_vector, NORM_2, &residual));
  KSPTUNE_PETSC_CALL(VecNorm(right_hand_side, NORM_2, &rhs));
  residual_norm = double(residual);
  rhs_norm = double(rhs);
  relative_residual = rhs > 0.0 ? double(residual / rhs) : double(residual);

  KSPTUNE_PETSC_CALL(VecDestroy(&residual_vector));
  KSPTUNE_PETSC_CALL(VecDestroy(&matrix_times_solution));
  return 0;
}

PetscErrorCode create_field_nullspace(Vec reference_vector,
                                      PetscInt block_size,
                                      PetscInt field,
                                      MatNullSpace* nullspace)
{
  if(block_size <= 0) {
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace block size must be positive\n");
    return PETSC_ERR_ARG_OUTOFRANGE;
  }
  if(field < 0 || field >= block_size) {
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace index must be in [0, block_size)\n");
    return PETSC_ERR_ARG_OUTOFRANGE;
  }

  Vec basis = nullptr;
  KSPTUNE_PETSC_CALL(VecDuplicate(reference_vector, &basis));

  PetscInt ownership_start = 0;
  PetscInt ownership_end = 0;
  PetscInt local_size = 0;
  KSPTUNE_PETSC_CALL(VecGetOwnershipRange(reference_vector, &ownership_start, &ownership_end));
  KSPTUNE_PETSC_CALL(VecGetLocalSize(reference_vector, &local_size));

  PetscScalar* values = nullptr;
  KSPTUNE_PETSC_CALL(VecGetArray(basis, &values));
  for(PetscInt local = 0; local < local_size; ++local) {
    const PetscInt global = ownership_start + local;
    values[local] = (global % block_size == field) ? PetscScalar(1.0) : PetscScalar(0.0);
  }
  KSPTUNE_PETSC_CALL(VecRestoreArray(basis, &values));
  KSPTUNE_PETSC_CALL(VecAssemblyBegin(basis));
  KSPTUNE_PETSC_CALL(VecAssemblyEnd(basis));

  PetscReal norm = 0.0;
  KSPTUNE_PETSC_CALL(VecNormalize(basis, &norm));
  if(norm == 0.0) {
    VecDestroy(&basis);
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace basis is empty\n");
    return PETSC_ERR_ARG_WRONGSTATE;
  }

  KSPTUNE_PETSC_CALL(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, &basis, nullspace));
  KSPTUNE_PETSC_CALL(VecDestroy(&basis));
  return 0;
}

PetscErrorCode apply_nullspace_actions(Mat matrix,
                                       Vec right_hand_side,
                                       MatNullSpace nullspace,
                                       const nullspace_actions& actions,
                                       replay_result& aggregate)
{
  if(actions.matrix) {
    KSPTUNE_PETSC_CALL(MatSetNullSpace(matrix, nullspace));
    ++aggregate.matrix_nullspace_attached_count;
  }
  if(actions.transpose) {
    KSPTUNE_PETSC_CALL(MatSetTransposeNullSpace(matrix, nullspace));
    ++aggregate.transpose_nullspace_attached_count;
  }
  if(actions.near) {
    KSPTUNE_PETSC_CALL(MatSetNearNullSpace(matrix, nullspace));
    ++aggregate.near_nullspace_attached_count;
  }
  if(actions.remove_rhs) {
    Vec projected = nullptr;
    Vec removed_component = nullptr;
    KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, &projected));
    KSPTUNE_PETSC_CALL(VecCopy(right_hand_side, projected));
    KSPTUNE_PETSC_CALL(MatNullSpaceRemove(nullspace, projected));

    KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, &removed_component));
    KSPTUNE_PETSC_CALL(VecCopy(right_hand_side, removed_component));
    KSPTUNE_PETSC_CALL(VecAXPY(removed_component, PetscScalar(-1.0), projected));

    PetscReal rhs_norm = 0.0;
    PetscReal removed_norm = 0.0;
    KSPTUNE_PETSC_CALL(VecNorm(right_hand_side, NORM_2, &rhs_norm));
    KSPTUNE_PETSC_CALL(VecNorm(removed_component, NORM_2, &removed_norm));
    KSPTUNE_PETSC_CALL(VecCopy(projected, right_hand_side));

    ++aggregate.rhs_nullspace_removed_count;
    aggregate.rhs_nullspace_removed_component_norm_max = std::max(
        aggregate.rhs_nullspace_removed_component_norm_max,
        double(removed_norm));
    aggregate.rhs_nullspace_removed_component_relative_norm_max = std::max(
        aggregate.rhs_nullspace_removed_component_relative_norm_max,
        rhs_norm > 0.0 ? double(removed_norm / rhs_norm) : double(removed_norm));

    KSPTUNE_PETSC_CALL(VecDestroy(&removed_component));
    KSPTUNE_PETSC_CALL(VecDestroy(&projected));
  }
  return 0;
}

const char* reason_string(int reason)
{
  switch(reason) {
    case 2: return "KSP_CONVERGED_RTOL";
    case 3: return "KSP_CONVERGED_ATOL";
    case 4: return "KSP_CONVERGED_ITS";
    case 5: return "KSP_CONVERGED_CG_NEG_CURVE";
    case 6: return "KSP_CONVERGED_CG_CONSTRAINED";
    case 7: return "KSP_CONVERGED_STEP_LENGTH";
    case 8: return "KSP_CONVERGED_HAPPY_BREAKDOWN";
    case 9: return "KSP_CONVERGED_ATOL_NORMAL";
    case -2: return "KSP_DIVERGED_NULL";
    case -3: return "KSP_DIVERGED_ITS";
    case -4: return "KSP_DIVERGED_DTOL";
    case -5: return "KSP_DIVERGED_BREAKDOWN";
    case -6: return "KSP_DIVERGED_BREAKDOWN_BICG";
    case -7: return "KSP_DIVERGED_NONSYMMETRIC";
    case -8: return "KSP_DIVERGED_INDEFINITE_PC";
    case -9: return "KSP_DIVERGED_NANORINF";
    case -10: return "KSP_DIVERGED_INDEFINITE_MAT";
    case -11: return "KSP_DIVERGED_PC_FAILED";
    default: return "KSP_REASON_UNKNOWN";
  }
}

double mean(const std::vector<double>& values)
{
  if(values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) / double(values.size());
}

double median(std::vector<double> values)
{
  if(values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if(values.size() % 2) return values[mid];
  return 0.5 * (values[mid - 1] + values[mid]);
}

double population_stddev(const std::vector<double>& values)
{
  if(values.empty()) return 0.0;
  const double values_mean = mean(values);
  double squared_delta_sum = 0.0;
  for(double value : values) {
    const double delta = value - values_mean;
    squared_delta_sum += delta * delta;
  }
  return std::sqrt(squared_delta_sum / double(values.size()));
}

double value_range(const std::vector<double>& values)
{
  if(values.empty()) return 0.0;
  auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
  return *maximum - *minimum;
}

double minimum_value(const std::vector<double>& values)
{
  if(values.empty()) return 0.0;
  return *std::min_element(values.begin(), values.end());
}

double maximum_value(const std::vector<double>& values)
{
  if(values.empty()) return 0.0;
  return *std::max_element(values.begin(), values.end());
}

PetscErrorCode replay_snapshot(const replay_args& args,
                               const snapshot& snap,
                               replay_result& aggregate,
                               std::vector<double>& solve_times,
                               std::vector<double>& iteration_counts)
{
  Mat matrix = nullptr;
  Vec right_hand_side = nullptr;
  Vec initial_guess = nullptr;
  Vec solution = nullptr;
  KSP ksp = nullptr;

  PetscLogDouble load_start = 0.0;
  PetscLogDouble load_end = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&load_start));
  KSPTUNE_PETSC_CALL(load_matrix_binary(snap, &matrix));
  KSPTUNE_PETSC_CALL(load_vector_binary(
      snap.right_hand_side_file_path,
      snap.rhs_ownership_ranges,
      snap.right_hand_side_size,
      &right_hand_side));
  const std::string initial_guess_path = args.initial_guess_vector_path.empty()
      ? snap.initial_guess_file_path
      : args.initial_guess_vector_path;
  KSPTUNE_PETSC_CALL(load_vector_binary(
      initial_guess_path,
      snap.x0_ownership_ranges,
      snap.right_hand_side_size,
      &initial_guess));
  KSPTUNE_PETSC_CALL(VecDuplicate(initial_guess, &solution));
  KSPTUNE_PETSC_CALL(VecCopy(initial_guess, solution));
  KSPTUNE_PETSC_CALL(PetscTime(&load_end));
  double load_elapsed = double(load_end - load_start);
  MPI_Allreduce(MPI_IN_PLACE, &load_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  aggregate.matrix_load_time_sec += load_elapsed;

  matrix_diagnostics diagnostics;
  KSPTUNE_PETSC_CALL(collect_matrix_diagnostics(matrix, args.diagnose_matrix, diagnostics));
  aggregate.matrices.push_back(diagnostics);

  replay_nullspace_choice nullspace_choice;
  KSPTUNE_PETSC_CALL(resolve_replay_nullspace_choice(args, snap, nullspace_choice));
  nullspace_actions actions;
  KSPTUNE_PETSC_CALL(resolve_nullspace_actions(
      args,
      snap,
      nullspace_choice.explicit_nullspace,
      actions));
  if(aggregate.nullspace_source.empty() || aggregate.nullspace_source == "none") {
    aggregate.nullspace_source = nullspace_choice.source;
  }

  if(nullspace_choice.enabled) {
    aggregate.nullspace_kind = nullspace_choice.kind;
    aggregate.nullspace_actions = actions.label;
    if(nullspace_choice.kind == "field") {
      aggregate.field_nullspace_index = nullspace_choice.field_index;
      aggregate.field_nullspace_block_size = nullspace_choice.block_size;
    }
  }

  if(nullspace_choice.kind == "constant") {
    MatNullSpace nullspace = nullptr;
    KSPTUNE_PETSC_CALL(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, nullptr, &nullspace));
    KSPTUNE_PETSC_CALL(apply_nullspace_actions(
        matrix,
        right_hand_side,
        nullspace,
        actions,
        aggregate));
    KSPTUNE_PETSC_CALL(MatNullSpaceDestroy(&nullspace));
  }
  if(nullspace_choice.kind == "field" && nullspace_choice.field_index >= 0) {
    MatNullSpace nullspace = nullptr;
    KSPTUNE_PETSC_CALL(create_field_nullspace(
        right_hand_side,
        PetscInt(nullspace_choice.block_size),
        PetscInt(nullspace_choice.field_index),
        &nullspace));
    KSPTUNE_PETSC_CALL(apply_nullspace_actions(
        matrix,
        right_hand_side,
        nullspace,
        actions,
        aggregate));
    KSPTUNE_PETSC_CALL(MatNullSpaceDestroy(&nullspace));
  }
  if(snap.symmetric) KSPTUNE_PETSC_CALL(MatSetOption(matrix, MAT_SYMMETRIC, PETSC_TRUE));
  if(snap.spd) KSPTUNE_PETSC_CALL(MatSetOption(matrix, MAT_SPD, PETSC_TRUE));

  KSPTUNE_PETSC_CALL(KSPCreate(PETSC_COMM_WORLD, &ksp));
  KSPTUNE_PETSC_CALL(KSPSetOperators(ksp, matrix, matrix));
  if(snap.relative_tolerance >= 0.0 ||
     snap.absolute_tolerance >= 0.0 ||
     snap.divergence_tolerance >= 0.0 ||
     snap.max_iterations >= 0) {
    const PetscReal rtol = snap.relative_tolerance >= 0.0
        ? PetscReal(snap.relative_tolerance)
        : PETSC_DEFAULT;
    const PetscReal atol = snap.absolute_tolerance >= 0.0
        ? PetscReal(snap.absolute_tolerance)
        : PETSC_DEFAULT;
    const PetscReal dtol = snap.divergence_tolerance >= 0.0
        ? PetscReal(snap.divergence_tolerance)
        : PETSC_DEFAULT;
    const PetscInt max_it = snap.max_iterations >= 0
        ? PetscInt(snap.max_iterations)
        : PETSC_DEFAULT;
    KSPTUNE_PETSC_CALL(KSPSetTolerances(ksp, rtol, atol, dtol, max_it));
  }
  KSPTUNE_PETSC_CALL(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  KSPTUNE_PETSC_CALL(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
  KSPTUNE_PETSC_CALL(KSPSetFromOptions(ksp));

  PetscLogDouble t0 = 0.0;
  PetscLogDouble t1 = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&t0));
  KSPTUNE_PETSC_CALL(KSPSetUp(ksp));
  KSPTUNE_PETSC_CALL(PetscTime(&t1));
  double setup_elapsed = double(t1 - t0);
  MPI_Allreduce(MPI_IN_PLACE, &setup_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  aggregate.solver_setup_time_sec += setup_elapsed;

  KSPType ksp_type = nullptr;
  PCType pc_type = nullptr;
  PC pc = nullptr;
  KSPTUNE_PETSC_CALL(KSPGetType(ksp, &ksp_type));
  KSPTUNE_PETSC_CALL(KSPGetPC(ksp, &pc));
  KSPTUNE_PETSC_CALL(PCGetType(pc, &pc_type));
  aggregate.ksp_type = ksp_type ? ksp_type : "";
  aggregate.pc_type = pc_type ? pc_type : "";

  for(int iteration = 0; iteration < args.warmup + args.repeat; ++iteration) {
    KSPTUNE_PETSC_CALL(VecCopy(initial_guess, solution));

    double initial_residual = 0.0;
    double rhs_norm = 0.0;
    double initial_relative = 0.0;
    KSPTUNE_PETSC_CALL(compute_residual_norm(
        matrix,
        right_hand_side,
        solution,
        initial_residual,
        rhs_norm,
        initial_relative));

    KSPTUNE_PETSC_CALL(PetscTime(&t0));
    KSPTUNE_PETSC_CALL(KSPSolve(ksp, right_hand_side, solution));
    KSPTUNE_PETSC_CALL(PetscTime(&t1));
    double solve_elapsed = double(t1 - t0);
    MPI_Allreduce(MPI_IN_PLACE, &solve_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

    KSPConvergedReason reason;
    PetscInt ksp_iterations = 0;
    KSPTUNE_PETSC_CALL(KSPGetConvergedReason(ksp, &reason));
    KSPTUNE_PETSC_CALL(KSPGetIterationNumber(ksp, &ksp_iterations));
    if(reason < 0) aggregate.converged = false;
    aggregate.reason_code = int(reason);

    double final_residual = 0.0;
    double final_rhs_norm = 0.0;
    double final_relative = 0.0;
    KSPTUNE_PETSC_CALL(compute_residual_norm(
        matrix,
        right_hand_side,
        solution,
        final_residual,
        final_rhs_norm,
        final_relative));

    if(iteration >= args.warmup) {
      step_result step;
      step.repeat_index = iteration - args.warmup;
      step.solve_index = snap.solve_index;
      step.solve_time_sec = solve_elapsed;
      step.initial_true_residual_norm = initial_residual;
      step.rhs_norm = rhs_norm;
      step.initial_true_relative_residual = initial_relative;
      step.final_true_residual_norm = final_residual;
      step.final_true_relative_residual = final_relative;
      step.iterations = int(ksp_iterations);
      step.reason_code = int(reason);
      aggregate.steps.push_back(step);
      solve_times.push_back(solve_elapsed);
      iteration_counts.push_back(double(ksp_iterations));
    }
  }

  KSPTUNE_PETSC_CALL(KSPDestroy(&ksp));
  KSPTUNE_PETSC_CALL(VecDestroy(&solution));
  KSPTUNE_PETSC_CALL(VecDestroy(&initial_guess));
  KSPTUNE_PETSC_CALL(VecDestroy(&right_hand_side));
  KSPTUNE_PETSC_CALL(MatDestroy(&matrix));
  return 0;
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

void write_json_result(const replay_result& result, const std::string& path)
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(rank != 0) return;

  std::ostringstream json;
  json << std::setprecision(17)
       << "{\n"
       << "  \"objective_time_sec_median\": " << result.objective_time_sec_median << ",\n"
       << "  \"total_wall_time_sec\": " << result.total_wall_time_sec << ",\n"
       << "  \"matrix_load_time_sec\": " << result.matrix_load_time_sec << ",\n"
       << "  \"solver_setup_time_sec\": " << result.solver_setup_time_sec << ",\n"
       << "  \"solve_time_sec_total\": " << result.solve_time_sec_total << ",\n"
       << "  \"solve_time_sec_mean\": " << result.solve_time_sec_mean << ",\n"
       << "  \"solve_time_sec_median\": " << result.solve_time_sec_median << ",\n"
       << "  \"solve_time_sec_min\": " << result.solve_time_sec_min << ",\n"
       << "  \"solve_time_sec_max\": " << result.solve_time_sec_max << ",\n"
       << "  \"solve_time_sec_stddev\": " << result.solve_time_sec_stddev << ",\n"
       << "  \"solve_time_sec_range\": " << result.solve_time_sec_range << ",\n"
       << "  \"initial_true_residual_norm_mean\": "
       << result.initial_true_residual_norm_mean << ",\n"
       << "  \"initial_true_relative_residual_mean\": "
       << result.initial_true_relative_residual_mean << ",\n"
       << "  \"final_true_residual_norm_mean\": "
       << result.final_true_residual_norm_mean << ",\n"
       << "  \"final_true_relative_residual_mean\": "
       << result.final_true_relative_residual_mean << ",\n"
       << "  \"initial_residual_norm_mean\": " << result.initial_true_residual_norm_mean << ",\n"
       << "  \"initial_relative_residual_mean\": "
       << result.initial_true_relative_residual_mean << ",\n"
       << "  \"final_residual_norm_mean\": " << result.final_true_residual_norm_mean << ",\n"
       << "  \"final_relative_residual_mean\": "
       << result.final_true_relative_residual_mean << ",\n"
       << "  \"peak_memory_mb_per_rank\": [";
  for(size_t i = 0; i < result.peak_memory_mb_per_rank.size(); ++i) {
    if(i > 0) json << ", ";
    json << result.peak_memory_mb_per_rank[i];
  }
  json << "],\n"
       << "  \"peak_memory_mb_max_per_rank\": " << result.peak_memory_mb_max_per_rank << ",\n"
       << "  \"peak_memory_mb_mean_per_rank\": " << result.peak_memory_mb_mean_per_rank
       << ",\n"
       << "  \"peak_memory_mb_sum\": " << result.peak_memory_mb_sum << ",\n"
       << "  \"peak_memory_rank_count\": " << result.peak_memory_rank_count << ",\n"
       << "  \"converged\": " << (result.converged ? "true" : "false") << ",\n"
       << "  \"reason\": \"" << reason_string(result.reason_code) << "\",\n"
       << "  \"reason_code\": " << result.reason_code << ",\n"
       << "  \"iterations_median\": " << result.iterations_median << ",\n"
       << "  \"iterations_total\": " << result.iterations_total << ",\n"
       << "  \"snapshots\": " << result.snapshots << ",\n"
       << "  \"repeat\": " << result.repeat << ",\n"
       << "  \"warmup\": " << result.warmup << ",\n"
       << "  \"solve_count\": " << result.solve_count << ",\n"
       << "  \"nullspace\": \"" << json_escape(result.nullspace) << "\",\n"
       << "  \"nullspace_source\": \"" << json_escape(result.nullspace_source) << "\",\n"
       << "  \"nullspace_kind\": \"" << json_escape(result.nullspace_kind) << "\",\n"
       << "  \"nullspace_actions\": \"" << json_escape(result.nullspace_actions) << "\",\n"
       << "  \"field_nullspace_index\": " << result.field_nullspace_index << ",\n"
       << "  \"field_nullspace_block_size\": " << result.field_nullspace_block_size << ",\n"
       << "  \"matrix_nullspace_attached_count\": "
       << result.matrix_nullspace_attached_count << ",\n"
       << "  \"transpose_nullspace_attached_count\": "
       << result.transpose_nullspace_attached_count << ",\n"
       << "  \"near_nullspace_attached_count\": "
       << result.near_nullspace_attached_count << ",\n"
       << "  \"rhs_nullspace_removed_count\": "
       << result.rhs_nullspace_removed_count << ",\n"
       << "  \"rhs_nullspace_removed_component_norm_max\": "
       << result.rhs_nullspace_removed_component_norm_max << ",\n"
       << "  \"rhs_nullspace_removed_component_relative_norm_max\": "
       << result.rhs_nullspace_removed_component_relative_norm_max << ",\n"
       << "  \"ksp_type\": \"" << json_escape(result.ksp_type) << "\",\n"
       << "  \"pc_type\": \"" << json_escape(result.pc_type) << "\",\n"
       << "  \"matrices\": [\n";
  for(size_t i = 0; i < result.matrices.size(); ++i) {
    const matrix_diagnostics& matrix = result.matrices[i];
    json << "    {"
         << "\"row_count\": " << matrix.row_count << ", "
         << "\"column_count\": " << matrix.column_count << ", "
         << "\"local_row_count\": " << matrix.local_row_count << ", "
         << "\"local_column_count\": " << matrix.local_column_count << ", "
         << "\"nonzero_count\": " << matrix.nonzero_count << ", "
         << "\"allocated_nonzero_count\": " << matrix.allocated_nonzero_count << ", "
         << "\"petsc_matrix_memory_bytes\": " << matrix.petsc_matrix_memory_bytes << ", "
         << "\"rows\": " << matrix.row_count << ", "
         << "\"cols\": " << matrix.column_count << ", "
         << "\"local_rows\": " << matrix.local_row_count << ", "
         << "\"local_cols\": " << matrix.local_column_count << ", "
         << "\"nonzeros_used\": " << matrix.nonzero_count << ", "
         << "\"nonzeros_allocated\": " << matrix.allocated_nonzero_count << ", "
         << "\"memory_bytes\": " << matrix.petsc_matrix_memory_bytes << ", "
         << "\"symmetric_tested\": " << (matrix.symmetric_tested ? "true" : "false") << ", "
         << "\"symmetric\": " << (matrix.symmetric ? "true" : "false") << "}";
    if(i + 1 < result.matrices.size()) json << ",";
    json << "\n";
  }
  json << "  ],\n"
       << "  \"steps\": [\n";
  for(size_t i = 0; i < result.steps.size(); ++i) {
    const step_result& step = result.steps[i];
    json << "    {"
         << "\"repeat_index\": " << step.repeat_index << ", "
         << "\"solve_index\": " << step.solve_index << ", "
         << "\"solve_time_sec\": " << step.solve_time_sec << ", "
         << "\"initial_true_residual_norm\": " << step.initial_true_residual_norm << ", "
         << "\"rhs_norm\": " << step.rhs_norm << ", "
         << "\"initial_true_relative_residual\": "
         << step.initial_true_relative_residual << ", "
         << "\"final_true_residual_norm\": " << step.final_true_residual_norm << ", "
         << "\"final_true_relative_residual\": "
         << step.final_true_relative_residual << ", "
         << "\"initial_residual_norm\": " << step.initial_true_residual_norm << ", "
         << "\"initial_relative_residual\": "
         << step.initial_true_relative_residual << ", "
         << "\"final_residual_norm\": " << step.final_true_residual_norm << ", "
         << "\"final_relative_residual\": " << step.final_true_relative_residual << ", "
         << "\"iterations\": " << step.iterations << ", "
         << "\"reason\": \"" << reason_string(step.reason_code) << "\", "
         << "\"reason_code\": " << step.reason_code << "}";
    if(i + 1 < result.steps.size()) json << ",";
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

int find_snapshot_by_solve_index(const std::vector<snapshot>& snapshots, int solve_index)
{
  for(size_t i = 0; i < snapshots.size(); ++i) {
    if(snapshots[i].solve_index == solve_index) return int(i);
  }
  return -1;
}

PetscErrorCode run_replay(const replay_args& args)
{
  PetscLogDouble replay_start = 0.0;
  PetscLogDouble replay_end = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&replay_start));

  if(args.snapshot_collection_path.empty()) {
    PetscFPrintf(
        PETSC_COMM_WORLD,
        stderr,
        "Usage: ksptune-petsc-replay -snapshot_collection snapshot_collection.csv "
        "[-replay_json_out result.json] [PETSc options]\n");
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

  replay_result aggregate;
  aggregate.repeat = args.repeat;
  aggregate.warmup = args.warmup;
  aggregate.snapshots = int(snapshots.size());
  aggregate.nullspace = nullspace_description(args);
  aggregate.nullspace_source = args.nullspace_mode == "from-metadata"
      ? "from-metadata"
      : (args.nullspace_mode == "none" ? "none" : "explicit");
  aggregate.nullspace_kind = "none";
  aggregate.nullspace_actions = "none";
  aggregate.field_nullspace_index = args.field_nullspace;
  aggregate.field_nullspace_block_size = args.field_nullspace >= 0
      ? args.field_nullspace_block_size
      : 0;

  std::vector<double> solve_times;
  std::vector<double> iteration_counts;
  for(const snapshot& snap : snapshots) {
    KSPTUNE_PETSC_CALL(replay_snapshot(args, snap, aggregate, solve_times, iteration_counts));
  }

  aggregate.solve_time_sec_total = std::accumulate(solve_times.begin(), solve_times.end(), 0.0);
  aggregate.solve_time_sec_mean = mean(solve_times);
  aggregate.solve_time_sec_median = median(solve_times);
  aggregate.solve_time_sec_min = minimum_value(solve_times);
  aggregate.solve_time_sec_max = maximum_value(solve_times);
  aggregate.solve_time_sec_stddev = population_stddev(solve_times);
  aggregate.solve_time_sec_range = value_range(solve_times);
  aggregate.solve_count = int(solve_times.size());
  aggregate.objective_time_sec_median = aggregate.solve_time_sec_median;
  aggregate.iterations_median = median(iteration_counts);
  aggregate.iterations_total = std::accumulate(iteration_counts.begin(), iteration_counts.end(), 0.0);

  std::vector<double> initial_residuals;
  std::vector<double> initial_relatives;
  std::vector<double> final_residuals;
  std::vector<double> final_relatives;
  for(const step_result& step : aggregate.steps) {
    initial_residuals.push_back(step.initial_true_residual_norm);
    initial_relatives.push_back(step.initial_true_relative_residual);
    final_residuals.push_back(step.final_true_residual_norm);
    final_relatives.push_back(step.final_true_relative_residual);
  }
  aggregate.initial_true_residual_norm_mean = mean(initial_residuals);
  aggregate.initial_true_relative_residual_mean = mean(initial_relatives);
  aggregate.final_true_residual_norm_mean = mean(final_residuals);
  aggregate.final_true_relative_residual_mean = mean(final_relatives);

  KSPTUNE_PETSC_CALL(PetscTime(&replay_end));
  double total_wall_time_sec = double(replay_end - replay_start);
  MPI_Allreduce(
      MPI_IN_PLACE,
      &total_wall_time_sec,
      1,
      MPI_DOUBLE,
      MPI_MAX,
      PETSC_COMM_WORLD);
  aggregate.total_wall_time_sec = total_wall_time_sec;
  collect_memory_diagnostics(aggregate);

  write_json_result(aggregate, args.json_output_path);
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if(ierr) return int(ierr);
  const replay_args args = parse_args(argc, argv);
  clear_replay_options();
  ierr = run_replay(args);
  const PetscErrorCode finalize_ierr = PetscFinalize();
  return int(ierr ? ierr : finalize_ierr);
}
