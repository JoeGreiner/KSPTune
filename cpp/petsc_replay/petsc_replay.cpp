#include <petscksp.h>
#include "json_io.hpp"
#include "snapshot_io.hpp"
#include "replay_cache.hpp"

#if defined(__has_include)
#if __has_include(<HYPRE_utilities.h>)
#include <HYPRE_utilities.h>
#define KSPTUNE_HAS_HYPRE_UTILITIES 1
#endif
#endif

#ifndef KSPTUNE_HAS_HYPRE_UTILITIES
#define KSPTUNE_HAS_HYPRE_UTILITIES 0
#endif

#if defined(__has_include)
#if __has_include(<petsc/private/kspimpl.h>)
#include <petsc/private/kspimpl.h>
#define KSPTUNE_HAS_PETSC_PRIVATE_KSPIMPL 1
#endif
#endif

#ifndef KSPTUNE_HAS_PETSC_PRIVATE_KSPIMPL
#define KSPTUNE_HAS_PETSC_PRIVATE_KSPIMPL 0
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

using ksptune::json;
using ksptune::snapshot;
using ksptune::read_snapshot_collection;
using ksptune::load_matrix_binary;
using ksptune::petsc_matrix;
using ksptune::petsc_vector;
using ksptune::petsc_ksp;
using ksptune::petsc_nullspace;
using ksptune::ksp_setup_context;
using ksptune::replay_cache;
using ksptune::matrix_memory_mb;
using ksptune::trim;

#define KSPTUNE_PETSC_CALL(call) \
  do { \
    PetscErrorCode ierr_ = (call); \
    if(ierr_) return ierr_; \
  } while(false)

struct replay_args {
  std::string snapshot_collection_path;
  std::string json_output_path;
  std::string initial_guess_vector_path;
  bool use_initial_guess = true;
  std::string nullspace_mode = "from-metadata";
  std::string nullspace_actions = "default";
  std::string petsc_options_key;
  int repeat = 1;
  int warmup = 0;
  int solve_target = -1;
  std::string snapshot_id;
  bool constant_nullspace = false;
  int field_nullspace = -1;
  int field_nullspace_block_size = 2;
  bool diagnose_matrix = false;
  bool reuse_ksp_setup = true;
  bool replay_server = false;
  bool options_help = false;
  bool hypre_hierarchy_diagnostics = false;
  double cache_memory_mb = 0.0;
  double soft_timeout_sec = -1.0;
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

struct matrix_diagnostics {
  PetscInt row_count = 0;
  PetscInt column_count = 0;
  PetscInt local_row_count = 0;
  PetscInt local_column_count = 0;
  double nonzero_count = 0.0;
  double allocated_nonzero_count = 0.0;
  double petsc_matrix_memory_bytes = 0.0;
  double estimated_matrix_memory_bytes = 0.0;
  bool symmetric_tested = false;
  bool symmetric = false;
};

struct memory_sample {
  std::string phase;
  int solve_index = -1;
  std::string snapshot_id;
  double rss_mb_sum = 0.0;
  double rss_mb_max_rank = 0.0;
  double rss_mb_max_node = 0.0;
};

struct pc_diagnostic_level {
  int level = 0;
  int level_from_finest = 0;
  std::map<std::string, double> metrics;
  std::map<std::string, std::string> string_metrics;
};

struct pc_diagnostic_record {
  std::string pc_type;
  std::string setup_key;
  int setup_index = 0;
  std::map<std::string, double> metrics;
  std::map<std::string, std::string> string_metrics;
  std::vector<pc_diagnostic_level> levels;
};

struct step_result {
  int repeat_index = 0;
  int solve_index = 0;
  std::string snapshot_id;
  double solve_time_sec = 0.0;
  double initial_true_residual_norm = 0.0;
  double rhs_norm = 0.0;
  double initial_true_relative_residual = 0.0;
  double final_true_residual_norm = 0.0;
  double final_true_relative_residual = 0.0;
  double initial_ksp_residual_norm = 0.0;
  double final_ksp_residual_norm = 0.0;
  double ksp_rtol_reference_norm = 0.0;
  double ksp_convergence_threshold_norm = 0.0;
  double final_ksp_relative_residual_norm = 0.0;
  int iterations = 0;
  int reason_code = 0;
};

struct replay_result {
  double objective_time_sec_median = 1.0e30;
  double total_wall_time_sec = 0.0;
  double matrix_load_time_sec = 0.0;
  double matrix_prepare_time_sec = 0.0;
  double vector_prepare_time_sec = 0.0;
  double nullspace_time_sec = 0.0;
  double true_residual_time_sec = 0.0;
  bool use_initial_guess = true;
  double soft_timeout_sec = -1.0;
  double soft_timeout_elapsed_sec = 0.0;
  bool soft_timeout_triggered = false;
  int matrix_cache_hits = 0;
  int matrix_cache_misses = 0;
  double replay_cache_memory_mb = 0.0;
  double matrix_cache_memory_mb = 0.0;
  double vector_cache_memory_mb = 0.0;
  double replay_cache_memory_limit_mb = 0.0;
  int replay_cache_evictions = 0;
  double solver_setup_time_sec = 0.0;
  double solver_setup_time_sec_actual = 0.0;
  double solver_setup_time_sec_logical = 0.0;
  int ksp_setup_cache_hits = 0;
  int ksp_setup_cache_misses = 0;
  double solve_time_sec_total = 0.0;
  double solve_time_sec_mean = 0.0;
  double solve_time_sec_median = 0.0;
  double solve_time_sec_min = 0.0;
  double solve_time_sec_max = 0.0;
  double solve_time_sec_stddev = 0.0;
  double solve_time_sec_sem = 0.0;
  double solve_time_sec_relative_sem = 0.0;
  double solve_time_sec_range = 0.0;
  double solve_mpi_message_count = 0.0;
  double solve_mpi_message_bytes = 0.0;
  double solve_mpi_message_bytes_mean = 0.0;
  double solve_mpi_reduction_count = 0.0;
  double initial_true_residual_norm_mean = 0.0;
  double initial_true_relative_residual_mean = 0.0;
  double final_true_residual_norm_mean = 0.0;
  double final_true_relative_residual_mean = 0.0;
  double initial_ksp_residual_norm_mean = 0.0;
  double final_ksp_residual_norm_mean = 0.0;
  double ksp_rtol_reference_norm_mean = 0.0;
  double ksp_convergence_threshold_norm_mean = 0.0;
  double final_ksp_relative_residual_norm_mean = 0.0;
  std::string ksp_residual_norm_type;
  std::string ksp_rtol_reference_source;
  double ksp_rtol = 0.0;
  double ksp_atol = 0.0;
  double ksp_dtol = 0.0;
  int ksp_max_it = 0;
  std::vector<memory_sample> memory_samples;
  double rss_request_start_mb_sum = 0.0;
  double rss_request_end_mb_sum = 0.0;
  double rss_request_peak_sample_mb_sum = 0.0;
  double rss_request_delta_mb_sum = 0.0;
  double rss_setup_delta_mb_sum = 0.0;
  double rss_solve_delta_mb_sum = 0.0;
  double rss_peak_sample_mb_max_rank = 0.0;
  double rss_peak_sample_mb_max_node = 0.0;
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
  std::vector<pc_diagnostic_record> pc_diagnostics;
  std::vector<matrix_diagnostics> matrices;
  std::vector<step_result> steps;
  std::vector<std::string> recorded_pc_diagnostic_keys;
};

struct replay_snapshot_plan {
  replay_nullspace_choice nullspace_choice;
  nullspace_actions actions;
  std::string matrix_cache_key;
  int matrix_cache_key_count = 0;
  std::string ksp_setup_cache_key;
  int ksp_setup_cache_key_count = 0;
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

replay_args parse_args(int argc, char** argv)
{
  replay_args args;
  arg_value(argc, argv, "-snapshot_collection", args.snapshot_collection_path);
  arg_value(argc, argv, "-replay_json_out", args.json_output_path);
  arg_value(argc, argv, "-replay_initial_guess_vec", args.initial_guess_vector_path);
  args.use_initial_guess = bool_arg(argc, argv, "-replay_use_initial_guess", true);
  arg_value(argc, argv, "-replay_nullspace", args.nullspace_mode);
  arg_value(argc, argv, "-replay_nullspace_actions", args.nullspace_actions);
  args.repeat = std::max(1, int_arg(argc, argv, "-replay_repeat", args.repeat));
  args.warmup = std::max(0, int_arg(argc, argv, "-replay_warmup", args.warmup));
  args.solve_target = int_arg(argc, argv, "-replay_solve_target", args.solve_target);
  arg_value(argc, argv, "-replay_snapshot_id", args.snapshot_id);
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
  args.reuse_ksp_setup = bool_arg(argc, argv, "-replay_reuse_ksp_setup", true);
  args.replay_server = bool_arg(argc, argv, "-replay_server", false);
  args.options_help = bool_arg(argc, argv, "-replay_options_help", false);
  args.hypre_hierarchy_diagnostics = bool_arg(
      argc,
      argv,
      "-replay_hypre_hierarchy_diagnostics",
      false);
  args.cache_memory_mb = std::max(
      0.0,
      double_arg(argc, argv, "-replay_cache_memory_mb", args.cache_memory_mb));
  args.soft_timeout_sec = double_arg(
      argc,
      argv,
      "-replay_soft_timeout_sec",
      args.soft_timeout_sec);
  return args;
}

void clear_replay_options()
{
  const char* keys[] = {
    "-snapshot_collection",
    "-replay_json_out",
    "-replay_initial_guess_vec",
    "-replay_use_initial_guess",
    "-replay_repeat",
    "-replay_warmup",
    "-replay_solve_target",
    "-replay_snapshot_id",
    "-replay_nullspace",
    "-replay_nullspace_actions",
    "-replay_constant_nullspace",
    "-replay_field_nullspace",
    "-replay_field_nullspace_block_size",
    "-replay_diagnose_matrix",
    "-replay_reuse_ksp_setup",
    "-replay_server",
    "-replay_options_help",
    "-replay_hypre_hierarchy_diagnostics",
    "-replay_cache_memory_mb",
    "-replay_soft_timeout_sec",
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

std::string matrix_cache_key(const snapshot& snap,
                             const replay_nullspace_choice& nullspace_choice,
                             const nullspace_actions& actions)
{
  std::ostringstream key;
  key << snap.matrix_file_path << '\n'
      << "symmetric=" << (snap.symmetric ? 1 : 0) << '\n'
      << "spd=" << (snap.spd ? 1 : 0) << '\n'
      << "nullspace_kind=" << nullspace_choice.kind << '\n'
      << "nullspace_field_index=" << nullspace_choice.field_index << '\n'
      << "nullspace_block_size=" << nullspace_choice.block_size << '\n'
      << "matrix_nullspace=" << (actions.matrix ? 1 : 0) << '\n'
      << "transpose_nullspace=" << (actions.transpose ? 1 : 0) << '\n'
      << "near_nullspace=" << (actions.near ? 1 : 0);
  return key.str();
}

std::string ksp_setup_cache_key(const snapshot& snap,
                                const std::string& matrix_key,
                                const std::string& petsc_options_key)
{
  std::ostringstream key;
  key << matrix_key << '\n'
      << "petsc_options=" << petsc_options_key << '\n'
      << "rows=" << snap.rows << '\n'
      << "cols=" << snap.cols << '\n'
      << "rhs_size=" << snap.right_hand_side_size << '\n'
      << "rtol=" << std::setprecision(17) << snap.relative_tolerance << '\n'
      << "atol=" << snap.absolute_tolerance << '\n'
      << "dtol=" << snap.divergence_tolerance << '\n'
      << "max_iterations=" << snap.max_iterations;
  return key.str();
}

PetscErrorCode build_replay_snapshot_plan(const replay_args& args,
                                          const snapshot& snap,
                                          replay_snapshot_plan& plan)
{
  KSPTUNE_PETSC_CALL(resolve_replay_nullspace_choice(args, snap, plan.nullspace_choice));
  KSPTUNE_PETSC_CALL(resolve_nullspace_actions(
      args,
      snap,
      plan.nullspace_choice.explicit_nullspace,
      plan.actions));
  plan.matrix_cache_key = matrix_cache_key(snap, plan.nullspace_choice, plan.actions);
  plan.ksp_setup_cache_key = ksp_setup_cache_key(
      snap,
      plan.matrix_cache_key,
      args.petsc_options_key);
  return 0;
}

double matrix_memory_estimate_bytes(Mat matrix)
{
  return matrix_memory_mb(matrix) * 1024.0 * 1024.0;
}

#if defined(__linux__)
double linux_status_memory_mb(const std::string& field_name)
{
  std::ifstream status("/proc/self/status");
  std::string line;
  while(std::getline(status, line)) {
    if(line.rfind(field_name, 0) != 0) continue;
    std::istringstream value_stream(line.substr(field_name.size()));
    double value_kb = 0.0;
    std::string unit;
    value_stream >> value_kb >> unit;
    if(value_stream) return value_kb / 1024.0;
  }
  return -1.0;
}
#endif

double local_current_memory_mb()
{
#if defined(__APPLE__) && defined(__MACH__)
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if(task_info(
         mach_task_self(),
         MACH_TASK_BASIC_INFO,
         reinterpret_cast<task_info_t>(&info),
         &count) != KERN_SUCCESS) {
    return 0.0;
  }
  return double(info.resident_size) / (1024.0 * 1024.0);
#elif defined(__linux__)
  const double vmrss_mb = linux_status_memory_mb("VmRSS:");
  if(vmrss_mb >= 0.0) return vmrss_mb;
  std::ifstream statm("/proc/self/statm");
  long size_pages = 0;
  long resident_pages = 0;
  statm >> size_pages >> resident_pages;
  const long page_size = sysconf(_SC_PAGESIZE);
  if(!statm || page_size <= 0) return 0.0;
  return double(resident_pages) * double(page_size) / (1024.0 * 1024.0);
#else
  return 0.0;
#endif
}

struct memory_tracker {
  MPI_Comm communicator = PETSC_COMM_WORLD;
  MPI_Comm node_communicator = MPI_COMM_NULL;
  ~memory_tracker() { destroy(); }

  PetscErrorCode initialize(MPI_Comm input_communicator)
  {
    communicator = input_communicator;
    const int split_error = MPI_Comm_split_type(
        communicator,
        MPI_COMM_TYPE_SHARED,
        0,
        MPI_INFO_NULL,
        &node_communicator);
    return split_error == MPI_SUCCESS ? 0 : PETSC_ERR_LIB;
  }

  PetscErrorCode sample(replay_result& result,
                        const std::string& phase,
                        int solve_index = -1,
                        const std::string& snapshot_id = "") const
  {
    memory_sample current_sample;
    current_sample.phase = phase;
    current_sample.solve_index = solve_index;
    current_sample.snapshot_id = snapshot_id;

    const double local_rss_mb = local_current_memory_mb();
    double node_rss_mb_sum = local_rss_mb;
    if(node_communicator != MPI_COMM_NULL) {
      MPI_Allreduce(
          MPI_IN_PLACE,
          &node_rss_mb_sum,
          1,
          MPI_DOUBLE,
          MPI_SUM,
          node_communicator);
    }

    MPI_Allreduce(
        &local_rss_mb,
        &current_sample.rss_mb_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        communicator);
    double local_max_values[2] = {local_rss_mb, node_rss_mb_sum};
    double global_max_values[2] = {0.0, 0.0};
    MPI_Allreduce(
        local_max_values,
        global_max_values,
        2,
        MPI_DOUBLE,
        MPI_MAX,
        communicator);
    current_sample.rss_mb_max_rank = global_max_values[0];
    current_sample.rss_mb_max_node = global_max_values[1];

    result.memory_samples.push_back(current_sample);
    return 0;
  }

  PetscErrorCode destroy()
  {
    if(node_communicator != MPI_COMM_NULL) {
      const int free_error = MPI_Comm_free(&node_communicator);
      node_communicator = MPI_COMM_NULL;
      if(free_error != MPI_SUCCESS) return PETSC_ERR_LIB;
    }
    return 0;
  }
};

double nonnegative_memory_delta(double after, double before)
{
  return std::max(0.0, after - before);
}

void summarize_request_memory(replay_result& result)
{
  if(result.memory_samples.empty()) return;

  const memory_sample* request_start = &result.memory_samples.front();
  const memory_sample* request_end = &result.memory_samples.back();
  std::map<std::string, double> before_setup_by_snapshot;
  std::map<std::string, double> before_solve_by_snapshot;

  for(const memory_sample& sample : result.memory_samples) {
    if(sample.phase == "request_start") request_start = &sample;
    if(sample.phase == "request_end") request_end = &sample;
    result.rss_request_peak_sample_mb_sum =
        std::max(result.rss_request_peak_sample_mb_sum, sample.rss_mb_sum);
    result.rss_peak_sample_mb_max_rank = std::max(
        result.rss_peak_sample_mb_max_rank,
        sample.rss_mb_max_rank);
    result.rss_peak_sample_mb_max_node = std::max(
        result.rss_peak_sample_mb_max_node,
        sample.rss_mb_max_node);

    if(sample.solve_index < 0) continue;
    if(sample.phase == "before_setup") {
      before_setup_by_snapshot[sample.snapshot_id] = sample.rss_mb_sum;
    } else if(sample.phase == "after_setup") {
      const auto before = before_setup_by_snapshot.find(sample.snapshot_id);
      if(before != before_setup_by_snapshot.end()) {
        result.rss_setup_delta_mb_sum = std::max(
            result.rss_setup_delta_mb_sum,
            nonnegative_memory_delta(sample.rss_mb_sum, before->second));
      }
    } else if(sample.phase == "before_solve") {
      before_solve_by_snapshot[sample.snapshot_id] = sample.rss_mb_sum;
    } else if(sample.phase == "after_solve") {
      const auto before = before_solve_by_snapshot.find(sample.snapshot_id);
      if(before != before_solve_by_snapshot.end()) {
        result.rss_solve_delta_mb_sum = std::max(
            result.rss_solve_delta_mb_sum,
            nonnegative_memory_delta(sample.rss_mb_sum, before->second));
      }
    }
  }

  result.rss_request_start_mb_sum = request_start->rss_mb_sum;
  result.rss_request_end_mb_sum = request_end->rss_mb_sum;
  result.rss_request_delta_mb_sum = nonnegative_memory_delta(
      result.rss_request_end_mb_sum,
      result.rss_request_start_mb_sum);
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
  diagnostics.estimated_matrix_memory_bytes = matrix_memory_estimate_bytes(matrix);

  if(diagnose_matrix) {
    PetscBool symmetric = PETSC_FALSE;
    KSPTUNE_PETSC_CALL(MatIsSymmetric(matrix, 1.0e-12, &symmetric));
    diagnostics.symmetric_tested = true;
    diagnostics.symmetric = symmetric == PETSC_TRUE;
  }
  return 0;
}

PetscErrorCode compute_residual_norm(Mat matrix,
                                     Vec right_hand_side,
                                     Vec solution,
                                     double& residual_norm,
                                     double& rhs_norm,
                                     double& relative_residual)
{
  petsc_vector matrix_times_solution;
  petsc_vector residual_vector;
  KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, matrix_times_solution.ptr()));
  KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, residual_vector.ptr()));
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

  KSPTUNE_PETSC_CALL(residual_vector.reset());
  KSPTUNE_PETSC_CALL(matrix_times_solution.reset());
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

  petsc_vector basis;
  KSPTUNE_PETSC_CALL(VecDuplicate(reference_vector, basis.ptr()));

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
    basis.reset();
    PetscFPrintf(PETSC_COMM_WORLD, stderr, "Field nullspace basis is empty\n");
    return PETSC_ERR_ARG_WRONGSTATE;
  }

  KSPTUNE_PETSC_CALL(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 1, basis.ptr(), nullspace));
  KSPTUNE_PETSC_CALL(basis.reset());
  return 0;
}

PetscErrorCode create_replay_nullspace(const replay_nullspace_choice& choice,
                                       Vec reference_vector,
                                       MatNullSpace* nullspace)
{
  *nullspace = nullptr;
  if(choice.kind == "constant") {
    KSPTUNE_PETSC_CALL(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, nullptr, nullspace));
  } else if(choice.kind == "field" && choice.field_index >= 0) {
    KSPTUNE_PETSC_CALL(create_field_nullspace(
        reference_vector,
        PetscInt(choice.block_size),
        PetscInt(choice.field_index),
        nullspace));
  }
  return 0;
}

PetscErrorCode apply_nullspace_actions(Mat matrix,
                                       Vec right_hand_side,
                                       MatNullSpace nullspace,
                                       const nullspace_actions& actions,
                                       replay_result& aggregate,
                                       bool apply_matrix_actions)
{
  if(actions.matrix) {
    if(apply_matrix_actions) KSPTUNE_PETSC_CALL(MatSetNullSpace(matrix, nullspace));
    ++aggregate.matrix_nullspace_attached_count;
  }
  if(actions.transpose) {
    if(apply_matrix_actions) KSPTUNE_PETSC_CALL(MatSetTransposeNullSpace(matrix, nullspace));
    ++aggregate.transpose_nullspace_attached_count;
  }
  if(actions.near) {
    if(apply_matrix_actions) KSPTUNE_PETSC_CALL(MatSetNearNullSpace(matrix, nullspace));
    ++aggregate.near_nullspace_attached_count;
  }
  if(actions.remove_rhs) {
    petsc_vector projected;
    petsc_vector removed_component;
    KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, projected.ptr()));
    KSPTUNE_PETSC_CALL(VecCopy(right_hand_side, projected));
    KSPTUNE_PETSC_CALL(MatNullSpaceRemove(nullspace, projected));

    KSPTUNE_PETSC_CALL(VecDuplicate(right_hand_side, removed_component.ptr()));
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

    KSPTUNE_PETSC_CALL(removed_component.reset());
    KSPTUNE_PETSC_CALL(projected.reset());
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

double standard_error_of_mean(const std::vector<double>& values)
{
  if(values.size() < 2) return 0.0;
  const double values_mean = mean(values);
  double squared_delta_sum = 0.0;
  for(double value : values) {
    const double delta = value - values_mean;
    squared_delta_sum += delta * delta;
  }
  return std::sqrt(squared_delta_sum / (double(values.size()) * double(values.size() - 1)));
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

struct scoped_petsc_option {
  std::string key;
  std::string previous_value;
  bool previous_was_set = false;
  bool active = false;

  PetscErrorCode set(const char* option, const char* value)
  {
    key = option;
    char buffer[4096] = "";
    PetscBool was_set = PETSC_FALSE;
    KSPTUNE_PETSC_CALL(PetscOptionsGetString(
        nullptr,
        nullptr,
        key.c_str(),
        buffer,
        sizeof(buffer),
        &was_set));
    previous_was_set = was_set == PETSC_TRUE;
    previous_value = previous_was_set ? buffer : "";
    KSPTUNE_PETSC_CALL(PetscOptionsSetValue(nullptr, key.c_str(), value));
    active = true;
    return 0;
  }

  PetscErrorCode restore()
  {
    if(!active) return 0;
    active = false;
    if(previous_was_set) {
      KSPTUNE_PETSC_CALL(PetscOptionsSetValue(
          nullptr,
          key.c_str(),
          previous_value.c_str()));
    } else {
      KSPTUNE_PETSC_CALL(PetscOptionsClearValue(nullptr, key.c_str()));
    }
    return 0;
  }

  ~scoped_petsc_option()
  {
    if(active) {
      (void)restore();
    }
  }
};

PetscErrorCode clear_hypre_errors()
{
#if KSPTUNE_HAS_HYPRE_UTILITIES
  HYPRE_ClearAllErrors();
#endif
  return 0;
}

PetscErrorCode create_configured_ksp(const replay_args& args,
                                     const snapshot& snap,
                                     Mat matrix,
                                     KSP* ksp)
{
  KSPTUNE_PETSC_CALL(KSPCreate(PETSC_COMM_WORLD, ksp));
  KSPTUNE_PETSC_CALL(KSPSetOperators(*ksp, matrix, matrix));
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
    KSPTUNE_PETSC_CALL(KSPSetTolerances(*ksp, rtol, atol, dtol, max_it));
  }
  KSPTUNE_PETSC_CALL(KSPSetInitialGuessNonzero(
      *ksp,
      args.use_initial_guess ? PETSC_TRUE : PETSC_FALSE));
  KSPTUNE_PETSC_CALL(KSPSetReusePreconditioner(*ksp, PETSC_TRUE));
  scoped_petsc_option hypre_view_option;
  if(args.hypre_hierarchy_diagnostics) {
    KSPTUNE_PETSC_CALL(hypre_view_option.set(
        "-pc_hypre_boomeramg_view_hierarchy",
        "true"));
  }
  const PetscErrorCode set_from_options_ierr = KSPSetFromOptions(*ksp);
  const PetscErrorCode restore_ierr = hypre_view_option.restore();
  if(set_from_options_ierr || restore_ierr) {
    (void)KSPDestroy(ksp);
    if(set_from_options_ierr) return set_from_options_ierr;
    return restore_ierr;
  }
  KSPTUNE_PETSC_CALL(KSPSetInitialGuessNonzero(
      *ksp,
      args.use_initial_guess ? PETSC_TRUE : PETSC_FALSE));
  return 0;
}

PetscErrorCode run_options_help()
{
  KSPTUNE_PETSC_CALL(PetscOptionsSetValue(nullptr, "-help", "true"));

  petsc_matrix matrix;
  petsc_ksp ksp;
  KSPTUNE_PETSC_CALL(MatCreate(PETSC_COMM_WORLD, matrix.ptr()));
  KSPTUNE_PETSC_CALL(MatSetSizes(matrix, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  KSPTUNE_PETSC_CALL(MatSetFromOptions(matrix));
  KSPTUNE_PETSC_CALL(MatSetUp(matrix));

  PetscInt row_start = 0;
  PetscInt row_end = 0;
  KSPTUNE_PETSC_CALL(MatGetOwnershipRange(matrix, &row_start, &row_end));
  if(row_start <= 0 && row_end > 0) {
    KSPTUNE_PETSC_CALL(MatSetValue(matrix, 0, 0, PetscScalar(1.0), INSERT_VALUES));
  }
  KSPTUNE_PETSC_CALL(MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY));
  KSPTUNE_PETSC_CALL(MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY));

  KSPTUNE_PETSC_CALL(KSPCreate(PETSC_COMM_WORLD, ksp.ptr()));
  KSPTUNE_PETSC_CALL(KSPSetOperators(ksp, matrix, matrix));
  KSPTUNE_PETSC_CALL(KSPSetFromOptions(ksp));
  KSPTUNE_PETSC_CALL(ksp.reset());
  KSPTUNE_PETSC_CALL(matrix.reset());
  return 0;
}

PetscErrorCode record_ksp_types(KSP ksp, replay_result& aggregate)
{
  KSPType ksp_type = nullptr;
  PCType pc_type = nullptr;
  PC pc = nullptr;
  KSPTUNE_PETSC_CALL(KSPGetType(ksp, &ksp_type));
  KSPTUNE_PETSC_CALL(KSPGetPC(ksp, &pc));
  KSPTUNE_PETSC_CALL(PCGetType(pc, &pc_type));
  aggregate.ksp_type = ksp_type ? ksp_type : "";
  aggregate.pc_type = pc_type ? pc_type : "";
  return 0;
}

std::string ksp_norm_type_name(KSPNormType norm_type)
{
  switch(norm_type) {
  case KSP_NORM_DEFAULT:
    return "default";
  case KSP_NORM_NONE:
    return "none";
  case KSP_NORM_PRECONDITIONED:
    return "preconditioned";
  case KSP_NORM_UNPRECONDITIONED:
    return "unpreconditioned";
  case KSP_NORM_NATURAL:
    return "natural";
  default:
    return std::string("unknown(") + std::to_string(int(norm_type)) + ")";
  }
}

PetscErrorCode record_ksp_residual_norm_type(KSP ksp, replay_result& aggregate)
{
  KSPNormType norm_type = KSP_NORM_DEFAULT;
  KSPTUNE_PETSC_CALL(KSPGetNormType(ksp, &norm_type));
  aggregate.ksp_residual_norm_type = ksp_norm_type_name(norm_type);
  return 0;
}

PetscErrorCode record_ksp_tolerances(KSP ksp, replay_result& aggregate)
{
  PetscReal rtol = 0.0;
  PetscReal atol = 0.0;
  PetscReal dtol = 0.0;
  PetscInt max_it = 0;
  KSPTUNE_PETSC_CALL(KSPGetTolerances(ksp, &rtol, &atol, &dtol, &max_it));
  aggregate.ksp_rtol = double(rtol);
  aggregate.ksp_atol = double(atol);
  aggregate.ksp_dtol = double(dtol);
  aggregate.ksp_max_it = int(max_it);
  return 0;
}

double ratio_or_zero(double numerator, double denominator)
{
  return denominator > 0.0 ? numerator / denominator : 0.0;
}

void set_metric(std::map<std::string, double>& metrics,
                const std::string& name,
                double value)
{
  metrics[name] = value;
}

void update_max_metric(std::map<std::string, double>& metrics,
                       const std::string& name,
                       double value)
{
  const auto entry = metrics.find(name);
  if(entry == metrics.end() || value > entry->second) metrics[name] = value;
}

void update_min_positive_metric(std::map<std::string, double>& metrics,
                                const std::string& name,
                                double value)
{
  if(value <= 0.0) return;
  const auto entry = metrics.find(name);
  if(entry == metrics.end() || entry->second <= 0.0 || value < entry->second) {
    metrics[name] = value;
  }
}

double metric_or_zero(const std::map<std::string, double>& metrics,
                      const std::string& name)
{
  const auto entry = metrics.find(name);
  return entry == metrics.end() ? 0.0 : entry->second;
}

bool replay_result_has_pc_diagnostic_key(const replay_result& result,
                                         const std::string& key)
{
  return std::find(
      result.recorded_pc_diagnostic_keys.begin(),
      result.recorded_pc_diagnostic_keys.end(),
      key) != result.recorded_pc_diagnostic_keys.end();
}

PetscErrorCode collect_gamg_diagnostics(PC pc, pc_diagnostic_record& record)
{
  PetscInt levels = 0;
  KSPTUNE_PETSC_CALL(PCMGGetLevels(pc, &levels));
  if(levels <= 0) return 0;

  PetscReal grid_complexity = 0.0;
  PetscReal operator_complexity = 0.0;
  KSPTUNE_PETSC_CALL(PCMGGetGridComplexity(pc, &grid_complexity, &operator_complexity));

  double total_rows = 0.0;
  double total_nonzeros = 0.0;
  double finest_rows = 0.0;
  double finest_nonzeros = 0.0;
  double coarsest_rows = 0.0;
  double coarsest_nonzeros = 0.0;
  double interpolation_nonzeros = 0.0;

  for(PetscInt level = 0; level < levels; ++level) {
    pc_diagnostic_level hierarchy_level;
    hierarchy_level.level = int(level);
    hierarchy_level.level_from_finest = int(levels - 1 - level);

    KSP smoother = nullptr;
    KSPTUNE_PETSC_CALL(PCMGGetSmoother(pc, level, &smoother));
    if(smoother) {
      Mat level_matrix = nullptr;
      KSPTUNE_PETSC_CALL(KSPGetOperators(smoother, nullptr, &level_matrix));
      if(level_matrix) {
        PetscInt rows = 0;
        PetscInt row_start = 0;
        PetscInt row_end = 0;
        MatInfo info;
        KSPTUNE_PETSC_CALL(MatGetSize(level_matrix, &rows, nullptr));
        KSPTUNE_PETSC_CALL(MatGetOwnershipRange(level_matrix, &row_start, &row_end));
        KSPTUNE_PETSC_CALL(MatGetInfo(level_matrix, MAT_GLOBAL_SUM, &info));
        double active_ranks = row_end > row_start ? 1.0 : 0.0;
        MPI_Allreduce(MPI_IN_PLACE, &active_ranks, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

        set_metric(hierarchy_level.metrics, "rows", double(rows));
        set_metric(hierarchy_level.metrics, "nonzeros", double(info.nz_used));
        set_metric(
            hierarchy_level.metrics,
            "avg_nonzeros_per_row",
            ratio_or_zero(info.nz_used, double(rows)));
        set_metric(hierarchy_level.metrics, "active_ranks", active_ranks);
        update_max_metric(
            record.metrics,
            "max_operator_nonzeros_per_row",
            metric_or_zero(hierarchy_level.metrics, "avg_nonzeros_per_row"));
        update_min_positive_metric(record.metrics, "min_active_ranks", active_ranks);

        total_rows += double(rows);
        total_nonzeros += double(info.nz_used);
        if(level == 0) {
          coarsest_rows = double(rows);
          coarsest_nonzeros = double(info.nz_used);
        }
        if(level == levels - 1) {
          finest_rows = double(rows);
          finest_nonzeros = double(info.nz_used);
        }
      }
    }
    if(level > 0) {
      Mat interpolation = nullptr;
      KSPTUNE_PETSC_CALL(PCMGGetInterpolation(pc, level, &interpolation));
      if(interpolation) {
        PetscInt interpolation_rows = 0;
        MatInfo info;
        KSPTUNE_PETSC_CALL(MatGetSize(interpolation, &interpolation_rows, nullptr));
        KSPTUNE_PETSC_CALL(MatGetInfo(interpolation, MAT_GLOBAL_SUM, &info));
        set_metric(
            hierarchy_level.metrics,
            "interpolation_nonzeros",
            double(info.nz_used));
        set_metric(
            hierarchy_level.metrics,
            "interpolation_avg_nonzeros_per_row",
            ratio_or_zero(info.nz_used, double(interpolation_rows)));
        update_max_metric(
            record.metrics,
            "max_interpolation_nonzeros_per_row",
            metric_or_zero(
                hierarchy_level.metrics,
                "interpolation_avg_nonzeros_per_row"));
        interpolation_nonzeros += double(info.nz_used);
      }
    }
    record.levels.push_back(hierarchy_level);
  }

  set_metric(record.metrics, "levels", double(levels));
  set_metric(record.metrics, "grid_complexity", double(grid_complexity));
  set_metric(record.metrics, "operator_complexity", double(operator_complexity));
  set_metric(record.metrics, "finest_rows", finest_rows);
  set_metric(record.metrics, "finest_nonzeros", finest_nonzeros);
  set_metric(record.metrics, "coarsest_rows", coarsest_rows);
  set_metric(record.metrics, "coarsest_nonzeros", coarsest_nonzeros);
  set_metric(record.metrics, "total_rows", total_rows);
  set_metric(record.metrics, "total_nonzeros", total_nonzeros);
  set_metric(record.metrics, "interpolation_nonzeros", interpolation_nonzeros);
  return 0;
}

struct hypre_diagnostics {
  std::map<std::string, double> metrics;
  std::vector<pc_diagnostic_level> levels;
};

bool line_starts_with(const std::string& line, const std::string& prefix)
{
  return line.size() >= prefix.size() &&
      line.compare(0, prefix.size(), prefix) == 0;
}

bool parse_labeled_double(const std::string& line,
                          const std::string& label,
                          double& value)
{
  if(!line_starts_with(line, label)) return false;
  std::stringstream stream(line.substr(label.size()));
  return bool(stream >> value);
}

hypre_diagnostics parse_hypre_view_text(const std::string& text)
{
  hypre_diagnostics diagnostics;
  std::stringstream input(text);
  std::string line;
  bool in_hierarchy = false;
  int finest_level = -1;
  int coarsest_level = -1;
  std::map<int, pc_diagnostic_level> hierarchy_by_level;

  while(std::getline(input, line)) {
    line = trim(line);
    if(line.find("BoomerAMG hierarchy statistics") != std::string::npos) {
      in_hierarchy = true;
      continue;
    }
    if(!in_hierarchy || line.empty()) continue;

    double value = 0.0;
    if(parse_labeled_double(line, "grid complexity", value)) {
      set_metric(diagnostics.metrics, "grid_complexity", value);
      continue;
    }
    if(parse_labeled_double(line, "operator complexity", value)) {
      set_metric(diagnostics.metrics, "operator_complexity", value);
      continue;
    }
    if(parse_labeled_double(line, "interpolation complexity", value)) {
      set_metric(diagnostics.metrics, "interpolation_complexity", value);
      continue;
    }

    std::stringstream stream(line);
    std::string op;
    int level = -1;
    double rows = 0.0;
    double nonzeros = 0.0;
    double nonzeros_per_row = 0.0;
    double max_row_nonzeros = 0.0;
    double offdiag_nonzeros = 0.0;
    if(!(stream >> op >> level >> rows >> nonzeros >> nonzeros_per_row >>
         max_row_nonzeros >> offdiag_nonzeros)) {
      continue;
    }
    if(op != "A" && op != "P") continue;

    update_max_metric(diagnostics.metrics, "max_row_nonzeros", max_row_nonzeros);
    diagnostics.metrics["offdiag_nonzeros"] += offdiag_nonzeros;

    pc_diagnostic_level& hierarchy_level = hierarchy_by_level[level];
    hierarchy_level.level = level;
    hierarchy_level.level_from_finest = level;

    if(op == "A") {
      set_metric(hierarchy_level.metrics, "rows", rows);
      set_metric(hierarchy_level.metrics, "nonzeros", nonzeros);
      set_metric(hierarchy_level.metrics, "avg_nonzeros_per_row", nonzeros_per_row);
      set_metric(hierarchy_level.metrics, "max_row_nonzeros", max_row_nonzeros);
      set_metric(hierarchy_level.metrics, "offdiag_nonzeros", offdiag_nonzeros);
      update_max_metric(
          diagnostics.metrics,
          "max_operator_nonzeros_per_row",
          nonzeros_per_row);
      diagnostics.metrics["total_rows"] += rows;
      diagnostics.metrics["total_nonzeros"] += nonzeros;
      if(finest_level < 0 || level < finest_level) {
        finest_level = level;
        set_metric(diagnostics.metrics, "finest_rows", rows);
        set_metric(diagnostics.metrics, "finest_nonzeros", nonzeros);
      }
      if(level > coarsest_level) {
        coarsest_level = level;
        set_metric(diagnostics.metrics, "coarsest_rows", rows);
        set_metric(diagnostics.metrics, "coarsest_nonzeros", nonzeros);
      }
    } else {
      set_metric(hierarchy_level.metrics, "interpolation_nonzeros", nonzeros);
      set_metric(
          hierarchy_level.metrics,
          "interpolation_avg_nonzeros_per_row",
          nonzeros_per_row);
      update_max_metric(
          diagnostics.metrics,
          "max_interpolation_nonzeros_per_row",
          nonzeros_per_row);
      diagnostics.metrics["interpolation_nonzeros"] += nonzeros;
    }
  }

  if(coarsest_level < 0) return diagnostics;
  set_metric(diagnostics.metrics, "levels", double(coarsest_level + 1));
  if(metric_or_zero(diagnostics.metrics, "grid_complexity") <= 0.0 &&
     metric_or_zero(diagnostics.metrics, "finest_rows") > 0.0) {
    set_metric(
        diagnostics.metrics,
        "grid_complexity",
        metric_or_zero(diagnostics.metrics, "total_rows") /
            metric_or_zero(diagnostics.metrics, "finest_rows"));
  }
  if(metric_or_zero(diagnostics.metrics, "operator_complexity") <= 0.0 &&
     metric_or_zero(diagnostics.metrics, "finest_nonzeros") > 0.0) {
    set_metric(
        diagnostics.metrics,
        "operator_complexity",
        metric_or_zero(diagnostics.metrics, "total_nonzeros") /
            metric_or_zero(diagnostics.metrics, "finest_nonzeros"));
  }
  if(metric_or_zero(diagnostics.metrics, "interpolation_complexity") <= 0.0 &&
     metric_or_zero(diagnostics.metrics, "finest_nonzeros") > 0.0) {
    set_metric(
        diagnostics.metrics,
        "interpolation_complexity",
        metric_or_zero(diagnostics.metrics, "interpolation_nonzeros") /
            metric_or_zero(diagnostics.metrics, "finest_nonzeros"));
  }
  for(auto& entry : hierarchy_by_level) {
    diagnostics.levels.push_back(entry.second);
  }
  return diagnostics;
}

std::string hypre_view_temporary_path(const std::string& json_output_path)
{
  if(json_output_path.empty()) return "";
  return json_output_path + ".hypre_view.tmp";
}

PetscErrorCode capture_pc_view_text(PC pc,
                                    const std::string& path,
                                    std::string& text)
{
  text.clear();
  if(path.empty()) return 0;

  PetscViewer viewer = nullptr;
  PetscErrorCode ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, path.c_str(), &viewer);
  if(!ierr) {
    ierr = PCView(pc, viewer);
  }
  if(viewer) {
    const PetscErrorCode destroy_ierr = PetscViewerDestroy(&viewer);
    if(!ierr) ierr = destroy_ierr;
  }
  if(ierr) return 0;

  if(MPI_Barrier(PETSC_COMM_WORLD) != MPI_SUCCESS) return PETSC_ERR_LIB;
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(rank == 0) {
    std::ifstream input(path.c_str());
    std::stringstream buffer;
    buffer << input.rdbuf();
    text = buffer.str();
    std::remove(path.c_str());
  }
  return 0;
}

PetscErrorCode collect_hypre_diagnostics(PC pc,
                                         const std::string& json_output_path,
                                         pc_diagnostic_record& record)
{
  std::string view_text;
  KSPTUNE_PETSC_CALL(capture_pc_view_text(
      pc,
      hypre_view_temporary_path(json_output_path),
      view_text));
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(rank == 0 && !view_text.empty()) {
    const hypre_diagnostics diagnostics = parse_hypre_view_text(view_text);
    record.metrics = diagnostics.metrics;
    record.levels = diagnostics.levels;
  }
  return 0;
}

std::string asm_type_name(PCASMType type)
{
  switch(type) {
  case PC_ASM_BASIC:
    return "basic";
  case PC_ASM_RESTRICT:
    return "restrict";
  case PC_ASM_INTERPOLATE:
    return "interpolate";
  case PC_ASM_NONE:
    return "none";
  default:
    return "unknown";
  }
}

PetscErrorCode record_first_subksp_types(KSP* subksps,
                                         PetscInt local_count,
                                         pc_diagnostic_record& record)
{
  if(local_count <= 0 || !subksps || !subksps[0]) return 0;
  KSPType sub_ksp_type = nullptr;
  KSPTUNE_PETSC_CALL(KSPGetType(subksps[0], &sub_ksp_type));
  if(sub_ksp_type) record.string_metrics["sub_ksp_type"] = sub_ksp_type;
  PC sub_pc = nullptr;
  KSPTUNE_PETSC_CALL(KSPGetPC(subksps[0], &sub_pc));
  if(sub_pc) {
    PCType sub_pc_type = nullptr;
    KSPTUNE_PETSC_CALL(PCGetType(sub_pc, &sub_pc_type));
    if(sub_pc_type) record.string_metrics["sub_pc_type"] = sub_pc_type;
  }
  return 0;
}

PetscErrorCode collect_bjacobi_diagnostics(PC pc, pc_diagnostic_record& record)
{
  PetscInt local_blocks = 0;
  PetscInt first_local_block = 0;
  KSP* subksps = nullptr;
  KSPTUNE_PETSC_CALL(
      PCBJacobiGetSubKSP(pc, &local_blocks, &first_local_block, &subksps));
  PetscInt global_blocks = local_blocks;
  MPI_Allreduce(MPI_IN_PLACE, &global_blocks, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  set_metric(record.metrics, "local_blocks", double(local_blocks));
  set_metric(record.metrics, "global_blocks", double(global_blocks));
  set_metric(record.metrics, "first_local_block", double(first_local_block));
  KSPTUNE_PETSC_CALL(record_first_subksp_types(subksps, local_blocks, record));
  return 0;
}

PetscErrorCode collect_asm_diagnostics(PC pc, pc_diagnostic_record& record)
{
  PCASMType asm_type;
  KSPTUNE_PETSC_CALL(PCASMGetType(pc, &asm_type));
  record.string_metrics["asm_type"] = asm_type_name(asm_type);

  PetscInt local_subdomains = 0;
  PetscInt first_local_subdomain = 0;
  KSP* subksps = nullptr;
  KSPTUNE_PETSC_CALL(
      PCASMGetSubKSP(pc, &local_subdomains, &first_local_subdomain, &subksps));
  PetscInt global_subdomains = local_subdomains;
  MPI_Allreduce(
      MPI_IN_PLACE,
      &global_subdomains,
      1,
      MPIU_INT,
      MPI_SUM,
      PETSC_COMM_WORLD);
  set_metric(record.metrics, "local_subdomains", double(local_subdomains));
  set_metric(record.metrics, "global_subdomains", double(global_subdomains));
  set_metric(record.metrics, "first_local_subdomain", double(first_local_subdomain));
  KSPTUNE_PETSC_CALL(record_first_subksp_types(subksps, local_subdomains, record));
  return 0;
}

PetscErrorCode record_pc_diagnostics_once(KSP ksp,
                                          const std::string& diagnostic_key,
                                          const std::string& json_output_path,
                                          replay_result& result)
{
  if(replay_result_has_pc_diagnostic_key(result, diagnostic_key)) return 0;

  PC pc = nullptr;
  PCType pc_type = nullptr;
  KSPTUNE_PETSC_CALL(KSPGetPC(ksp, &pc));
  KSPTUNE_PETSC_CALL(PCGetType(pc, &pc_type));
  if(!pc_type) return 0;

  pc_diagnostic_record record;
  record.pc_type = pc_type;
  record.setup_key = diagnostic_key;
  record.setup_index = int(result.pc_diagnostics.size()) + 1;
  result.recorded_pc_diagnostic_keys.push_back(diagnostic_key);

  const std::string pc_type_text(pc_type);
  if(pc_type_text == "gamg") {
    KSPTUNE_PETSC_CALL(collect_gamg_diagnostics(pc, record));
  } else if(pc_type_text == "hypre") {
    KSPTUNE_PETSC_CALL(collect_hypre_diagnostics(pc, json_output_path, record));
  } else if(pc_type_text == "bjacobi") {
    KSPTUNE_PETSC_CALL(collect_bjacobi_diagnostics(pc, record));
  } else if(pc_type_text == "asm") {
    KSPTUNE_PETSC_CALL(collect_asm_diagnostics(pc, record));
  }
  result.pc_diagnostics.push_back(record);
  return 0;
}

struct ksp_rtol_reference {
  double norm = 0.0;
  double threshold_norm = 0.0;
  double relative_final_norm = 0.0;
  std::string source = "residual-history";
};

ksp_rtol_reference get_ksp_rtol_reference(KSP ksp,
                                          double fallback_reference_norm,
                                          double final_ksp_residual_norm,
                                          double rtol,
                                          double atol)
{
  ksp_rtol_reference reference;
  reference.norm = fallback_reference_norm;
  reference.threshold_norm = std::max(rtol * fallback_reference_norm, atol);

  // PETSc does not expose the default convergence test's rnorm0/ttol via public KSP API.
#if KSPTUNE_HAS_PETSC_PRIVATE_KSPIMPL
  if(ksp) {
    reference.norm = double(ksp->rnorm0);
    reference.threshold_norm = double(ksp->ttol);
    reference.source = "petsc-rnorm0";
  }
#else
  (void)ksp;
#endif

  if(reference.norm > 0.0) {
    reference.relative_final_norm = final_ksp_residual_norm / reference.norm;
  }
  return reference;
}

PetscErrorCode prepare_ksp_residual_history(KSP ksp,
                                            std::vector<PetscReal>& residual_history)
{
  PetscInt max_iterations = 10000;
  KSPTUNE_PETSC_CALL(KSPGetTolerances(ksp, nullptr, nullptr, nullptr, &max_iterations));
  if(max_iterations <= 0) max_iterations = 10000;
  const PetscInt history_size = std::min<PetscInt>(
      std::max<PetscInt>(max_iterations + 2, 2),
      100000);
  residual_history.assign(static_cast<size_t>(history_size), PetscReal(0.0));
  KSPTUNE_PETSC_CALL(KSPSetResidualHistory(
      ksp,
      residual_history.data(),
      history_size,
      PETSC_TRUE));
  return 0;
}

struct soft_timeout_convergence_context {
  PetscErrorCode (*original_converged)(KSP, PetscInt, PetscReal, KSPConvergedReason*, void*) =
      nullptr;
  void* original_context = nullptr;
  PetscErrorCode (*original_destroy)(void*) = nullptr;
  double soft_timeout_sec = -1.0;
  double solve_elapsed_before_sec = 0.0;
  double solve_start_time_sec = 0.0;
  bool* soft_timeout_triggered = nullptr;
  double* soft_timeout_elapsed_sec = nullptr;
};

PetscErrorCode soft_timeout_converged(KSP ksp,
                                      PetscInt iteration,
                                      PetscReal residual_norm,
                                      KSPConvergedReason* reason,
                                      void* context)
{
  auto* timeout_context = static_cast<soft_timeout_convergence_context*>(context);
  if(timeout_context && timeout_context->original_converged) {
    KSPTUNE_PETSC_CALL(timeout_context->original_converged(
        ksp,
        iteration,
        residual_norm,
        reason,
        timeout_context->original_context));
  }
  if(!timeout_context || timeout_context->soft_timeout_sec < 0.0) return 0;
  if(*reason != KSP_CONVERGED_ITERATING) return 0;

  PetscLogDouble now = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&now));
  const double elapsed = timeout_context->solve_elapsed_before_sec +
      double(now - timeout_context->solve_start_time_sec);
  if(elapsed >= timeout_context->soft_timeout_sec) {
    *reason = KSP_DIVERGED_ITS;
    if(timeout_context->soft_timeout_triggered) {
      *timeout_context->soft_timeout_triggered = true;
    }
    if(timeout_context->soft_timeout_elapsed_sec) {
      *timeout_context->soft_timeout_elapsed_sec = elapsed;
    }
  }
  return 0;
}

PetscErrorCode replay_snapshot(const replay_args& args,
                               const snapshot& snap,
                               const replay_snapshot_plan& plan,
                               replay_cache& cache,
                               PetscLogStage solve_log_stage,
                               bool solve_mpi_logging_enabled,
                               const memory_tracker& memory,
                               replay_result& aggregate,
                               std::vector<double>& solve_times,
                               std::vector<double>& iteration_counts)
{
  petsc_matrix local_matrix;
  petsc_vector right_hand_side, initial_guess, solution;
  ksp_setup_context local_context;
  Mat matrix = nullptr;

  PetscLogDouble load_start = 0.0;
  PetscLogDouble load_end = 0.0;
  PetscLogDouble matrix_prepare_start = 0.0;
  PetscLogDouble matrix_prepare_end = 0.0;
  PetscLogDouble vector_prepare_start = 0.0;
  PetscLogDouble vector_prepare_end = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&load_start));
  KSPTUNE_PETSC_CALL(PetscTime(&matrix_prepare_start));
  const bool cache_ksp_setup =
      args.reuse_ksp_setup && (args.replay_server || plan.ksp_setup_cache_key_count > 1);
  auto& context = cache_ksp_setup ? cache.ksps[plan.ksp_setup_cache_key] : local_context;
  const bool ksp_context_hit = context.ksp.value != nullptr;
  context.matrix_cache_key = plan.matrix_cache_key;
  if(ksp_context_hit) ++aggregate.ksp_setup_cache_hits;
  bool matrix_hit = false;
  if(args.replay_server || plan.matrix_cache_key_count > 1) {
    KSPTUNE_PETSC_CALL(cache.load_matrix(snap, plan.matrix_cache_key, matrix, matrix_hit));
  } else {
    KSPTUNE_PETSC_CALL(load_matrix_binary(snap, local_matrix.ptr()));
    matrix = local_matrix;
  }
  if(matrix_hit) ++aggregate.matrix_cache_hits;
  else ++aggregate.matrix_cache_misses;
  KSPTUNE_PETSC_CALL(PetscTime(&matrix_prepare_end));
  double matrix_prepare_elapsed = double(matrix_prepare_end - matrix_prepare_start);
  MPI_Allreduce(MPI_IN_PLACE, &matrix_prepare_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  aggregate.matrix_prepare_time_sec += matrix_prepare_elapsed;
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "after_matrix", snap.solve_index, snap.snapshot_id));
  KSPTUNE_PETSC_CALL(PetscTime(&vector_prepare_start));
  KSPTUNE_PETSC_CALL(cache.load_vector_copy(
      snap.right_hand_side_file_path,
      snap.rhs_ownership_ranges,
      snap.right_hand_side_size,
      args.replay_server,
      right_hand_side.ptr()));
  const std::string initial_guess_path = args.initial_guess_vector_path.empty()
      ? snap.initial_guess_file_path
      : args.initial_guess_vector_path;
  KSPTUNE_PETSC_CALL(cache.load_vector_copy(
      initial_guess_path,
      snap.x0_ownership_ranges,
      snap.right_hand_side_size,
      args.replay_server,
      initial_guess.ptr()));
  KSPTUNE_PETSC_CALL(VecDuplicate(initial_guess, solution.ptr()));
  if(args.use_initial_guess) {
    KSPTUNE_PETSC_CALL(VecCopy(initial_guess, solution));
  } else {
    KSPTUNE_PETSC_CALL(VecSet(solution, 0.0));
  }
  KSPTUNE_PETSC_CALL(PetscTime(&vector_prepare_end));
  double vector_prepare_elapsed = double(vector_prepare_end - vector_prepare_start);
  MPI_Allreduce(MPI_IN_PLACE, &vector_prepare_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  aggregate.vector_prepare_time_sec += vector_prepare_elapsed;
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "after_vectors", snap.solve_index, snap.snapshot_id));
  KSPTUNE_PETSC_CALL(PetscTime(&load_end));
  double load_elapsed = double(load_end - load_start);
  MPI_Allreduce(MPI_IN_PLACE, &load_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  aggregate.matrix_load_time_sec += load_elapsed;

  matrix_diagnostics diagnostics;
  KSPTUNE_PETSC_CALL(collect_matrix_diagnostics(matrix, args.diagnose_matrix, diagnostics));
  aggregate.matrices.push_back(diagnostics);

  if(aggregate.nullspace_source.empty() || aggregate.nullspace_source == "none") {
    aggregate.nullspace_source = plan.nullspace_choice.source;
  }

  PetscLogDouble nullspace_start = 0.0;
  PetscLogDouble nullspace_end = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&nullspace_start));
  if(plan.nullspace_choice.enabled) {
    aggregate.nullspace_kind = plan.nullspace_choice.kind;
    aggregate.nullspace_actions = plan.actions.label;
    if(plan.nullspace_choice.kind == "field") {
      aggregate.field_nullspace_index = plan.nullspace_choice.field_index;
      aggregate.field_nullspace_block_size = plan.nullspace_choice.block_size;
    }
  }

  if(plan.nullspace_choice.enabled) {
    petsc_nullspace nullspace;
    KSPTUNE_PETSC_CALL(create_replay_nullspace(
        plan.nullspace_choice,
        right_hand_side,
        nullspace.ptr()));
    KSPTUNE_PETSC_CALL(apply_nullspace_actions(
        matrix,
        right_hand_side,
        nullspace,
        plan.actions,
        aggregate,
        !ksp_context_hit));
    KSPTUNE_PETSC_CALL(nullspace.reset());
  }
  KSPTUNE_PETSC_CALL(PetscTime(&nullspace_end));
  double nullspace_elapsed = double(nullspace_end - nullspace_start);
  MPI_Allreduce(MPI_IN_PLACE, &nullspace_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  aggregate.nullspace_time_sec += nullspace_elapsed;

  PetscLogDouble t0 = 0.0;
  PetscLogDouble t1 = 0.0;
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "before_setup", snap.solve_index, snap.snapshot_id));
  if(!ksp_context_hit) {
    if(snap.symmetric) KSPTUNE_PETSC_CALL(MatSetOption(matrix, MAT_SYMMETRIC, PETSC_TRUE));
    if(snap.spd) KSPTUNE_PETSC_CALL(MatSetOption(matrix, MAT_SPD, PETSC_TRUE));
    KSPTUNE_PETSC_CALL(create_configured_ksp(args, snap, matrix, context.ksp.ptr()));
    KSPTUNE_PETSC_CALL(PetscTime(&t0));
    KSPTUNE_PETSC_CALL(KSPSetUp(context.ksp));
    KSPTUNE_PETSC_CALL(PetscTime(&t1));
    double elapsed = double(t1 - t0);
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    context.setup_time_sec = elapsed;
    aggregate.solver_setup_time_sec_actual += elapsed;
    ++aggregate.ksp_setup_cache_misses;
  }
  aggregate.solver_setup_time_sec += context.setup_time_sec;
  aggregate.solver_setup_time_sec_logical += context.setup_time_sec;
  KSP ksp = context.ksp;
  KSPTUNE_PETSC_CALL(record_ksp_types(ksp, aggregate));
  KSPTUNE_PETSC_CALL(record_ksp_residual_norm_type(ksp, aggregate));
  KSPTUNE_PETSC_CALL(record_ksp_tolerances(ksp, aggregate));
  KSPTUNE_PETSC_CALL(record_pc_diagnostics_once(
      ksp, plan.ksp_setup_cache_key, args.json_output_path, aggregate));
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "after_setup", snap.solve_index, snap.snapshot_id));

  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "before_solve", snap.solve_index, snap.snapshot_id));
  for(int iteration = 0; iteration < args.warmup + args.repeat; ++iteration) {
    const bool measured_iteration = iteration >= args.warmup;
    if(args.use_initial_guess) {
      KSPTUNE_PETSC_CALL(VecCopy(initial_guess, solution));
    } else {
      KSPTUNE_PETSC_CALL(VecSet(solution, 0.0));
    }
    KSPTUNE_PETSC_CALL(prepare_ksp_residual_history(ksp, context.residual_history));

    double initial_residual = 0.0;
    double rhs_norm = 0.0;
    double initial_relative = 0.0;
    PetscLogDouble residual_start = 0.0;
    PetscLogDouble residual_end = 0.0;
    KSPTUNE_PETSC_CALL(PetscTime(&residual_start));
    KSPTUNE_PETSC_CALL(compute_residual_norm(
        matrix,
        right_hand_side,
        solution,
        initial_residual,
        rhs_norm,
        initial_relative));
    KSPTUNE_PETSC_CALL(PetscTime(&residual_end));
    double true_residual_elapsed = double(residual_end - residual_start);
    MPI_Allreduce(MPI_IN_PLACE, &true_residual_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    aggregate.true_residual_time_sec += true_residual_elapsed;

    KSPTUNE_PETSC_CALL(PetscTime(&t0));
    soft_timeout_convergence_context soft_timeout_context;
    bool soft_timeout_installed = false;
    if(aggregate.soft_timeout_sec >= 0.0) {
      KSPTUNE_PETSC_CALL(KSPGetAndClearConvergenceTest(
          ksp,
          &soft_timeout_context.original_converged,
          &soft_timeout_context.original_context,
          &soft_timeout_context.original_destroy));
      soft_timeout_context.soft_timeout_sec = aggregate.soft_timeout_sec;
      soft_timeout_context.solve_elapsed_before_sec = aggregate.soft_timeout_elapsed_sec;
      soft_timeout_context.solve_start_time_sec = double(t0);
      soft_timeout_context.soft_timeout_triggered = &aggregate.soft_timeout_triggered;
      soft_timeout_context.soft_timeout_elapsed_sec = &aggregate.soft_timeout_elapsed_sec;
      KSPTUNE_PETSC_CALL(KSPSetConvergenceTest(
          ksp,
          soft_timeout_converged,
          &soft_timeout_context,
          nullptr));
      soft_timeout_installed = true;
    }
    bool solve_stage_pushed = false;
    if(measured_iteration && solve_mpi_logging_enabled) {
      solve_stage_pushed = PetscLogStagePush(solve_log_stage) == 0;
    }
    const PetscErrorCode solve_error = KSPSolve(ksp, right_hand_side, solution);
    if(solve_stage_pushed) PetscLogStagePop();
    PetscErrorCode restore_error = 0;
    if(soft_timeout_installed) {
      restore_error = KSPSetConvergenceTest(
          ksp,
          soft_timeout_context.original_converged,
          soft_timeout_context.original_context,
          soft_timeout_context.original_destroy);
    }
    if(solve_error) return solve_error;
    if(restore_error) return restore_error;
    KSPTUNE_PETSC_CALL(PetscTime(&t1));
    double solve_elapsed = double(t1 - t0);
    MPI_Allreduce(MPI_IN_PLACE, &solve_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    if(aggregate.soft_timeout_sec >= 0.0) {
      const double solve_elapsed_before = soft_timeout_installed
          ? soft_timeout_context.solve_elapsed_before_sec
          : aggregate.soft_timeout_elapsed_sec;
      aggregate.soft_timeout_elapsed_sec = solve_elapsed_before + solve_elapsed;
    }

    KSPConvergedReason reason;
    PetscInt ksp_iterations = 0;
    PetscReal final_ksp_residual_norm = 0.0;
    KSPTUNE_PETSC_CALL(KSPGetConvergedReason(ksp, &reason));
    KSPTUNE_PETSC_CALL(KSPGetIterationNumber(ksp, &ksp_iterations));
    KSPTUNE_PETSC_CALL(KSPGetResidualNorm(ksp, &final_ksp_residual_norm));
    if(reason < 0) {
      if(aggregate.converged) aggregate.reason_code = int(reason);
      aggregate.converged = false;
    } else if(aggregate.converged) {
      aggregate.reason_code = int(reason);
    }

    double final_residual = 0.0;
    double final_rhs_norm = 0.0;
    double final_relative = 0.0;
    KSPTUNE_PETSC_CALL(PetscTime(&residual_start));
    KSPTUNE_PETSC_CALL(compute_residual_norm(
        matrix,
        right_hand_side,
        solution,
        final_residual,
        final_rhs_norm,
        final_relative));
    KSPTUNE_PETSC_CALL(PetscTime(&residual_end));
    true_residual_elapsed = double(residual_end - residual_start);
    MPI_Allreduce(MPI_IN_PLACE, &true_residual_elapsed, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
    aggregate.true_residual_time_sec += true_residual_elapsed;

    if(measured_iteration) {
      step_result step;
      step.repeat_index = iteration - args.warmup;
      step.solve_index = snap.solve_index;
      step.snapshot_id = snap.snapshot_id;
      step.solve_time_sec = solve_elapsed;
      step.initial_true_residual_norm = initial_residual;
      step.rhs_norm = rhs_norm;
      step.initial_true_relative_residual = initial_relative;
      step.final_true_residual_norm = final_residual;
      step.final_true_relative_residual = final_relative;
      const PetscReal* ksp_residual_history_values = nullptr;
      PetscInt ksp_residual_history_count = 0;
      KSPTUNE_PETSC_CALL(KSPGetResidualHistory(
          ksp,
          &ksp_residual_history_values,
          &ksp_residual_history_count));
      if(ksp_residual_history_values && ksp_residual_history_count > 0) {
        step.initial_ksp_residual_norm = double(ksp_residual_history_values[0]);
      }
      step.final_ksp_residual_norm = double(final_ksp_residual_norm);
      const ksp_rtol_reference rtol_reference = get_ksp_rtol_reference(
          ksp,
          step.initial_ksp_residual_norm,
          step.final_ksp_residual_norm,
          aggregate.ksp_rtol,
          aggregate.ksp_atol);
      step.ksp_rtol_reference_norm = rtol_reference.norm;
      step.ksp_convergence_threshold_norm = rtol_reference.threshold_norm;
      step.final_ksp_relative_residual_norm = rtol_reference.relative_final_norm;
      if(aggregate.ksp_rtol_reference_source.empty()) {
        aggregate.ksp_rtol_reference_source = rtol_reference.source;
      }
      step.iterations = int(ksp_iterations);
      step.reason_code = int(reason);
      aggregate.steps.push_back(step);
      solve_times.push_back(solve_elapsed);
      iteration_counts.push_back(double(ksp_iterations));
    }
    if(aggregate.soft_timeout_triggered) break;
  }
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "after_solve", snap.solve_index, snap.snapshot_id));

  KSPTUNE_PETSC_CALL(local_context.ksp.reset());
  KSPTUNE_PETSC_CALL(solution.reset());
  KSPTUNE_PETSC_CALL(initial_guess.reset());
  KSPTUNE_PETSC_CALL(right_hand_side.reset());
  KSPTUNE_PETSC_CALL(local_matrix.reset());
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "after_snapshot_cleanup", snap.solve_index, snap.snapshot_id));
  return 0;
}

void read_solve_mpi_perf_info(PetscLogStage solve_log_stage,
                              bool solve_mpi_logging_enabled,
                              PetscEventPerfInfo& info)
{
  info = PetscEventPerfInfo{};
  if(!solve_mpi_logging_enabled) return;
  if(PetscLogStageGetPerfInfo(solve_log_stage, &info) != 0) {
    info = PetscEventPerfInfo{};
  }
}

double nonnegative_delta(PetscLogDouble after, PetscLogDouble before)
{
  return std::max(0.0, double(after - before));
}

void collect_solve_mpi_diagnostics(PetscLogStage solve_log_stage,
                                   bool solve_mpi_logging_enabled,
                                   const PetscEventPerfInfo& solve_mpi_start_info,
                                   replay_result& result)
{
  if(!solve_mpi_logging_enabled) return;
  PetscEventPerfInfo solve_info;
  read_solve_mpi_perf_info(solve_log_stage, solve_mpi_logging_enabled, solve_info);
  double message_count_sum =
      nonnegative_delta(solve_info.numMessages, solve_mpi_start_info.numMessages);
  double message_bytes_sum =
      nonnegative_delta(solve_info.messageLength, solve_mpi_start_info.messageLength);
  double reduction_count_sum =
      nonnegative_delta(solve_info.numReductions, solve_mpi_start_info.numReductions);
  MPI_Allreduce(MPI_IN_PLACE, &message_count_sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &message_bytes_sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &reduction_count_sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  int communicator_size = 1;
  MPI_Comm_size(PETSC_COMM_WORLD, &communicator_size);
  result.solve_mpi_message_count = 0.5 * message_count_sum;
  result.solve_mpi_message_bytes = 0.5 * message_bytes_sum;
  result.solve_mpi_reduction_count = communicator_size > 0
      ? reduction_count_sum / double(communicator_size)
      : reduction_count_sum;
  result.solve_mpi_message_bytes_mean = result.solve_mpi_message_count > 0.0
      ? result.solve_mpi_message_bytes / result.solve_mpi_message_count
      : 0.0;
}

PetscErrorCode write_json_result(const replay_result& result, const std::string& path)
{
  json output = {
    {"objective_time_sec_median", result.objective_time_sec_median},
    {"total_wall_time_sec", result.total_wall_time_sec},
    {"matrix_load_time_sec", result.matrix_load_time_sec},
    {"matrix_prepare_time_sec", result.matrix_prepare_time_sec},
    {"vector_prepare_time_sec", result.vector_prepare_time_sec},
    {"nullspace_time_sec", result.nullspace_time_sec},
    {"true_residual_time_sec", result.true_residual_time_sec},
    {"use_initial_guess", result.use_initial_guess},
    {"soft_timeout_sec", result.soft_timeout_sec},
    {"soft_timeout_elapsed_sec", result.soft_timeout_elapsed_sec},
    {"soft_timeout_triggered", result.soft_timeout_triggered},
    {"matrix_cache_hits", result.matrix_cache_hits},
    {"matrix_cache_misses", result.matrix_cache_misses},
    {"replay_cache_memory_mb", result.replay_cache_memory_mb},
    {"matrix_cache_memory_mb", result.matrix_cache_memory_mb},
    {"vector_cache_memory_mb", result.vector_cache_memory_mb},
    {"replay_cache_memory_limit_mb", result.replay_cache_memory_limit_mb},
    {"replay_cache_evictions", result.replay_cache_evictions},
    {"solver_setup_time_sec", result.solver_setup_time_sec},
    {"solver_setup_time_sec_actual", result.solver_setup_time_sec_actual},
    {"solver_setup_time_sec_logical", result.solver_setup_time_sec_logical},
    {"ksp_setup_cache_hits", result.ksp_setup_cache_hits},
    {"ksp_setup_cache_misses", result.ksp_setup_cache_misses},
    {"solve_time_sec_total", result.solve_time_sec_total},
    {"solve_time_sec_mean", result.solve_time_sec_mean},
    {"solve_time_sec_median", result.solve_time_sec_median},
    {"solve_time_sec_min", result.solve_time_sec_min},
    {"solve_time_sec_max", result.solve_time_sec_max},
    {"solve_time_sec_stddev", result.solve_time_sec_stddev},
    {"solve_time_sec_sem", result.solve_count > 1 ? json(result.solve_time_sec_sem) : json(nullptr)},
    {"solve_time_sec_relative_sem", result.solve_count > 1 ? json(result.solve_time_sec_relative_sem) : json(nullptr)},
    {"solve_time_sec_range", result.solve_time_sec_range},
    {"solve_mpi_message_count", result.solve_mpi_message_count},
    {"solve_mpi_message_bytes", result.solve_mpi_message_bytes},
    {"solve_mpi_message_bytes_mean", result.solve_mpi_message_bytes_mean},
    {"solve_mpi_reduction_count", result.solve_mpi_reduction_count},
    {"initial_true_residual_norm_mean", result.initial_true_residual_norm_mean},
    {"initial_true_relative_residual_mean", result.initial_true_relative_residual_mean},
    {"final_true_residual_norm_mean", result.final_true_residual_norm_mean},
    {"final_true_relative_residual_mean", result.final_true_relative_residual_mean},
    {"initial_ksp_residual_norm_mean", result.initial_ksp_residual_norm_mean},
    {"final_ksp_residual_norm_mean", result.final_ksp_residual_norm_mean},
    {"ksp_rtol_reference_norm_mean", result.ksp_rtol_reference_norm_mean},
    {"ksp_convergence_threshold_norm_mean", result.ksp_convergence_threshold_norm_mean},
    {"final_ksp_relative_residual_norm_mean", result.final_ksp_relative_residual_norm_mean},
    {"ksp_residual_norm_type", result.ksp_residual_norm_type},
    {"ksp_rtol_reference_source", result.ksp_rtol_reference_source},
    {"ksp_rtol", result.ksp_rtol},
    {"ksp_atol", result.ksp_atol},
    {"ksp_dtol", result.ksp_dtol},
    {"ksp_max_it", result.ksp_max_it},
    {"rss_request_start_mb_sum", result.rss_request_start_mb_sum},
    {"rss_request_end_mb_sum", result.rss_request_end_mb_sum},
    {"rss_request_peak_sample_mb_sum", result.rss_request_peak_sample_mb_sum},
    {"rss_request_delta_mb_sum", result.rss_request_delta_mb_sum},
    {"rss_setup_delta_mb_sum", result.rss_setup_delta_mb_sum},
    {"rss_solve_delta_mb_sum", result.rss_solve_delta_mb_sum},
    {"rss_peak_sample_mb_max_rank", result.rss_peak_sample_mb_max_rank},
    {"rss_peak_sample_mb_max_node", result.rss_peak_sample_mb_max_node},
    {"converged", result.converged},
    {"reason", reason_string(result.reason_code)},
    {"reason_code", result.reason_code},
    {"iterations_median", result.iterations_median},
    {"iterations_total", result.iterations_total},
    {"snapshots", result.snapshots},
    {"repeat", result.repeat},
    {"warmup", result.warmup},
    {"solve_count", result.solve_count},
    {"nullspace", result.nullspace},
    {"nullspace_source", result.nullspace_source},
    {"nullspace_kind", result.nullspace_kind},
    {"nullspace_actions", result.nullspace_actions},
    {"field_nullspace_index", result.field_nullspace_index},
    {"field_nullspace_block_size", result.field_nullspace_block_size},
    {"matrix_nullspace_attached_count", result.matrix_nullspace_attached_count},
    {"transpose_nullspace_attached_count", result.transpose_nullspace_attached_count},
    {"near_nullspace_attached_count", result.near_nullspace_attached_count},
    {"rhs_nullspace_removed_count", result.rhs_nullspace_removed_count},
    {"rhs_nullspace_removed_component_norm_max", result.rhs_nullspace_removed_component_norm_max},
    {"rhs_nullspace_removed_component_relative_norm_max", result.rhs_nullspace_removed_component_relative_norm_max},
    {"ksp_type", result.ksp_type},
    {"pc_type", result.pc_type},
    {"schema_version", 1}
  };
  output["memory_samples"] = json::array();
  for(const memory_sample& sample : result.memory_samples) {
    json entry = {
      {"phase", sample.phase},
      {"solve_index", sample.solve_index},
      {"snapshot_id", sample.snapshot_id},
      {"rss_mb_sum", sample.rss_mb_sum},
      {"rss_mb_max_rank", sample.rss_mb_max_rank},
      {"rss_mb_max_node", sample.rss_mb_max_node}
    };
    output["memory_samples"].push_back(std::move(entry));
  }
  output["matrices"] = json::array();
  for(const matrix_diagnostics& matrix : result.matrices) {
    json entry = {
      {"row_count", matrix.row_count},
      {"column_count", matrix.column_count},
      {"local_row_count", matrix.local_row_count},
      {"local_column_count", matrix.local_column_count},
      {"nonzero_count", matrix.nonzero_count},
      {"allocated_nonzero_count", matrix.allocated_nonzero_count},
      {"petsc_matrix_memory_bytes", matrix.petsc_matrix_memory_bytes},
      {"estimated_matrix_memory_bytes", matrix.estimated_matrix_memory_bytes},
      {"rows", matrix.row_count},
      {"cols", matrix.column_count},
      {"local_rows", matrix.local_row_count},
      {"local_cols", matrix.local_column_count},
      {"nonzeros_used", matrix.nonzero_count},
      {"nonzeros_allocated", matrix.allocated_nonzero_count},
      {"memory_bytes", matrix.estimated_matrix_memory_bytes},
      {"symmetric_tested", matrix.symmetric_tested},
      {"symmetric", matrix.symmetric}
    };
    output["matrices"].push_back(std::move(entry));
  }
  output["steps"] = json::array();
  for(const step_result& step : result.steps) {
    json entry = {
      {"repeat_index", step.repeat_index},
      {"solve_index", step.solve_index},
      {"snapshot_id", step.snapshot_id},
      {"solve_time_sec", step.solve_time_sec},
      {"initial_true_residual_norm", step.initial_true_residual_norm},
      {"rhs_norm", step.rhs_norm},
      {"initial_true_relative_residual", step.initial_true_relative_residual},
      {"final_true_residual_norm", step.final_true_residual_norm},
      {"final_true_relative_residual", step.final_true_relative_residual},
      {"initial_ksp_residual_norm", step.initial_ksp_residual_norm},
      {"final_ksp_residual_norm", step.final_ksp_residual_norm},
      {"ksp_rtol_reference_norm", step.ksp_rtol_reference_norm},
      {"ksp_convergence_threshold_norm", step.ksp_convergence_threshold_norm},
      {"final_ksp_relative_residual_norm", step.final_ksp_relative_residual_norm},
      {"iterations", step.iterations},
      {"reason", reason_string(step.reason_code)},
      {"reason_code", step.reason_code}
    };
    output["steps"].push_back(std::move(entry));
  }
  output["pc_diagnostics"] = json::array();
  for(const auto& record : result.pc_diagnostics) {
    json entry = {{"pc_type", record.pc_type}, {"setup_index", record.setup_index},
                  {"setup_key", record.setup_key}, {"metrics", record.metrics},
                  {"string_metrics", record.string_metrics}, {"levels", json::array()}};
    for(const auto& level : record.levels) {
      entry["levels"].push_back({{"level", level.level}, {"level_from_finest", level.level_from_finest},
                                 {"metrics", level.metrics}, {"string_metrics", level.string_metrics}});
    }
    output["pc_diagnostics"].push_back(std::move(entry));
  }
  return ksptune::write_json_output(output, path);
}


bool option_token(const std::string& token)
{
  return token.size() > 1 &&
      token[0] == '-' &&
      (std::isalpha(static_cast<unsigned char>(token[1])) || token[1] == '_');
}

std::string petsc_options_cache_key(const std::vector<std::string>& petsc_options)
{
  std::ostringstream key;
  for(const std::string& token : petsc_options) {
    key << token.size() << ':' << token << '\n';
  }
  return key.str();
}

PetscErrorCode set_request_petsc_options(const std::vector<std::string>& petsc_options,
                                         std::vector<std::string>& set_keys)
{
  for(size_t index = 0; index < petsc_options.size(); ++index) {
    const std::string& key = petsc_options[index];
    if(!option_token(key)) continue;
    const char* value = "true";
    if(index + 1 < petsc_options.size() && !option_token(petsc_options[index + 1])) {
      value = petsc_options[index + 1].c_str();
      ++index;
    }
    KSPTUNE_PETSC_CALL(PetscOptionsSetValue(nullptr, key.c_str(), value));
    set_keys.push_back(key);
  }
  return 0;
}

void clear_request_petsc_options(const std::vector<std::string>& set_keys)
{
  for(const std::string& key : set_keys) {
    PetscOptionsClearValue(nullptr, key.c_str());
  }
}

struct replay_server_request {
  std::string id;
  std::string command;
  std::string replay_json_out;
  std::vector<std::string> petsc_options;
  int repeat = -1;
  int warmup = -1;
  int solve_target = -2;
  std::string snapshot_id;
  double soft_timeout_sec = -1.0;
};

bool parse_replay_server_request(const std::string& line,
                                 replay_server_request& request,
                                 std::string& error)
{
  const auto invalid = [&](const std::string& message) {
    error = "invalid replay server request: " + message;
    return false;
  };
  const json value = json::parse(line, nullptr, false);
  if(!value.is_object()) return invalid("expected a JSON object");
  if(!value.contains("id") || !value["id"].is_string()) return invalid("id must be a string");
  request.id = value["id"].get<std::string>();
  if(request.id.empty()) return invalid("id must not be empty");
  if(value.contains("command")) {
    if(!value["command"].is_string()) return invalid("command must be a string");
    request.command = value["command"].get<std::string>();
    if(request.command == "shutdown") return true;
    if(!request.command.empty()) return invalid("unknown command");
  }
  if(!value.contains("replay_json_out") || !value["replay_json_out"].is_string()) {
    return invalid("replay_json_out must be a string");
  }
  request.replay_json_out = value["replay_json_out"].get<std::string>();
  if(request.replay_json_out.empty() || request.replay_json_out.find('\0') != std::string::npos) {
    return invalid("replay_json_out must be a nonempty path without NUL");
  }
  if(!value.contains("petsc_options") || !value["petsc_options"].is_array()) {
    return invalid("petsc_options must be an array of strings");
  }
  for(const auto& item : value["petsc_options"]) {
    if(!item.is_string()) return invalid("PETSc options must be strings");
    const auto option = item.get<std::string>();
    if(option.find('\0') != std::string::npos) return invalid("PETSc option contains NUL");
    request.petsc_options.push_back(option);
  }
  const auto read_integer = [&](const char* name, int minimum, int& destination) {
    if(!value.contains(name)) return true;
    const auto& number = value[name];
    if(!number.is_number_integer() || number < minimum || number > std::numeric_limits<int>::max()) {
      return invalid(std::string(name) + " is outside its integer range");
    }
    destination = number.get<int>();
    return true;
  };
  if(!read_integer("repeat", 1, request.repeat) ||
     !read_integer("warmup", 0, request.warmup) ||
     !read_integer("solve_target", -1, request.solve_target)) return false;
  if(value.contains("snapshot_id")) {
    if(!value["snapshot_id"].is_string() || value["snapshot_id"].get<std::string>().empty()) {
      return invalid("snapshot_id must be a nonempty string");
    }
    request.snapshot_id = value["snapshot_id"].get<std::string>();
  }
  if(value.contains("soft_timeout_sec")) {
    const auto& timeout = value["soft_timeout_sec"];
    if(!timeout.is_number()) return invalid("soft_timeout_sec must be numeric");
    request.soft_timeout_sec = timeout.get<double>();
    if(!std::isfinite(request.soft_timeout_sec) || request.soft_timeout_sec < 0.0) {
      return invalid("soft_timeout_sec must be finite and nonnegative");
    }
  }
  return true;
}

bool broadcast_server_line(std::string& line)
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  int length = 0;
  if(rank == 0) {
    if(std::getline(std::cin, line)) {
      length = int(line.size());
    } else {
      length = -1;
    }
  }
  MPI_Bcast(&length, 1, MPI_INT, 0, PETSC_COMM_WORLD);
  if(length < 0) return false;
  if(rank != 0) line.assign(size_t(length), '\0');
  if(length > 0) {
    MPI_Bcast(&line[0], length, MPI_CHAR, 0, PETSC_COMM_WORLD);
  }
  return true;
}

void write_server_response(const std::string& id,
                           int returncode,
                           const std::string& failure_reason)
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if(rank != 0) return;
  json response = {{"id", id}, {"returncode", returncode}};
  if(!failure_reason.empty()) response["failure_reason"] = failure_reason;
  std::cout << response.dump() << std::endl;
}

PetscErrorCode selected_snapshots(const replay_args& args,
                                  const std::vector<snapshot>& all_snapshots,
                                  std::vector<snapshot>& snapshots)
{
  snapshots = all_snapshots;
  return ksptune::select_snapshot(snapshots, args.solve_target, args.snapshot_id);
}

PetscErrorCode run_replay_snapshots(const replay_args& args,
                                    const std::vector<snapshot>& available_snapshots,
                                    replay_cache& cache,
                                    PetscLogStage solve_log_stage,
                                    bool solve_mpi_logging_enabled)
{
  PetscLogDouble replay_start = 0.0;
  PetscLogDouble replay_end = 0.0;
  KSPTUNE_PETSC_CALL(PetscTime(&replay_start));

  std::vector<snapshot> snapshots;
  KSPTUNE_PETSC_CALL(selected_snapshots(args, available_snapshots, snapshots));

  replay_result aggregate;
  aggregate.repeat = args.repeat;
  aggregate.warmup = args.warmup;
  aggregate.snapshots = int(snapshots.size());
  aggregate.use_initial_guess = args.use_initial_guess;
  aggregate.soft_timeout_sec = args.soft_timeout_sec;
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
  PetscEventPerfInfo solve_mpi_start_info;
  read_solve_mpi_perf_info(
      solve_log_stage,
      solve_mpi_logging_enabled,
      solve_mpi_start_info);
  memory_tracker memory;
  KSPTUNE_PETSC_CALL(memory.initialize(PETSC_COMM_WORLD));
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "request_start"));

  std::vector<replay_snapshot_plan> plans(snapshots.size());
  std::map<std::string, int> matrix_cache_key_counts;
  std::map<std::string, int> ksp_setup_cache_key_counts;
  for(size_t index = 0; index < snapshots.size(); ++index) {
    KSPTUNE_PETSC_CALL(build_replay_snapshot_plan(args, snapshots[index], plans[index]));
    ++matrix_cache_key_counts[plans[index].matrix_cache_key];
    ++ksp_setup_cache_key_counts[plans[index].ksp_setup_cache_key];
  }
  for(replay_snapshot_plan& plan : plans) {
    plan.matrix_cache_key_count = matrix_cache_key_counts[plan.matrix_cache_key];
    plan.ksp_setup_cache_key_count = ksp_setup_cache_key_counts[plan.ksp_setup_cache_key];
  }

  std::vector<double> solve_times;
  std::vector<double> iteration_counts;
  for(size_t index = 0; index < snapshots.size(); ++index) {
    KSPTUNE_PETSC_CALL(replay_snapshot(
        args,
        snapshots[index],
        plans[index],
        cache,
        solve_log_stage,
        solve_mpi_logging_enabled,
        memory,
        aggregate,
        solve_times,
        iteration_counts));
    if(aggregate.soft_timeout_triggered) break;
  }

  aggregate.solve_time_sec_total = std::accumulate(solve_times.begin(), solve_times.end(), 0.0);
  aggregate.solve_time_sec_mean = mean(solve_times);
  aggregate.solve_time_sec_median = median(solve_times);
  aggregate.solve_time_sec_min = minimum_value(solve_times);
  aggregate.solve_time_sec_max = maximum_value(solve_times);
  aggregate.solve_time_sec_stddev = population_stddev(solve_times);
  aggregate.solve_time_sec_sem = standard_error_of_mean(solve_times);
  aggregate.solve_time_sec_relative_sem = aggregate.solve_time_sec_mean != 0.0
      ? aggregate.solve_time_sec_sem / aggregate.solve_time_sec_mean
      : 0.0;
  aggregate.solve_time_sec_range = value_range(solve_times);
  aggregate.solve_count = int(solve_times.size());
  aggregate.objective_time_sec_median = aggregate.solve_time_sec_median;
  collect_solve_mpi_diagnostics(
      solve_log_stage,
      solve_mpi_logging_enabled,
      solve_mpi_start_info,
      aggregate);
  aggregate.iterations_median = median(iteration_counts);
  aggregate.iterations_total = std::accumulate(iteration_counts.begin(), iteration_counts.end(), 0.0);

  const auto step_mean = [&](double step_result::*metric) {
    double total = 0.0;
    for(const auto& step : aggregate.steps) total += step.*metric;
    return aggregate.steps.empty() ? 0.0 : total / aggregate.steps.size();
  };
  aggregate.initial_true_residual_norm_mean = step_mean(&step_result::initial_true_residual_norm);
  aggregate.initial_true_relative_residual_mean = step_mean(&step_result::initial_true_relative_residual);
  aggregate.final_true_residual_norm_mean = step_mean(&step_result::final_true_residual_norm);
  aggregate.final_true_relative_residual_mean = step_mean(&step_result::final_true_relative_residual);
  aggregate.initial_ksp_residual_norm_mean = step_mean(&step_result::initial_ksp_residual_norm);
  aggregate.final_ksp_residual_norm_mean = step_mean(&step_result::final_ksp_residual_norm);
  aggregate.ksp_rtol_reference_norm_mean = step_mean(&step_result::ksp_rtol_reference_norm);
  aggregate.ksp_convergence_threshold_norm_mean = step_mean(&step_result::ksp_convergence_threshold_norm);
  aggregate.final_ksp_relative_residual_norm_mean = step_mean(&step_result::final_ksp_relative_residual_norm);

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
  const int replay_cache_evictions_before = cache.evictions;
  KSPTUNE_PETSC_CALL(cache.evict(args.cache_memory_mb));
  if(args.replay_server) KSPTUNE_PETSC_CALL(cache.clear_ksps());
  KSPTUNE_PETSC_CALL(memory.sample(aggregate, "request_end"));
  summarize_request_memory(aggregate);
  KSPTUNE_PETSC_CALL(memory.destroy());
  aggregate.matrix_cache_memory_mb = cache.matrix_memory_mb();
  aggregate.vector_cache_memory_mb = cache.vector_memory_mb();
  aggregate.replay_cache_memory_mb = cache.memory_mb();
  aggregate.replay_cache_memory_limit_mb = args.cache_memory_mb;
  aggregate.replay_cache_evictions = cache.evictions - replay_cache_evictions_before;

  KSPTUNE_PETSC_CALL(write_json_result(aggregate, args.json_output_path));
  return 0;
}

PetscErrorCode run_replay(const replay_args& args)
{
  PetscLogStage solve_log_stage = -1;
  bool solve_mpi_logging_enabled = PetscLogDefaultBegin() == 0;
  if(solve_mpi_logging_enabled) {
    solve_mpi_logging_enabled =
        PetscLogStageRegister("KSPTune KSPSolve", &solve_log_stage) == 0 &&
        solve_log_stage >= 0;
  }

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

  replay_cache cache;
  const PetscErrorCode ierr = run_replay_snapshots(
      args,
      snapshots,
      cache,
      solve_log_stage,
      solve_mpi_logging_enabled);
  const PetscErrorCode destroy_ierr = cache.clear();
  return ierr ? ierr : destroy_ierr;
}

PetscErrorCode run_replay_server(const replay_args& base_args)
{
  if(base_args.snapshot_collection_path.empty()) {
    PetscFPrintf(
        PETSC_COMM_WORLD,
        stderr,
        "Usage: ksptune-petsc-replay -replay_server -snapshot_collection snapshot_collection.csv "
        "[PETSc replay options]\n");
    return PETSC_ERR_ARG_WRONG;
  }

  std::vector<snapshot> snapshots = read_snapshot_collection(base_args.snapshot_collection_path);
  if(snapshots.empty()) {
    PetscFPrintf(
        PETSC_COMM_WORLD,
        stderr,
        "No snapshots found in %s\n",
        base_args.snapshot_collection_path.c_str());
    return PETSC_ERR_FILE_OPEN;
  }

  PetscLogStage solve_log_stage = -1;
  bool solve_mpi_logging_enabled = PetscLogDefaultBegin() == 0;
  if(solve_mpi_logging_enabled) {
    solve_mpi_logging_enabled =
        PetscLogStageRegister("KSPTune KSPSolve", &solve_log_stage) == 0 &&
        solve_log_stage >= 0;
  }

  replay_cache cache;

  std::string line;
  while(broadcast_server_line(line)) {
    replay_server_request request;
    std::string request_error;
    if(!parse_replay_server_request(line, request, request_error)) {
      write_server_response(request.id, int(PETSC_ERR_ARG_WRONG), request_error);
      continue;
    }
    if(request.command == "shutdown") {
      write_server_response(request.id, 0, "");
      break;
    }

    replay_args request_args = base_args;
    request_args.replay_server = true;
    request_args.json_output_path = request.replay_json_out;
    request_args.petsc_options_key = petsc_options_cache_key(request.petsc_options);
    if(request.repeat >= 1) request_args.repeat = request.repeat;
    if(request.warmup >= 0) request_args.warmup = request.warmup;
    if(request.solve_target != -2) {
      request_args.solve_target = request.solve_target;
      request_args.snapshot_id.clear();
    }
    if(!request.snapshot_id.empty()) {
      request_args.snapshot_id = request.snapshot_id;
      if(request.solve_target == -2) request_args.solve_target = -1;
    }
    if(request.soft_timeout_sec >= 0.0) request_args.soft_timeout_sec = request.soft_timeout_sec;

    std::vector<std::string> set_keys;
    PetscErrorCode ierr = set_request_petsc_options(request.petsc_options, set_keys);
    if(!ierr) {
      ierr = run_replay_snapshots(
          request_args,
          snapshots,
          cache,
          solve_log_stage,
          solve_mpi_logging_enabled);
    }
    clear_request_petsc_options(set_keys);

    if(ierr) {
      std::ostringstream reason;
      reason << "replay server request failed with PETSc error " << int(ierr);
      const PetscErrorCode cleanup_ierr = cache.clear_ksps();
      clear_hypre_errors();
      if(cleanup_ierr) {
        reason << "; replay server cleanup failed with PETSc error "
               << int(cleanup_ierr);
        write_server_response(request.id, int(cleanup_ierr), reason.str());
        return cleanup_ierr;
      }
      write_server_response(request.id, int(ierr), reason.str());
    } else {
      write_server_response(request.id, 0, "");
    }
  }

  return cache.clear();
}

}  // namespace

int main(int argc, char** argv)
{
  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if(ierr) return int(ierr);
  const replay_args args = parse_args(argc, argv);
  clear_replay_options();
  if(args.options_help) {
    ierr = run_options_help();
  } else {
    ierr = args.replay_server ? run_replay_server(args) : run_replay(args);
  }
  const PetscErrorCode finalize_ierr = PetscFinalize();
  return int(ierr ? ierr : finalize_ierr);
}
