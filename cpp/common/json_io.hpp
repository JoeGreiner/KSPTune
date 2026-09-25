#pragma once

#include <petscksp.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>

namespace ksptune {
using json = nlohmann::json;

// All ranks observe output failures, just as they do PETSc solve failures.
inline PetscErrorCode write_json_output(const json& value, const std::string& path)
{
  int rank = 0;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  int error = 0;
  if(rank == 0) {
    try {
      const std::string text = value.dump(2, ' ', false, json::error_handler_t::replace);
      if(path.empty()) {
        std::cout << text << std::endl;
        if(!std::cout) error = PETSC_ERR_FILE_WRITE;
      } else {
        const std::string temporary_path = path + ".tmp";
        std::ofstream output(temporary_path);
        output << text << '\n';
        output.close();
        if(!output || std::rename(temporary_path.c_str(), path.c_str()) != 0) {
          error = PETSC_ERR_FILE_WRITE;
          std::remove(temporary_path.c_str());
        }
      }
    } catch(const std::exception& exception) {
      std::cerr << "Cannot write JSON result: " << exception.what() << std::endl;
      error = PETSC_ERR_FILE_WRITE;
    }
  }
  if(MPI_Bcast(&error, 1, MPI_INT, 0, PETSC_COMM_WORLD) != MPI_SUCCESS) return PETSC_ERR_MPI;
  return static_cast<PetscErrorCode>(error);
}
}  // namespace ksptune
