include_guard(GLOBAL)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
  OUTPUT_VARIABLE Python_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_path(MIND_SIM_HDF5_INCLUDE_DIR hdf5.h
  HINTS "${Python_PREFIX}/include"
  REQUIRED
)
find_library(MIND_SIM_HDF5_LIBRARY hdf5
  HINTS "${Python_PREFIX}/lib"
  REQUIRED
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)
