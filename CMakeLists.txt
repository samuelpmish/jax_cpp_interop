cmake_minimum_required(VERSION 3.15...3.26)

project(example LANGUAGES CXX)

# Try to import all Python components potentially needed by nanobind
find_package(Python 
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
add_subdirectory(nanobind)

# We are now ready to compile the actual extension module
nanobind_add_module(

  # Name of the extension
  nanobind_example_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ext.cpp
)

# Install directive for scikit-build-core
install(TARGETS nanobind_example_ext LIBRARY DESTINATION nanobind_example)
