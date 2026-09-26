# kinema: self-contained skeletal animation runtime. No dependencies beyond
# the C++ standard library; drop the directory into any project and
# add_subdirectory it. The optional Jolt adapter is header-only
# (include/kinema/jolt_adapter.h) and only needs Jolt headers at the
# including translation unit.
#
# KINEMA_USE_BASE=ON builds kinema on equilibrium's base library instead of the
# standard library, without exceptions (include/kinema/config.h maps the
# types). The public API then takes and returns base types.

cmake_minimum_required(VERSION 3.24)
project(kinema LANGUAGES CXX)

option(KINEMA_USE_BASE "Build on equilibrium's base library instead of the C++ standard library" OFF)
# A parent project that already builds equilibrium::base (rx, say) supplies it;
# otherwise KINEMA_EQUILIBRIUM_DIR names the checkout to build it from.
set(KINEMA_EQUILIBRIUM_DIR "" CACHE PATH
  "equilibrium checkout used when KINEMA_USE_BASE=ON and no equilibrium::base target exists")

if(KINEMA_USE_BASE AND NOT TARGET equilibrium::base)
  if(NOT EXISTS "${KINEMA_EQUILIBRIUM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "KINEMA_USE_BASE=ON needs an equilibrium::base target or KINEMA_EQUILIBRIUM_DIR "
      "pointing at an equilibrium checkout")
  endif()
  set(EQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${KINEMA_EQUILIBRIUM_DIR}" equilibrium EXCLUDE_FROM_ALL)
  # Keeps base's std::formatter interop, and with it <format>, out of kinema.
  target_compile_definitions(eq_base PUBLIC BASE_NO_STD_FORMAT)
endif()

add_library(kinema STATIC
  src/clip.cc
  src/blend.cc
  src/graph.cc
  src/state.cc
  src/pose_tools.cc
)
add_library(kinema::kinema ALIAS kinema)
target_include_directories(kinema PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(kinema PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(KINEMA_USE_BASE)
  # base's headers require C++23, and its rules forbid exceptions.
  target_compile_features(kinema PUBLIC cxx_std_23)
  target_compile_definitions(kinema PUBLIC KINEMA_USE_BASE)
  target_link_libraries(kinema PUBLIC equilibrium::base)
  target_compile_options(kinema PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/EHs-c-,-fno-exceptions>)
else()
  target_compile_features(kinema PUBLIC cxx_std_17)
endif()

option(KINEMA_BUILD_TESTS "Build the kinema unit tests" OFF)
if(KINEMA_BUILD_TESTS)
  enable_testing()  # without this add_test is a no-op and ctest finds nothing
  add_executable(kinematest test/kinema_test.cc)
  target_link_libraries(kinematest PRIVATE kinema)
  add_test(NAME kinematest COMMAND kinematest)

  # golden: regenerates the blob byte-layout hashes baked into the test. Run it
  # by hand after a format change and diff its output against the kGolden*
  # constants in test/kinema_test.cc.
  add_executable(kinema_golden tools/golden.cc)
  target_link_libraries(kinema_golden PRIVATE kinema)
  if(KINEMA_USE_BASE)
    target_compile_options(kinematest PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/EHs-c-,-fno-exceptions>)
    target_compile_options(kinema_golden PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/EHs-c-,-fno-exceptions>)
  endif()
endif()
