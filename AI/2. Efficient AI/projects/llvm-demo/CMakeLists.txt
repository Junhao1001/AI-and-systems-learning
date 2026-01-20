cmake_minimum_required(VERSION 3.20)
project(HelloPass LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_library(HelloPass SHARED MyFirstPass/HelloPass.cpp)

set_target_properties(HelloPass PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)

message(STATUS "LLVM include dir: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM libraries: ${LLVM_LIBS}")

target_include_directories(HelloPass PRIVATE
  ${LLVM_INCLUDE_DIRS}
)

target_compile_definitions(HelloPass PRIVATE
  ${LLVM_DEFINITIONS}
)

set_target_properties(HelloPass PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# ⭐ 关键：让 LLVM 决定链接哪些库（顺序 + 循环依赖）
llvm_map_components_to_libnames(LLVM_LIBS
  Core
  Support
  Object
  Analysis
  TransformUtils
  Passes
  DebugInfoDWARF
)

target_link_libraries(HelloPass PRIVATE ${LLVM_LIBS})
