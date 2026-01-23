cmake_minimum_required(VERSION 3.20)
project(MyPasses LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 使用Config模式，找到LLVMConfig.cmake
# 其中定义LLVM相关的变量、target和构建规则
find_package(LLVM REQUIRED CONFIG)

#把 LLVM 提供的 CMake 模块路径，加到 CMake 查找模块的搜索路径里
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)

add_library(InstructionCountPass MODULE InstructionCountPass.cpp)
add_library(AddToSubPass MODULE AddToSubPass.cpp)

set_target_properties(InstructionCountPass PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
set_target_properties(AddToSubPass PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_compile_definitions(InstructionCountPass PRIVATE ${LLVM_DEFINITIONS})
target_compile_definitions(AddToSubPass PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(InstructionCountPass PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(AddToSubPass PRIVATE ${LLVM_INCLUDE_DIRS})

llvm_update_compile_flags(InstructionCountPass)
llvm_update_compile_flags(AddToSubPass)

llvm_map_components_to_libnames(LLVM_LIBS
  Core
  Support
  Object
  Analysis
  TransformUtils
  Passes
  DebugInfoDWARF
)

target_link_libraries(InstructionCountPass PRIVATE ${LLVM_LIBS})
target_link_libraries(AddToSubPass PRIVATE ${LLVM_LIBS})
