# PASS

```
Phase 0  LLVM 基础巩固
Phase 1  Analysis Pass 体系（Dominator / Loop / Memory）
Phase 2  Transformation Pass 设计与合法性
Phase 3  Pass Pipeline 与优化顺序
Phase 4  从 LLVM 到 AI Compiler IR（MLIR / Graph）
Phase 5  AI Compiler 核心优化（Fusion / Tiling / Scheduling）
Phase 6  后端 Lowering（LLVM / 硬件）
```

## Some Concepts

- **out-of-tree pass**：LLVM 的插件；在 LLVM 源码内写 Pass 的问题
  - 编译 LLVM 非常慢（几十分钟）
  - CMake 体系复杂
  - 不利于快速调试

## 0. 环境配置

- 之前IR编写时，并不需要所有的LLVM源码编译产物，所以只编译了部分

- 为了后续学习，选择全局编译LLVM源码：
  - Cmake配置更新（在x64 Native Tools Command Prompt for VS 2022中输入）
  

  ```cmd
  cmake -S D:\LLVM\llvm-project\llvm ^
        -B D:\LLVM\llvm-build ^
        -G "Visual Studio 17 2022" ^
        -A x64 ^
        -DLLVM_ENABLE_PROJECTS="clang" ^
        -DLLVM_TARGETS_TO_BUILD="X86" ^
        -DCMAKE_BUILD_TYPE=Release
  ```

  - 开始编译
  
  ```
  cmake --build D:\LLVM\llvm-build --config Release
  ```
  
- 编译后可以将bin目录加入到系统路径中：

  ```
  D:\LLVM\llvm-build\Release\bin
  ```

- 或者在vscode 中设置项目环境`.vscode/settings.json`

  ```json
  {
    "terminal.integrated.env.windows": {
      "PATH": "D:/LLVM/llvm-build/Release/bin;${env:PATH}"
    }
  }
  ```

## 1. 编译MyFirstPASS

参考GPT，写了第一个PASS:

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct HelloPass : public PassInfoMixin<HelloPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    errs() << "Hello from pass: " << F.getName() << "\n";
    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION,
      "HelloPass",
      LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &FPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "hello-pass") {
                FPM.addPass(HelloPass());
                return true;
              }
              return false;
            });
      }};
}
```

在命令行输入如下命令编译：

- linux下运行如下命令

  ```
  clang++ -fPIC -shared ./MyFirstPass/HelloPass.cpp $(llvm-config --cxxflags --ldflags --system-libs --libs core passes) -o HelloPass.so
  ```

- Windows命令行建议执行如下命令：

  ```
  for /f "delims=" %i in ('llvm-config --cxxflags --ldflags --system-libs --libs core passes') do clang++ -shared MyFirstPass\HelloPass.cpp %i -o HelloPass.dll
  ```

### clang++ 和 clang-cl的区别？

上面的windows命令执行失败，报错如下：

```cmd
D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo>for /f "delims=" %i in ('llvm-config --cxxflags --ldflags --system-libs --libs core passes') do clang++ -shared MyFirstPass\HelloPass.cpp %i -o HelloPass.dll

D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo>clang++ -shared MyFirstPass\HelloPass.cpp -ID:\LLVM\llvm-project\llvm\include -ID:\LLVM\llvm-build\include -std:c++17  -D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_NONSTDC_NO_WARNINGS -D_SCL_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS -DUNICODE -D_UNICODE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS /GR- -o HelloPass.dll
clang++: error: unknown argument: '-std:c++17'
clang++: error: no such file or directory: '/GR-'
```

**本质原因**：

- llvm-config提供的MSVC参数
- 因为之前LLVM的编译环境是：

> **Windows + MSVC toolchain**

也就是说：

- LLVM 是用 **MSVC ABI**
- LLVM 的 CMake 检测到的是 **MSVC 风格**

所以当前不应该使用**`clang++`（GNU 风格），而是`clang-cl`（MSVC 风格）**

所以尝试**使用clang-cl来编译Pass插件**：

```
for /f "delims=" %i in ('llvm-config --cxxflags --ldflags --system-libs --libs core passes') do clang-cl /LD MyFirstPass\HelloPass.cpp %i /Fe:HelloPass.dll
```

#### GNU是什么

GNU ≠ Linux

- GNU：工具链 + 库 + 用户态系统
- Linux：内核
   👉 我们常说的 **Linux 系统 = GNU + Linux kernel**

在编译器/系统领域：

> **GNU ≈ GCC 工具链 + GNU ABI 规范**

主要包括：

| 组件         | 说明                             |
| ------------ | -------------------------------- |
| `gcc / g++`  | C / C++ 编译器                   |
| `binutils`   | ld / as / objdump                |
| `glibc`      | C 标准库                         |
| ELF          | 可执行文件格式                   |
| GNU 风格参数 | `-std=c++17`, `-fPIC`, `-shared` |

#### MSVC是什么

**MSVC = Microsoft Visual C++**

它不是一个项目，而是：

> **Windows 官方 C/C++ 工具链 + ABI 规范**

------

MSVC 包含什么？

| 组件               | 说明                        |
| ------------------ | --------------------------- |
| `cl.exe`           | C / C++ 编译器              |
| `link.exe`         | 链接器                      |
| `ucrt / vcruntime` | C/C++ 运行时                |
| COFF / PE          | `.exe / .dll` 格式          |
| MSVC 参数风格      | `/std:c++17`, `/LD`, `/GR-` |

#### clang的角色

clang **只是前端**， 皆可以是GNU模式，也可以是MSVC模式

`clang-cl` ≈ 用 clang 前端，模拟 cl.exe 行为

简单来说：

**GNU 是“Unix 世界的规则”**， **MSVC 是“Windows 世界的规则”**， **clang 可以同时说两种“语言”**

同时为了排除工具链干扰，不再使用`llvm-config`，而是直接使用显示命令：

```cmd
clang-cl /LD MyFirstPass\HelloPass.cpp ^
  -ID:\LLVM\llvm-project\llvm\include ^
  -ID:\LLVM\llvm-build\include ^
  /std:c++17 /GR- ^
  /Fe:HelloPass.dll
```

### 寻找Plugin.h

#### 什么是PassPlugin(后续补充)

#### 当前报错

上述命令后，报错如下：

```cmd
MyFirstPass\HelloPass.cpp(4,10): fatal error: 'llvm/Passes/PassPlugin.h' file not found 4 | #include "llvm/Passes/PassPlugin.h" | ^~~~~~~~~~~~~~~~~~~~~~~~~~ 1 error generated.
```

本来使用的是GPT一开始直接给我的`HelloPass.cpp`

其中PassPlugin.h的位置是错误，需要在源码目录`llvm-project`下使用find 命令下找其正确位置：

```cmd
find llvm/include -name "PassPlugin.h"
```

最后发现其正确位置：

```
llvm/Passes/PassPlugin.h ❌ GPT默认给我的位置
llvm/Plugin/PassPlugin.h ✅ 正确位置
```

修改后当前错误消失

### lib链接问题

继续报错如下：

```cmd
D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo>clang-cl /LD MyFirstPass\HelloPass.cpp ^ More? -ID:\LLVM\llvm-project\llvm\include ^ More? -ID:\LLVM\llvm-build\include ^ More? /std:c++17 /GR- ^ More? /Fe:HelloPass.dll HelloPass-fcb2eb.obj : error LNK2019: 无法解析的外部符号 "public: unsigned __int64 __cdecl llvm::SmallVectorBase<unsigned int>::size(void)const " (?size@?$SmallVectorBase@I@llvm@@QEBA_KXZ)，函数 "public: void __cdecl llvm::SmallVectorTemplateBase<class std::function<bool __cdecl(class llvm::StringRef,class llvm::PassManager<class llvm::Function,class llvm::AnalysisManager<class llvm::Function> > &,class llvm::ArrayRef<struct llvm::PassBuilder::PipelineElement>)>,0>::push_back(class std::function<bool __cdecl(class llvm::StringRef,class llvm::PassManager<class llvm::Function,class llvm::AnalysisManager<class llvm::Function> > &,class llvm::ArrayRef<struct llvm::PassBuilder::PipelineElement>)> const &)" (?push_back@?$SmallVectorTemplateBase@V?$function@$$A6A_NVStringRef@llvm@@AEAV?$PassManager@VFunction@llvm@@V?$AnalysisManager@VFunction@llvm@@$$V@2@$$V@2@V?$ArrayRef@UPipelineElement@PassBuilder@llvm@@@2@@Z@std@@$0A@@llvm@@QEAAXAEBV?$function@$$A6A_NVStringRef@llvm@@AEAV?$PassManager@VFunction@llvm@@V?$AnalysisManager@VFunction@llvm@@$$V@2@$$V@2@V?$ArrayRef@UPipelineElement@PassBuilder@llvm@@@2@@Z@std@@@Z) 中引用了该符号
```

**问题原因为**：

- **只编译了 `.cpp`，但没有把 LLVM 的库链接进来**

- 在 Windows + MSVC ABI（`clang-cl`）下，**Pass Plugin 必须显式链接 LLVM lib**，否则一定出现你这个 `LNK2019`

#### 显示链接

一开始GPT给我的方案是显示链接相关的lib库；

找到LLVM lib库:

```
D:\LLVM\llvm-build\Release\lib
```

然后使用`clang-cl`正确链接：

```
clang-cl /LD MyFirstPass\HelloPass.cpp ^
  -ID:\LLVM\llvm-project\llvm\include ^
  -ID:\LLVM\llvm-build\include ^
  /std:c++17 /GR- ^
  /md ^
  /link ^
  /LIBPATH:D:\LLVM\llvm-build\Release\lib ^
  LLVMCore.lib ^
  LLVMSupport.lib ^
  LLVMAnalysis.lib ^
  LLVMTransformUtils.lib ^
  LLVMPasses.lib ^
  /OUT:HelloPass.dll
```

**后续添加了各种lib库，仍然存在此处的`LINK2019`问题**

原因在于：

- LLVM 的静态库之间存在「循环依赖 + 条件编译 + ODR 边界」
- MSVC 链接器无法靠“顺序”一次性解析完

因此最终更换解决方案：

**用 CMake + LLVMConfig.cmake 编译 PassPlugin**

#### CMake + LLVMConfig.cmake 编译 PassPlugin

1. **确认 LLVMConfig.cmake 位置**：

   - 在build目录里找到：

     ```
     D:\LLVM\llvm-build\lib\cmake\llvm\LLVMConfig.cmake
     ```

   - 不在上述目录，可以用`find`目录找

2. 在 `llvm-demo` 目录写一个最小 CMakeLists.txt

   ```cmake
   cmake_minimum_required(VERSION 3.20)
   project(HelloPass LANGUAGES CXX)
   
   find_package(LLVM REQUIRED CONFIG)
   
   message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
   message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
   
   add_library(HelloPass SHARED MyFirstPass/HelloPass.cpp)
   
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
   
   ```

3. **构建 PassPlugin**

   - 注意这里的`LLVM_DIR`目录要和前面找到的LLVMConfig.cmake位置一致

   ```
   cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DLLVM_DIR=D:\LLVM\llvm-build\lib\cmake\llvm
   
   cmake --build build --config Release
   ```
   
4. **终于成功生成HelloPass.dll!!!!!!!!**

   ```
   build\Release\HelloPass.dll
   ```

## 2. 运行MyFirstPass

### 符号导出失败问题

命令：

```
opt -load-pass-plugin .\build\Release\HelloPass.dll -passes=hello input.ll -disable-output
```

报错如下：

```cmd
D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo>opt -load-pass-plugin .\build\Release\HelloPass.dll -passes=hello input.ll -disable-output 

LLVM ERROR: Plugin entry point not found in '.\build\Release\HelloPass.dll'. Is this a legacy plugin? 

PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace and instructions to reproduce the bug. 

Stack dump: 0. Program arguments: opt -load-pass-plugin .\\build\\Release\\HelloPass.dll -passes=hello input.ll -disable-output
```

在命令行中使用如下命令查找符号：

```cmd
dumpbin /EXPORTS HelloPass.dll | findstr llvmGetPassPluginInfo
```

**发现输出为空，说明符号没有正确导出**

**原因**：

- **在 Windows 上：`LLVM_ATTRIBUTE_WEAK` 并不会自动导出符号**；
- 你必须显式使用 `__declspec(dllexport)`（或 LLVM 提供的宏）

| 项                    | Linux | Windows            |
| --------------------- | ----- | ------------------ |
| `LLVM_ATTRIBUTE_WEAK` | 有用  | ❌ 没用             |
| 默认导出              | 可    | ❌ 不可             |
| MSVC                  | —     | **必须 dllexport** |

改动如下：将`LLVM_ATTRIBUTE_WEAK`更换为`LLVM_PLUGIN_EXPORT`

```cpp
extern "C" LLVM_PLUGIN_EXPORT PassPluginLibraryInfo
llvmGetPassPluginInfo() {
	......
}
```

#### LLVM_PLUGIN_EXPORT 编译问题

更换后，在编译dll文件时出现了编译报错

```cmd
D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo>cmake --build build --config Release 适用于 .NET Framework MSBuild 版本 17.14.40+3e7442088 HelloPass.cpp D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\MyFirstPass\HelloPass.cpp(20,12): error C4430: 缺少类型说明符 - 假定为 int。注意: C++ 不支持默认 int [D :\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\build\HelloPass.vcxproj] D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\MyFirstPass\HelloPass.cpp(20,31): error C2146: 语法错误: 缺少“;”(在标识符“PassPluginLibraryInfo ”的前面) [D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\build\HelloPass.vcxproj]
```

更换回 `LLVM_ATTRIBUTE_WEAK`,就能成功编译：

GPT给的原因在于：

- `PassPluginLibraryInfo` 是 **模板类型的实例**（含 `PassBuilder` 回调 lambda）
- MSVC 对模板 + `__declspec(dllexport)` 导出有限制，尤其是返回值是 **非 POD 的复杂类型**
- 所以直接在函数上加 `LLVM_PLUGIN_EXPORT` 会导致 MSVC 报错（缺少类型说明符 / 语法错误）

因此现在有了矛盾的地方：

- 如果用LLVM_ATTRIBUTE_WEAK，能编译dll文件，但是符号没有正确导出，所以不能加载
- 如果用LLVM_PLUGIN_EXPORT, 无法正常编译dll文件

| 平台    | 编译器    | DLL 能否用 `LLVM_ATTRIBUTE_WEAK` 导出？ | DLL 能否被 `opt -load-pass-plugin` 加载？ |
| ------- | --------- | --------------------------------------- | ----------------------------------------- |
| Linux   | Clang/GCC | ✅                                       | ✅                                         |
| Windows | Clang     | ✅                                       | ✅                                         |
| Windows | MSVC      | ✅ (能生成 DLL)                          | ❌（找不到 llvmGetPassPluginInfo）         |

#### Clang-cl 编译 dll

- 之前使用这个方案，当时的问题是：**使用显示命令编译，会存在lib库无法正常链接的问题，所以最终选择了使用cmake+cmakelists.txt的方案**

```
clang++ -fPIC -shared MyFirstPass/HelloPass.cpp \
  -ID:/LLVM/llvm-project/llvm/include \
  -ID:/LLVM/llvm-build/include \
  -std=c++17 \
  -DLLVM_PLUGIN_API_VERSION=1 \
  -o HelloPass.dll
```

- 因此当前的新思路是：
  - 仍然使用CMakelists.txt，来解决链接问题
  - 尝试配置CMake，让其不使用默认的MSVC的cl.exe编译器，而是使用LLVM编译出的clang-cl.exe编译器

**为什么之前使用的是`cl.exe`，而不是PATH里的`clang-cl.exe`**?

之前用的命令是：

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ...
```

这意味着：

- CMake **不是自己编译**
- 而是：
  - 生成 `.vcxproj`
  - 交给 **MSBuild**
  - MSBuild 再调用 **cl.exe**

- 路径是：

```
cmake
 └─> MSBuild
      └─> cl.exe
```

- **clang-cl 完全不在这个链路里**

**如何调用clang-cl**？

- 情况 A：使用 Ninja / Makefile Generator

```
cmake -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
```

- 情况 B：命令行直接调用

```
clang-cl /LD HelloPass.cpp ...
```

- **只要你用了 Visual Studio Generator，就 100% 不可能用 clang-cl**

方案B之前会有链接问题，尝试使用方案A

- 安装ninjia: 

  - 官方发布页： https://ninja-build.org/

  - 下载：ninja-win.zip

  - 解压缩后，将对应目录配置到PATH中，例如：

    ```
    D:\tools\ninja\
    ```

  - 检测是否安装成功

    ```
    ninja --version
    ```

- **重新配置CMake (需要注意需要将原来的build目录先删除)**

```
rmdir /s /q build

cmake -S . -B build ^
  -G ninja ^
  -DCMAKE_C_COMPILER=clang-cl ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_DIR=D:/LLVM/llvm-build/lib/cmake/llvm
```

- 编译：

  ```
  ninja -C build
  ```

- 验证导出符号（需要在visual studio 2022命令行中）：

  ```
  dumpbin /EXPORTS build\MyFirstPass\HelloPass.dll | findstr llvmGetPassPluginInfo
  ```

#### 强制导出

- 然而还是编译失败了，依然是无法识别`LLVM_PLUGIN_EXPORT`

  ```cmd
  D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo>cmake --build build                                                                   
  [1/2] Building CXX object CMakeFiles\HelloPass.dir\MyFirstPass\HelloPass.cpp.obj
  FAILED: [code=1] CMakeFiles/HelloPass.dir/MyFirstPass/HelloPass.cpp.obj
  D:\LLVM\llvm-build\Release\bin\clang-cl.exe  /nologo -TP -DHelloPass_EXPORTS -D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_NONSTDC_NO_WARNINGS -D_SCL_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS -DUNICODE -D_UNICODE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -ID:\LLVM\llvm-project\llvm\include -ID:\LLVM\llvm-build\include /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -std:c++17 -MD /showIncludes /FoCMakeFiles\HelloPass.dir\MyFirstPass\HelloPass.cpp.obj /FdCMakeFiles\HelloPass.dir\ -c -- "D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\MyFirstPass\HelloPass.cpp"
  D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\MyFirstPass\HelloPass.cpp(20,12): error: unknown type name 'LLVM_PLUGIN_EXPORT'       
     20 | extern "C" LLVM_PLUGIN_EXPORT PassPluginLibraryInfo
        |            ^
  D:\Westlake University\AI-and-systems-learning\AI\2. Efficient AI\projects\llvm-demo\MyFirstPass\HelloPass.cpp(20,52): error: expected ';' after top level declarator      
     20 | extern "C" LLVM_PLUGIN_EXPORT PassPluginLibraryInfo
        |                                                    ^
        |                                                    ;
  2 errors generated.
  ninja: build stopped: subcommand failed.
  ```

原因：

- Windows 下，LLVM 官方并不推荐第三方 PassPlugin 使用 `LLVM_PLUGIN_EXPORT`
- 但是在Windows中，使用`LLVM_ATTRIBUTE_WEAK`又不会自动导出符号

解决方案：

- cpp中还是使用`LLVM_ATTRIBUTE_WEAK`

- 在CMakeLists.txt中，强制**把 DLL 里所有 extern "C" 的符号都导出**

- 在你的 `add_library(HelloPass SHARED ...)` 后面，**加这一句**：

  ```cmake
  set_target_properties(HelloPass PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
  ```

**最终结果：终于成功！！！！！！！！**

成功编译，并且能运行opt

- 运行命令：

  ```cmd
  opt -load-pass-plugin build\HelloPass.dll -passes=hello-pass .\MyFirstIR\add.ll  -disable-output
  ```

- 这里的`-passes=hello-pass `一定要代码里对应

  ```cpp
  if (Name == "hello-pass") {
                  FPM.addPass(HelloPass());
                  return true;
                }
  ```

- 同时这里的`.ll`文件时需要真实存在的IR文件

### 最终运行方案：

- HelloPass.cpp

  ```cpp
  #include "llvm/IR/Function.h"
  #include "llvm/IR/PassManager.h"
  #include "llvm/Passes/PassBuilder.h"
  #include "llvm/Plugins/PassPlugin.h"
  #include "llvm/Support/raw_ostream.h"
  
  using namespace llvm;
  
  namespace {
  
  struct HelloPass : public PassInfoMixin<HelloPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
      errs() << "Hello from pass: " << F.getName() << "\n";
      return PreservedAnalyses::all();
    }
  };
  
  } // namespace
  
  extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
  llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION,
        "HelloPass",
        LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
          PB.registerPipelineParsingCallback(
              [](StringRef Name, FunctionPassManager &FPM,
                  ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == "hello-pass") {
                  FPM.addPass(HelloPass());
                  return true;
                }
                return false;
              });
        }};
  }
  ```

- CMakeLists.txt

  ```cmake
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
  ```

- CMake配置：

  - 删除原来build设置

  ```
  rmdir /s /q build
  ```

  - 可以使用`Visual Studio Generator`编译：

  ```
  cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DLLVM_DIR=D:\LLVM\llvm-build\lib\cmake\llvm
  ```

  - 也可以使用`Ninja`编译

  ```cmd
  cmake -S . -B build ^
    -G Ninja ^
    -DCMAKE_C_COMPILER=clang-cl ^
    -DCMAKE_CXX_COMPILER=clang-cl ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DLLVM_DIR=D:/LLVM/llvm-build/lib/cmake/llvm
  ```

  - 最终编译命令


  ```
  cmake --build build --config Release
  ```

  

- opt 运行 

  - 文件位置根据实际情况改动

  ```cmd
  opt -load-pass-plugin build\Release\HelloPass.dll -passes=hello-pass .\MyFirstIR\add.ll  -disable-output
  ```

  

## 3. 简单的Pass (分析/修改IR)

创建一个示例IR: test.ll

```
; ModuleID = 'test'
source_filename = "test.c"

define i32 @main() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 10, i32* %a, align 4
  store i32 20, i32* %b, align 4
  %x = load i32, i32* %a, align 4
  %y = load i32, i32* %b, align 4
  %sum = add i32 %x, %y
  ret i32 %sum
}
```

### InstructionCountPass (统计指令数)

代码核心如下所示：

```cpp
struct InstructionCountPass : public PassInfoMixin<InstructionCountPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
        int count = 0;
        for (auto &BB : F)
            for (auto &I : BB)
                count++;
        errs() << "Function " << F.getName() << " has " << count << " instructions\n";
        return PreservedAnalyses::all();
    }
};
```

- `struct InstructionCountPass : public PassInfoMixin<InstructionCountPass>`:
  - LLVM中Pass 通常用 struct 定义
  - 在新Pass Manager中，**只要你提供了 `run()` 函数，并继承 `PassInfoMixin`，LLVM 就认为你是一个 Pass**
- `PassInfoMixin<T>`:
  - 使用了**CRTP(Curiously Recurring Template Pattern)**：奇异递归模板模式
    - **基类拿到派生类的类型**
    - 在**编译期**完成绑定
    - 不需要虚函数
  - LLVM用它来：
    - 给你的 Pass 生成唯一类型信息
    - 进行 Pass 管理、调度、缓存分析结果
  - 简单来说：**用于告诉 LLVM：这个 struct 是一个 Pass**

- `PreservedAnalyses run(Function &F, FunctionAnalysisManager &)`

  - Pass的入口函数, 统一约定用`run`作为入口：
  - 不同粒度的Pass, `run`的第一参数不同：

  | Pass 类型     | run 参数      |
  | ------------- | ------------- |
  | Function Pass | `Function &F` |
  | Module Pass   | `Module &M`   |
  | Loop Pass     | `Loop &L`     |

  - `Function &F`即指传入的被处理的函数
  - `FunctionAnalysisManager &`:
    - 管理 **分析结果缓存**
    - 这里暂时没用到
  - `PreservedAnalyses`: 用于说明**Pass 执行完后，哪些分析结果仍然是有效的**

- `for (auto &BB : F)` & `for (auto &I : BB)`

  - 常见的遍历模板
  - 先遍历BasicBlock，再在每个BB中遍历指令Instruction

- 最后打印输出

- `PreservedAnalyses::all()`:
  - `all()`: 说明我这个 Pass **没有修改 IR**， 所有已有的分析结果都仍然有效
  - `none()`: 说明修改了IR

### AddToSubPass (把add替换为sub)

- `AddToSubPass.cpp` (函数实现部分) 最初实现

```cpp
struct AddToSubPass : public PassInfoMixin<AddToSubPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *op = dyn_cast<BinaryOperator>(&I)) {
                    if (op->getOpcode() == Instruction::Add) {
                        IRBuilder<> builder(op);
                        auto *newSub = builder.CreateSub(op->getOperand(0), op->getOperand(1));
                        op->replaceAllUsesWith(newSub);
                        op->eraseFromParent();
                    }
                }
            }
        }
        return PreservedAnalyses::none();
    }
};
```

上述代码有一个严重问题：

- `eraseFromParent()` 会：
  - **把当前指令从 BasicBlock 链表中移除**
  - **使当前迭代器失效**
- 因此下一步循环访问的是：**已经被释放 / 无效的内存**
- 这会导致崩溃

因此改动如下：

```cpp
struct AddToSubPass : public PassInfoMixin<AddToSubPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
        SmallVector<BinaryOperator*, 8> Adds;

    // 第一步：只遍历，不修改
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *op = dyn_cast<BinaryOperator>(&I)) {
                    if (op->getOpcode() == Instruction::Add) {
                        Adds.push_back(op);
                    }
                }
            }
        }

    // 第二步：安全地修改
        for (auto *op : Adds) {
            IRBuilder<> builder(op);
            auto *newSub = builder.CreateSub(
                op->getOperand(0), op->getOperand(1)
            );
            op->replaceAllUsesWith(newSub);
            op->eraseFromParent();
        }
        
        return PreservedAnalyses::none();
    }
};
```

- `op->eraseFromParent()`在此处
  - `op` 从 `BasicBlock` 的链表中移除
  - 内存被释放
  - **但是没有修改 `Adds` 本身**
  - 所以`Adds`迭代器仍然有效，不会崩溃
- 上述实现的Pass能将所有`Add`语句替换为`Sub`



## 4. Analysis Pass

**Analysisi Pass不修改IR，只回答问题**

需要了解的Analysis Pass (和AI compiler相关)

| Analysis                   | 作用           | AI Compiler 中的角色         |
| -------------------------- | -------------- | ---------------------------- |
| **DominatorTree**          | 控制流支配关系 | 判断合法 hoist / fusion      |
| **LoopInfo**               | 循环结构       | Tiling / Unrolling / Mapping |
| **ScalarEvolution (SCEV)** | 循环迭代次数   | Shape / Range 推导           |
| **PostDominatorTree**      | 控制流后支配   | 控制依赖消除                 |

### 4.1 Dominator Tree

#### Some Concepts

- **CFG**: 控制流

  | 视角          | 说明                                        |
  | ------------- | ------------------------------------------- |
  | IR 文件       | **隐式存在**（由 br / switch / ret 等决定） |
  | LLVM 内存     | **显式图结构（analysis 视图）**             |
  | DominatorTree | **基于 CFG 计算出来的树**                   |

- **Dominance定义**:  

  - 对 CFG 中两个节点 A 和 B：

  - **A dominates B** ⇔ **所有从入口（entry）到 B 的路径，都必须经过 A**

  - 记作：

  ```
  A dom B
  ```

- **Immediate Dominator (idom)**：
  - B 的 **Immediate Dominator** = 严格支配 B，且离 B 最近的那个节点

- **Dominator Tree**:
  - 把每个节点连到它的 idom
  - 得到一棵树（entry 是根）
  - DT 是 CFG 的“抽象结构”, 而非CFG本身

#### DT的输入和输出

**输入**：一个 Function 的 CFG

- BasicBlock 列表

- Terminator 指令（br / switch / ret）

  - br: 跳转命令

    - 无条件跳转

    ```
    br label %next
    ```

    - 条件跳转

    ```
    br i1 %cond, label %then, label %else //%cond为1，则跳转then，否则跳转else
    ```

  - ret: 函数出口，CFG重点

    ```
    ret void
    ret i32 %x
    ```

  - switch: 多分支控制流：

    ```
    switch i32 %x, label %default [
      i32 0, label %case0
      i32 1, label %case1
    ]
    ```

- CFG 边关系

**输出**：可以是如下输出结构

- **dominate(A, B)** 查询接口
- **Immediate Dominator（idom）**
- **Dominator Tree 本身（树结构）**
- **Dominator 前序 / 后序遍历**

#### LLVM中常用的DT的API

- 获取Dominator Tree:

```cpp
auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
```

- 判断支配关系：

```cpp
DT.dominates(A, B);
```

- 获取IDom:

```cpp
BasicBlock *IDom = DT.getNode(BB)->getIDom()->getBlock();
```

- 获取Dominator Tree节点

```cpp
DomTreeNode *N = DT.getNode(BB);
```

- 遍历Dominator Tree:

```cpp
for (auto *Child : *DT.getNode(BB)) {
    // Child 是 DomTreeNode*
}
```

#### AI Compiler中的应用

AI Compiler中也有CFG，但是形式略有变化：

| LLVM       | AI Compiler    |
| ---------- | -------------- |
| BasicBlock | Region / Block |
| Branch     | Control Op     |
| CFG        | Region CFG     |

AI Compiler中对DT的应用例如：

- **Operator Fusion 合法性**：能不能把算子 A 和 B fuse？
  - 必须保证A 在所有路径上都先于 B 执行
  - 中间没有条件执行破坏
- **Hoisting / Sinking（调度）**：
  - 把算子提前到公共路径
  - 或下沉到特定分支
- **死代码消除（DCE）**
  - 没有被支配的使用
  - 不可达
- **Shape / Buffer 安全性**
  - buffer 初始化是否一定发生？
  - 访问是否可能在未定义路径？

### 4.2 LoopInfo

目标：**识别函数中的循环结构（Loop / Loop Nest），并提供循环信息给优化 Pass**

#### Some Concepts

在LLLVM中，**Loop对象**往往包含：

- **Loop Header**: Loop 的入口 BasicBlock
  - 支配循环内所有 BasicBlock
  - 有回边跳回 Header
- **Latch**: 包含回边的BB
  - Backedge: **从循环内某个 BB 跳回 Header** → 回边
  - LLVM 利用 **backedge = predecessors(header) ∩ loop** 来识别循环
- **Preheader**：循环前的单前驱 BB
- **Body / Blocks**：所有循环内的 BasicBlock
- **SubLoops**：嵌套循环

#### LoopInfo

- **LoopInfo** = `Function` 对象下的 `Loop` 集合
- 结构是 **树状（Loop Nest Tree）**

```
LoopInfo
 ├─ Loop1
 │    ├─ header = BB1
 │    ├─ blocks = {BB1, BB2, BB3}
 │    └─ subloops = {Loop1.1}
 └─ Loop2
      ├─ header = BB10
      └─ blocks = {BB10, BB11}
```

- 每个 `Loop` 对象知道：
  - header
  - 所有块
  - 子循环
  - 父循环

#### LLVM API使用方法

- 获取LoopInfo

  ```cpp
  FunctionAnalysisManager FAM;
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  ```

- 常用API

| 方法                          | 描述                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `LI.begin() / LI.end()`       | 遍历函数顶层循环                                             |
| `Loop *L = LI.getLoopFor(BB)` | 获取某个 BasicBlock 所属循环                                 |
| `L->getHeader()`              | 获取循环头                                                   |
| `L->getLoopLatch()`           | 获取循环回边块                                               |
| `L->getBlocks()`              | 获取循环包含的所有 BasicBlock                                |
| `L->getSubLoops()`            | 获取嵌套循环                                                 |
| `L->isLoopInvariant(Value*)`  | 检查一个操作数是否循环不变 （判断指令是否不依赖循环内部变量） |

#### AI Compiler 中LoopInfo的作用 （现在有点理解不了）

| LLVM Pass 场景              | AI Compiler 场景                      |
| --------------------------- | ------------------------------------- |
| LICM (hoist loop invariant) | 将算子移到循环外，减少重复计算        |
| Loop Unroll                 | 将 Tensor 运算展开到多层循环          |
| Loop Tiling                 | 划分 Tile / Block，映射到 GPU 或 SIMD |
| Fusion / Scheduling         | 判断 loop nest 内算子顺序和合法性     |

- **Loop-Invariant Code Motion (LICM)**
  - **把循环里“每一轮都算一样结果”的代码，搬到循环外面算一次**
  - 减少重复计算
  - 需要 **LoopInfo + DominatorTree**
  - 条件：**指令对循环内变量无依赖**

-  **Loop Unroll（循环展开）**

  - **把循环体复制多次，减少分支 / 提高指令级并行**
  - 用 **代码体积换性能**
  - AI Compiler 中常用于 **小循环 / 内层循环**

- **Loop Tiling（Blocking）**：

  - 把大循环切成小块，提升 cache / memory locality

    ```cpp
    for i in N:
      for j in M:
        C[i][j] += A[i][k] * B[k][j]
    ```

    - 每次迭代都要从内存加载`A[i][k]`和`B[k][j]`
    - 如果A/B很大，cache可能放不下；每次都要访问DRAM

  - Tiling后：

    ```cpp
    for ii in tiles of i:
      for jj in tiles of j:
        for i in ii:
          for j in jj:
            ...
    ```

  - `A[ii:ii+Ti]` 和 `B[jj:jj+Tj]`

    - 可以 **完整放进 cache**

  - 每次 tile 内：

    - 数据被反复复用
    - DRAM 访问次数显著减少

  - 解决 **memory bandwidth** 问题

- **Loop Vectorization（向量化）**

  - **把标量循环，变成 SIMD 指令一次算多个元素**
  - 利用**硬件向量单元（AVX / NEON / GPU）**

### 4.3 ScalarEvolution（SCEV）（暂时不用深入）

**SCEV 是 LLVM 用来“理解循环中标量值如何随循环迭代变化”的符号分析系统**

- 它构建的是 **符号表达式（Symbolic Expression）**

- ```cpp
  i = 0;
  for (...) {
    ...
    i = i + 1;
  }
  ```

- SCEV 会把 `i` 理解为：i = 初始值 + 迭代次数 × 步长

#### SCEV的目标：

```cpp
for (i = 0; i < N; i++)
  A[i] = ...
```

编译器会想知道：

- `i` 会取哪些值？

- `i` 会不会溢出？

- `A[i]` 是否越界？

- 两个访问 `A[i]` 和 `A[i+1]` 是否冲突？

- 这个循环能不能：

  - vectorize

  - unroll

  - tile

  - 并行化

- SCEV是回答这些问题的**基础**

#### 输入输出

输入可以是多种形式：

| 输入          | 来自                   |
| ------------- | ---------------------- |
| CFG           | 控制流                 |
| LoopInfo      | 循环结构               |
| IR 指令       | PHI / add / mul / icmp |
| DataLayout    | 位宽 / 对齐            |
| DominatorTree | 辅助判断               |

**输出：**

- **一个SCEV表达式树**
  - 标量值随 loop iteration 的变化规律
  - 上界 / 下界（有时）
  - stride / recurrence

- 一些常用SCEV 的核心表达式类型：

  - `SCEVConstant`

    ```
    5
    ```

  - `SCEVUnknown`

    ```
    N
    ```

    - 运行期才知道
    - 但 SCEV 会当成一个“符号”

  -  `SCEVAddExpr`

    ```
    i + 3
    ```

  -  `SCEVMulExpr`

    ```
    i * 4
    ```

  -  `SCEVAddRecExpr`（循环递推，最重要）

    ```
    {Start, +, Step}
    ```

    表示：

    ```
    Start + Iteration × Step
    ```

#### AI Compiler中的SCEV用途：

**内存访问分析**：

- 判断 stride
- 判断是否连续
- 判断 alias （ 两个不同的指针 / 内存访问，可能指向同一块内存）
- 判断可 vectorize

**Tiling/Fusion 合法性**：

- producer / consumer 是否访问同一 tile
- 判断是否存在 loop-carried dependence （当前迭代依赖上一次（或更早）迭代的结果）

**并行化：**

- 如果SCEV 证明访问互不冲突，则可以并行

### 4.4 PostDominatorTree（PDT）

PostDominatorTree 描述的是：**程序“无论怎么走，最终一定会经过哪里”**

和 DominatorTree 正好是**时间方向相反**的概念：

| 概念              | 关注点                          |
| ----------------- | ------------------------------- |
| **Dominator**     | 从 *entry* 出发，**必经**       |
| **PostDominator** | 从某点出发，到 *exit*，**必经** |

#### Some Concepts：

- Post-dominance:

  - 节点 **A post-dominates B**, 当且仅当： **从 B 出发到任意 exit 的所有路径，都必须经过 A**
  - 记作：

  ```
  A pdom B
  ```

-  Immediate Post-Dominator（ipdom）

  - A 是 B 的 immediate post-dominator， 当且仅当：
    - A post-dominates B
    - 且 A 是离 B 最近的那个 post-dominator

- PostDominatorTree（PDT）

  - 把 `ipdom` 关系连成一棵树：

    ```
    exit
     └── ...
         └── A
             └── B
                 └── C
    ```

  - 这棵树：

    - 根节点是 **exit**
    - 子节点在“控制流收敛意义上”更早结束

#### 常用API

- **判断 post-dominance：**

  ```cpp
  PDT.dominates(A, B);
  ```

  - 名字还是 `dominates`，但**语义是 post-dominates**

- **取 immediate post-dominator**

  ```cpp
  DomTreeNodeBase<BasicBlock> *Node = PDT.getNode(BB);
  BasicBlock *IPDom = Node->getIDom()->getBlock();
  ```

- **查公共 post-dominator（超级常用）**

  ```cpp
  BasicBlock *CPD = PDT.findNearestCommonDominator(BB1, BB2);
  ```

  - 找 if-else 的 merge block
  - 找异常路径的最终汇合点
  - 找所有路径“最终都会执行”的 cleanup



## 5. Transformation Pass

需要关注的核心问题：**在什么条件下，修改 IR 仍然语义等价？**

### 5.1 Leagality Check

**在变换前后，程序在所有可观察行为上等价**

#### 控制流合法性（CFG Legality）

往往需要考虑：

- 是否引入了不可达 block？
- 是否破坏了 dominance / post-dominance？
- 是否破坏异常路径？

#### 内存依赖合法性

需要保证：**重排 / 拆分 / 合并的内存访问，在所有路径上语义一致**

尝尝会用到以下工具：

- AliasAnalysis (AA)
- MemorySSA
- DependenceAnalysis

#### 副作用与不可移动指令

以下指令**基本不能乱动**：

- `volatile load/store`
- `atomic`
- `call`
- I/O/syscalls

####  控制依赖合法性（if / predication）

必须保证：

- 原来在某些路径不执行的指令
- 现在不会被“强制执行”

通过判断：**目标位置 post-dominates 原位置**

#### 循环语义合法性

对 loop 变换（tiling / unroll / interchange）：

需要检查：

- Loop 是否 canonical
- 是否存在 loop-carried dependence
- 是否是单 exit / 单 latch
- induction variable 是否可重建

#### 目标相关合法性

在 AI / NPU / GPU 编译里非常重要：

- 是否超出 scratchpad / SRAM
- 是否破坏 vector width / warp 语义
- 是否破坏 memory alignment

并非是与平台无关的

### 5.2 经典范式：

**Guarded Transformation**：

```
if (!isLegal(...)) return PreservedAnalyses::all();
doTransform();
return PreservedAnalyses::none();
```

特点：

- 保守，稳
- LLVM 主线大量使用

在AI Compiler里，有如下例子：（当前先记录，后续进一步了解）

**为 NPU 做 Tiling**：

**Phase 1**

- SCEV → affine index
- Dependence → 无 loop-carried dep
- Memory model → SRAM fits

**Phase 2**

- legality：
  - no alias across tiles
  - no cross-tile dependence
  - tile size aligned to vector width
- transform：
  - split loop
  - rebuild induction
  - insert prologue/epilogue



## 6. Pass Pipeline 与优化顺序

Pass Pipeline: **让一堆“各自正确的 Pass”在一起仍然正确、而且更强**

- 一组 Pass + 执行顺序 + 执行粒度

### 6.1 Pipeline的维度

#### Pass 粒度

| 粒度         | Pass 看到的东西 |
| ------------ | --------------- |
| ModulePass   | 全程序          |
| CGSCCPass    | 递归调用图      |
| FunctionPass | CFG / Loop      |
| LoopPass     | 单个 loop       |
| BasicBlock   | 单个 BB         |

#### Analysis / Transformation 交错

一个成熟的pipeline应该为如下架构：

```
[Canonicalize]
→ [Analysis]
→ [Transform]
→ [Cleanup]
→ repeat
```

#### Canonical Form

**Pipeline 的真正目标不是“优化”，而是把 IR 推向某种 canonical form**

- Canonical IR = **结构统一、形式稳定、便于分析和变换的 IR 表达**
- 其不一定是性能高的，但是一般易分析和变换，更加关注结构

- 例如

  - SSA

  - Loop Simplify

  - LCSSA

  - Affine loop

### 6.2 LLVM Pipeline 常见顺序

- **IR 正规化**：

  ```
  - mem2reg
  - instcombine （折叠指令成最简单形式，较为激进，常多轮执行）
  - simplifycfg （整理控制流结构，如删除空BB，合并单前继BB，删除不可达分支等）
  ```

  - 去除栈变量
  - 把 IR 变“干净”
  - 统一结构

- **标量优化（反复多轮）**：

  ```
  - early-cse
  - gvn
  - licm （把循环中“不变”的指令挪出循环）
  - dce
  - reassociate
  ```

  - 小步快跑
  - 多轮 fixed-point

- **循环优化**：

  ```
  - loop-simplify
  - lcssa
  - licm
  - loop-unroll
  - loop-vectorize
  ```

  - 注意要先simplify, 再transform

- **清理**：

  ```
  - instcombine
  - dce
  - simplifycfg
  ```

  - Transformation 往往会：
    - 引入临时变量
    - 产生死代码
    - 打破简洁表达
  - 需要Cleanup Pass进行清理
