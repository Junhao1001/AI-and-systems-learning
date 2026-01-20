# PASS

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

- opt 运行 

  - 文件位置根据实际情况改动

  ```cmd
  opt -load-pass-plugin build\Release\HelloPass.dll -passes=hello-pass .\MyFirstIR\add.ll  -disable-output
  ```

  

