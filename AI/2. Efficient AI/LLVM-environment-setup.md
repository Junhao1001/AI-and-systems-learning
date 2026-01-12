# LLVM 环境配置

之前本地已经安装了Windows + Visual Studio 2022 + VS Code，具体过程可以见[setup](../../systems/projects/Local environment setup.md)

**下载**：

```
LLVM-21.x.x-win64.exe
(AI推荐 17.x.x 会和教程更为适配，这里我下载最新的21)
```

**安装时需要**：

- 勾选 **Add LLVM to PATH**
- 安装路径记住

```
C:\Program Files\LLVM
```

**验证安装**：

```
打开PowerShell / CMD：

clang --version
opt --version
llvm-config --version
```

上述都能输出版本号，则安装成功

实操中，后两者都没有输出：

- 检查文件是否存在

```
dir "C:\Program Files\LLVM\bin\opt.exe"
dir "C:\Program Files\LLVM\bin\llvm-config.exe"
```

- 如果找不到 `llvm-config.exe` → 说明 Windows release 没自带

- 解决：可以忽略它（Windows 上可以用 `clang --version` 验证 LLVM）

  - 在CmakeList.txt中：

  ```
  set(LLVM_DIR "C:/Program Files/LLVM/lib/cmake/llvm")
  find_package(LLVM REQUIRED CONFIG)
  ```

- **Windows 官方 LLVM release 自带的 17.x Installer 里确实没有 `opt.exe`** 
  - 需要自行编译LLVM源码
  - 或者使用vcpkg安装
  - 这里我尝试使用LLVM

## 自行编译LLVM

- 下载LLVM源码

```
git clone --depth 1 https://github.com/llvm/llvm-project.git
```

- 同时创建**构建目录**，和源码分离

```
D:\LLVM\llvm-build
```

- 配置CMake，打开 **“x64 Native Tools Command Prompt for VS 2022”**（VS自带的开发者命令行），输入如下Cmake命令

```
cmake -S D:\LLVM\llvm-project\llvm ^
      -B D:\LLVM\llvm-build ^
      -G "Visual Studio 17 2022" ^
      -A x64 ^
      -DLLVM_ENABLE_PROJECTS="clang" ^
      -DLLVM_TARGETS_TO_BUILD="X86" ^
      -DCMAKE_BUILD_TYPE=Release
```

- 开始编译：

```
cmake --build D:\LLVM\llvm-build --config Release --target opt
```

- 验证：

```
D:\llvm-build\Release\bin\opt.exe --version
D:\llvm-build\Release\bin\llvm-config.exe --version
```



## LLVM第一次编译失败

- 命令行没有报错，但是目标目录下没有生成opt.exe

```
D:\LLVM\llvm-build\Release\bin

上述目录下仅有四个文件
llvm-lit
llvm-it.py
llvm-min-tblgen
llvm-tblgen
```

- 说明并没有编译完全
- 输入如下命令

```
cmake --build D:\LLVM\llvm-build --config Release --target opt --verbose
```

- 报错如下：

```
生成失败。

“D:\LLVM\llvm-build\tools\opt\opt.vcxproj”(默认目标) (1) ->
“D:\LLVM\llvm-build\lib\Transforms\AggressiveInstCombine\LLVMAggressiveInstCombine.vcxproj”(默认目标) (10) ->
“D:\LLVM\llvm-build\lib\Analysis\LLVMAnalysis.vcxproj”(默认目标) (11) ->
“D:\LLVM\llvm-build\lib\DebugInfo\PDB\LLVMDebugInfoPDB.vcxproj”(默认目标) (33) ->
(ClCompile 目标) ->
  D:\LLVM\llvm-project\llvm\include\llvm\DebugInfo\PDB\DIA\DIASupport.h(25,1): error C1083: 无法打开包括文件: “atlbase.h”: No s
uch file or directory [D:\LLVM\llvm-build\lib\DebugInfo\PDB\LLVMDebugInfoPDB.vcxproj]
  D:\LLVM\llvm-project\llvm\include\llvm\DebugInfo\PDB\DIA\DIASupport.h(25,1): error C1083: 无法打开包括文件: “atlbase.h”: No s
uch file or directory [D:\LLVM\llvm-build\lib\DebugInfo\PDB\LLVMDebugInfoPDB.vcxproj]
  D:\LLVM\llvm-project\llvm\include\llvm\DebugInfo\PDB\DIA\DIASupport.h(25,1): error C1083: 无法打开包括文件: “atlbase.h”: No s
uch file or directory [D:\LLVM\llvm-build\lib\DebugInfo\PDB\LLVMDebugInfoPDB.vcxproj]
......
```

- **根因在于：Visual Studio 里没有安装 ATL（Active Template Library）组件**

  - LLVM 在 Windows 下，默认会启用 PDB/DIA 支持
  - 而 DIA 依赖 ATL

- 解决方案：

  - 打开 VS Installer
  - Modify
  - 在 **Individual components**（单个组件）里勾选： **C++ ATL for latest v143 build tools (x86 & x64)**
  - apply
  - 重新build:

  ```
  cmake --build D:\LLVM\llvm-build --config Release --target opt
  cmake --build D:\LLVM\llvm-build --config Release --target llvm-config
  ```

  - 成功生成

  
