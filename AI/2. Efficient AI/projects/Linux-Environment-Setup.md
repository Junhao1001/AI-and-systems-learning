# Linux Environment Setup

## Windows环境下安装Linux子系统

- 使用WSL2 在Windows系统上安装Linux子系统，基本参考如下教程：

  [【保姆级教程】Windows上安装Linux子系统，搞台虚拟机玩玩](https://zhuanlan.zhihu.com/p/689560472)

  - 实际选择了其中先安装后转移的方案
  - 根据GPT推荐，安装了Ubuntu-20.04 (适合TVM/AI-Compiler开发，不确定是否准确)

### WSL安装问题记录

#### 1. “已禁止(403)”报错

- 在命令行输入wsl安装命令，报错如下:

  ```
  C:\Windows\System32>wsl --install
  已禁止(403)
  ```

- 原因：
  - Windows 访问 Microsoft 在线资源被拒绝了
  - 常见环境：
    - 校园网
    - 公司网
    - 开了代理 / VPN / Clash
    - DNS 被污染或劫持
- 最终解决方案：
  - **校园网换手机热点**

## Linux环境基础依赖配置

### 基础环境配置

```shell
sudo apt install -y \
  build-essential \
  clang \
  unzip \
  zip \
  cmake \
  git \
  gdb \
  ninja-build \
  python3-dev \
  python3-pip\
  libssl-dev
```

### Cmake安装

- [ubuntu 20.04安装(升级)cmake](https://zhuanlan.zhihu.com/p/519732843)

- 直接使用`apt install`安装的cmake版本太低，需要到官网下载安装

- 官网下载：

  ```
  wget https://cmake.org/files/v3.28/cmake-3.28.6.tar.gz
  ```

- 解压：

  ```
  sudo tar -zxvf cmake-3.28.6.tar.gz
  ```

- 安装

  ```
  cd cmake-3.28.6
  sudo ./configure
  sudo make -j8 
  sudo make install
  ```

- 若要卸载，进入到安装时执行`make install`时的路径下， 执行卸载命令：

  ```
  sudo make uninstall
  ```

- 后续在tvm源码编译时，在conda环境里也可以安装特定版本的cmake

  ```
  conda install -c conda-forge cmake=3.24
  ```

  

### LLVM安装(>=15)

- **也可以在conda环境中进行安装，但是tvm编译可能会有问题**

  ```
  conda install -c conda-forge llvmdev=15
  ```

  

- 使用 **Ubuntu 官方 LLVM PPA**

  ```
  # 添加 LLVM 官方仓库
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh
  sudo ./llvm.sh 15
  
  # 配置 clang/llvm
  sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
  sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
  
  # 检查版本
  clang --version
  llvm-config --version
  ```

#### llvm-config --version没有输出

- 先确认是否安装成功：

  ```
  ls /usr/bin | grep llvm
  ```

- 大概率可以看到

  ```
  llvm-config-15
  llvm-ar-15
  llvm-as-15
  ...
  ```

- 再次确认：

  ```
  llvm-config-15 --version
  ```

- **LLVM 15 实际是装成功了，只是系统默认没有创建 llvm-config 的软链接**

- **设置默认 llvm-config 指向 15**

  ```
  sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-15 100
  ```

### Miniconda安装

- 安装命令

  ```
  # 下载 Miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  
  # 按提示安装到默认路径（~/.miniconda3）
  source ~/.bashrc
  
  # 创建 TVM Python 环境
  conda create -n tvm python=3.9 -y
  conda activate tvm
  ```

- 在安装时，最好使用普通用户权限，而不是直接安装在`/root`目录下

- 安装好conda后，安装一些基础库

  ```
  conda install numpy scipy psutil decorator attrs -y
  pip install topi tvm
  ```


## TVM安装

从源码安装TVM： [从源代码构建和安装 TVM 软件包](https://tvm.hyper.ai/docs/getting-started/installing-tvm/install-from-source/)

- 下载源码：

  ```
  cd ~/workspace
  git clone --recursive https://github.com/apache/tvm.git
  cd tvm
  mkdir build
  cp cmake/config.cmake build
  cd build
  ```

- 调整编译选项(复制到config.cmake尾部)：

  ```
  # 控制默认编译标志（可选值：Release、Debug、RelWithDebInfo）
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
  
  # LLVM 是编译器端的必需依赖项
  set(USE_LLVM "llvm-config --ignore-libllvm --link-static")
  set(HIDE_PRIVATE_SYMBOLS ON)
  
  # GPU SDK，按需启用
  set(USE_CUDA   OFF)
  set(USE_METAL  OFF)
  set(USE_VULKAN OFF)
  set(USE_OPENCL OFF)
  
  # 支持 cuBLAS、cuDNN、cutlass，按需启用
  set(USE_CUBLAS OFF)
  set(USE_CUDNN  OFF)
  set(USE_CUTLASS OFF)
  ```

- 然后：

  ```
  cmake .. && cmake --build . --parallel $(nproc)
  cmake .. -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ -DCMAKE_LINKER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ld
  ```

- 最后：

  ```
  cd ..
  pip install -e .
  ```

#### conda环境下安装LLVM后，TVM编译报错

- 一开始我在系统环境下安装了LLVM15，当时成功编译了TVM，没有报错

- 后面再GPT的建议下，我选择卸载了系统环境里的LLVM，在conda的虚拟环境里下载了LLVM-15，并尝试build TVM

- 但是一开始出现链接错误，无法正常链接，询问GPT，它让我在conda环境里重新安装GCC

  ```
  全部使用 Conda 工具链，不要混用系统 /usr/bin/g++ 和 Conda LLVM
  ```

- 因此我在conda环境里重新安装了相关工具链，和系统环境分割开，最终使用如下命令编译：

  ```
  cmake .. -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ -DCMAKE_CXX_STANDARD=17
  ```

- 结果还是编译失败，报错为：

  ```
  [4/64] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/disco/disco_worker.cc.o [5/64] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o FAILED: [code=1] CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o /home/zhujunhao/miniconda3/envs/tvm-build-venv/bin/x86_64-conda-linux-gnu-g++ -DDMLC_USE_FOPEN64=0 -DDMLC_USE_LOGGING_LIBRARY="<tvm/runtime/logging.h>" -DNDEBUG=1 -DTVM_FFI_CMAKE_LITTLE_ENDIAN=1 -DTVM_INDEX_DEFAULT_I64=1 -DTVM_KALLOC_ALIGNMENT=64 -DTVM_LLVM_HAS_AARCH64_TARGET=1 -DTVM_LLVM_VERSION=150 -DTVM_THREADPOOL_USE_OPENMP=0 -DUSE_FALLBACK_STL_MAP=0 -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -DDMLC_ENABLE_RTTI=0 -I/home/zhujunhao/workspace/tvm/include -I/home/zhujunhao/workspace/tvm/3rdparty/libcrc/include -I/home/zhujunhao/workspace/tvm/3rdparty/tvm-ffi/include -I/home/zhujunhao/workspace/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include -isystem /home/zhujunhao/workspace/tvm/3rdparty/dmlc-core/include -isystem /home/zhujunhao/workspace/tvm/3rdparty/rang/include -isystem /home/zhujunhao/workspace/tvm/3rdparty/compiler-rt -isystem /home/zhujunhao/workspace/tvm/3rdparty/picojson -faligned-new -O2 -Wall -fPIC -fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/zhujunhao/miniconda3/envs/tvm-build-venv/include -O2 -g -DNDEBUG -ffile-prefix-map=..=/home/zhujunhao/workspace/tvm -std=gnu++17 -fno-rtti -MD -MT CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o -MF CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o.d -o CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o -c /home/zhujunhao/workspace/tvm/src/target/llvm/codegen_blob.cc In file included from /home/zhujunhao/workspace/tvm/src/target/llvm/codegen_blob.cc:27: /home/zhujunhao/miniconda3/envs/tvm-build-venv/include/llvm/ADT/SmallVector.h:88:69: error: 'uint64_t' was not declared in this scope 88 | typename std::conditional<sizeof(T) < 4 && sizeof(void *) >= 8, uint64_t, | ^~~~~~~~ /home/zhujunhao/miniconda3/envs/tvm-build-venv/include/llvm/ADT/SmallVector.h:28:1: note: 'uint64_t' is defined in header '<cstdint>'; this is probably fixable by adding '#include <cstdint>' 27 | #include <limits> +++ |+#include <cstdint> 28 | #include <memory> /home/zhujunhao/miniconda3/envs/tvm-build-venv/include/llvm/ADT/SmallVector.h:89:31: error: 'uint32_t' was not declared in this scope 89 | uint32_t>::type;
  ```

- 大致意思是`llvm/ADT/SmallVector.h`中找不到`uint64_t`的类型？

- 我手动在这个文件里加上了  `include <cstdint>`, 然后编译成功了

- 但是后来我在系统环境里重新安装了llvm，检查了`SmallVector.h`也确实没有include这个库，但确实能编译成功，不知道究竟什么原因了

- 我不愿意改源码，最终还是选择了在系统环境里安装了LLVM,并进行了TVM的编译

  

#### 源码安装TVM检验时报错No module named 'tvm_ffi'

- Apache TVM relies on the tvm-ffi package to support its python bindings. Therefore, after we finish the build, we need to install the tvm-ffi package.

- 解决方案：

  ```
  cd 3rdparty/tvm-ffi; pip install .; cd ..
  ```

  

