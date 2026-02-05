# Machine Learning Compilation

## 1. Introduction

### 1.1 What is MLC

- 传统的编译：将高级编程语言编写的源程序转换成机器语言目标程序的过程

- 机器学习编译更像是“**部署（deployment）**”的概念

  <img src="assets/image-20260205150505596.png" alt="image-20260205150505596" style="zoom:50%;" />

  - 将机器学习的执行从**开发形式(Development Form)**转换并优化为**部署形式(Deployment Form)**的过程

  - **开发形式** 是指我们在开发机器学习模型时使用的形式。典型的开发形式包括用 PyTorch、TensorFlow 或 JAX 等通用框架编写的模型描述，以及与之相关的权重

  - **部署形式** 是指执行机器学习应用程序所需的形式。它通常涉及机器学习模型的每个步骤的支撑代码、管理资源（例如内存）的控制器，以及与应用程序开发环境的接口（例如用于 android 应用程序的 java API）

    <img src="assets/dev-deploy-form.png" alt="../_images/dev-deploy-form.png" style="zoom: 50%;" />

    - 将模型逐层分解：模型 -> 计算图 -> 算子 -> 循环/硬件映射 ->硬件指令
    - 根据硬件平台的不同，每一层的实现都可以灵活实现

  - 传统编译和机器学习编译的另一张对比图，便于理解：

  <img src="assets/image-20260129141001475.png" alt="image-20260129141001475" style="zoom:50%;" />

- MLC的目标：

  - **Integration and Dependency Minimization**：
    - 将必要的元素组合在一起以用于部署应用程序
    - 该能力能够减小应用的大小，并且可以使应用程序部署到的更多的环境
  - **Leverage Hardware Native Acceleration**
    - 利用硬件本身的特性进行加速
  - **Optimization in General**
    - 优化内存使用
    - 提升执行效率
    - .....

### 1.2 Why Learning MLC

- **构建机器学习部署解决方案**
  - 在将模型进行实际部署时，往往涉及很多复杂的环节
  - MLC希望提供一套解决方案，能够有效解决部署过程中遇到的**内存占用、性能优化、依赖项最小化**等问题
- **深入理解现有深度学习框架**
  - 掌握一些底层技术原理，有利我们构建与**底层技术协同优化的模型**
- **构建新兴硬件的软件栈**
  - 机器学习编译技术提供了构建软件栈的工具
  - 能够持续适应**新型硬件**加速功能与**模型演进**

### 1.3 Key elements in MLC

- **计算图（Computation Graph)**

  - **节点**：**Tensor Function**
    - 不一定对应计算的单个步骤（单个算子），部分计算甚至整个端到端计算可以看做张量函数
  
  - **边**： **Tensor**

<img src="assets/image-20260203001142886.png" alt="image-20260203001142886" style="zoom:67%;" />

- Abstraction and Implementation:

  - **Abstraction**: 抽象是指以不同方式表示同一张量函数(tensor function)
  - **Implementation**: 在实践中，更专业的版本是高层抽象的实现

  - 因此大多数MLC过程可视为**在相同或不同抽象下转换和组装张量函数的过程**

- 四个抽象层级（MLC 的 四个层级，基于TVM，但是其他深度学习编译框架也差不多）
  - Computational Graphs
  - Tensor Programs
  - Libraries and Runtimes
  - Hardware Primitives

<img src="assets/image-20260205113003802.png" alt="image-20260205113003802" style="zoom:50%;" />

1. 

## 2. Common AI Compilers

### 2.1 TVM (Tensor Virtual Machine)

Apache TVM 是一个机器学习编译框架，**遵循Python 优先开发和通用部署的原则。** 

它接受经过预训练的机器学习模型，编译并生成可部署的模块，这些模块可以嵌入到任何地方运行。Apache TVM 还支持自定义优化过程，以引入新的优化方法、库、代码生成等等。

- **Python 优先：** 优化过程完全可以在 Python 中自定义。无需重新编译 TVM 栈即可轻松定制优化流程。
- **可组合性：** 优化过程具有可组合性。可以轻松地将新的优化过程、库以及代码生成器组合进已有的流程中

#### 2.1.1 Key Flow （Deployment）

1. **导入 / 构建一个机器学习模型**
   - TVM 支持从多种框架导入模型用于通用机器学习模型，如 PyTorch、TensorFlow。同时对于大语言模型场景，也可以使用 Relax 前端直接创建模型。
2. **通过** `pipelines`**执行可组合的优化转换**
   - pipeline 封装了一系列转换，以实现两个目标：
     - **图优化：** 如算子融合、布局重写等。
     - **张量程序优化：** 将算子映射到底层实现（包括库或代码生成器）
3. **构建并通用部署**
   - Apache TVM 旨在提供一种通用部署方案，将机器学习带到任何地方，以最少的运行时支持适配各种语言。TVM 的运行时可在非 Python 环境中运行，因此适用于移动端、边缘设备，甚至是裸机设备。

#### 2.1.2 Architecture

编译流程：

- **导入：** 前端组件将模型引入到 IRModule 中，它包含了内部表示模型的函数集合。
- **转换：** 编译器将 IRModule 转换为功能与之等效或近似等效（例如在量化的情况下）的 IRModule。许多转换与 target（后端）无关，并且允许 target 配置转换 pipeline。
- **Target 转换：** 编译器将 IRModule 转换（codegen）为指定 target 的可执行格式。target 的转换结果被封装为 runtime.Module，可以在 runtime 环境中导出、加载和执行。
- **Runtime 执行：** 用户加载 runtime.Module，并在支持的 runtime 环境中运行编译好的函数

<img src="assets/image-20260205112800326.png" alt="image-20260205112800326" style="zoom:50%;" />

#### 2.1.3 Key Concepts

- **IRModule**:是整个堆栈中使用的主要数据结构。一个 IRModule（intermediate representation module）包含一组函数。目前支持两种主要的功能变体（variant）
  - **relay::Function** 是一种高层功能程序表示。一个 relay.Function 通常对应一个端到端的模型
  - **tir::PrimFunc** 是一种底层程序表示，包含循环嵌套选择、多维加载/存储、线程和向量/张量指令的元素。通常用于表示算子程序，这个程序在模型中执行一个（可融合的）层
  - 在编译期间，Relay 函数可降级为多个 tir::PrimFunc 函数和一个调用这些 tir::PrimFunc 函数的顶层函数
- **TensorIR**:  是 Apache TVM 栈中的核心抽象之一，用于表示和优化原始的张量函数
- **Relay (Relax)**: Relax 是 Apache TVM 栈中用于图优化和转换的高级抽象层。此外，Apache TVM 将 Relax 和 TensorIR 结合在一起，作为跨层优化的统一策略。因此，Relax 通常与 TensorIR 紧密协作，用于表示和优化整个 IRModule

#### 2.1.4 Shorts

- 虽然支持多种后端（CPU、GPU、ARM、NPU等），但对**非主流或新兴硬件**（特别是定制化AI加速器）的支持需要投入大量工程工作
- 早期 TVM 对动态形状（Dynamic Shapes）的支持较弱，但是好像后续开发的Relax在不断增加对这方面的支持。当前具体支持程度需要后续再学习调研



### 2.2 MNN/TFLite/NNAPI

三者都是移动端的**通用高效推理引擎**

- TensorFlow Lite (TFLite): 谷歌发布的高效的移动深度学习框架  (Google, 2017)。**TF-Lite 针对性能较弱的设备（如移动电话和嵌入式设备）进行了优化**
- 对于 Android 智能手机，Google 也提供了自己的设备推理解决方案，即 ML-kit 和神经网络 API (NNAPI) (Google，2016)
- 这里主要针对MNN来概括

#### 2.2.1 MNN Features

- 轻量性
- 通用性
  - 支持 Tensorflow、Caffe、ONNX、Torchscripts 等主流模型文件格式，支持CNN / RNN / GAN / Transformer 等主流网络结构。
  - 支持多输入多输出，支持任意维度的输入输出，支持动态输入（输入大小可变），支持带控制流的模型
  - 算子丰富，支持 178 个Tensorflow Op、52个 Caffe Op、163个 Torchscipts Op、158 个 ONNX Op（ONNX 基本完整支持）
  - 支持 服务器 / 个人电脑 / 手机 及具有POSIX接口的嵌入式设备，支持使用设备的 CPU / GPU 计算，支持部分设备的 NPU 计算（IOS 11 + CoreML / Huawei + HIAI / Android + NNAPI）
  - 支持 Windows / iOS 8.0+ / Android 4.3+ / Linux  及具有POSIX接口的操作系统
- 高性能：
  - 对iOS / Android / PC / Server 的CPU架构进行了适配，编写SIMD代码或手写汇编以实现核心运算，充分发挥 CPU的算力，单线程下运行常见CV模型接近设备算力峰值
  - 支持基于 Metal / OpenCL / Vulkan 使用移动端设备上的GPU进行推理
  - 支持基于 CUDA 使用 PC / Server 上的 NVIDIA GPU 实现更快速的推理
  - 广泛运用了 Winograd 卷积算法提升卷积性能，首次在业界工程实践中实现转置卷积的Winograd算法优化与矩阵乘的Strassen算法优化，并取得加速效果
  - 支持低精度计算（ int8 / fp16 / bf16）以提升推理性能。
- 易用性

MNN提出了三种核心创新（不完全展开）：

- 运行时半自动搜索架构
  - TVM 为代表的的全自动搜索（i.e. 自动调优)
  - NCNN 为代表的全手动搜索（i.e. 手工实现每个 case）
  - **MNN 提出了一个特殊的处理过程，称为「预推理」。预推理过程中，会提前进行算子的计算策略选择和资源分配**
- 卷积算法优化创新
- 异构设备混合调度

#### 2.2.2 Differences between AI Compiler and Inference Engine ？

(这个问题还是基于chatgpt，对于推理引擎的理解还比较浅)

MNN:

- MNN也支持一些编译器做的事情：图优化、算子融合、多硬件支持
- **不支持IR分层**
- 不支持自动调度搜索（但是提出了半自动搜索架构，但是支持的算子仍然有限）
- **不是通用 compiler infrastructure**：不能拿去支持“任意新硬件”

推理框架做的事（AI提供，不确定准确性）

```
模型文件
  ↓
解析网络结构
  ↓
构建执行图（Execution Graph）
  ↓
选择 backend（CPU / GPU / NPU）
  ↓
调度算子
  ↓
调用 kernel 执行
```

而AI Compiler:

```
模型
 ↓
Graph IR
 ↓
Tensor / Loop IR
 ↓
Codegen
 ↓
真正的高效 kernel
```

- 给出算子描述，通过schedule和lowering，生成kernel。**kernel是结果而不是输入**

但对于我们的目标来说，“模型在端侧平台的快速部署”，是否可以选择这种推理引擎：

- MNN也支持多种硬件，也许够我们用了？
- 半自动推理的自动化究竟有多高？手写算子的需求究竟如何？需要进一步探究

### 2.3 CoreML

Core ML 是苹果推出的机器学习框架，支持在 iOS、macOS、watchOS 和 tvOS 设备上本地运行机器学习模型。其核心特点包括：

- 设备端执行‌：所有计算在设备上完成，无需网络连接，保障数据隐私和实时性‌。
- 统一模型格式‌：支持神经网络、决策树、支持向量机等多种模型类型，统一转换为 .mlmodel 格式‌。
- 硬件优化‌：利用 CPU、GPU 和神经网络引擎（[Neural Engine](https://zhida.zhihu.com/search?content_id=260261833&content_type=Article&match_order=1&q=Neural+Engine&zhida_source=entity)）加速计算，降低功耗和内存占用‌

#### 2.3.1 Key Flow

<img src="assets/image-20260205151324091.png" alt="image-20260205151324091" style="zoom:50%;" />

- 使用 Xcode 内置的Create ML App来构建和训练模型；或者使用 coremltools 将其他框架的模型转换为 Core ML 格式‌
- CoreML可以执行算子融合,精度转换（FP32 → FP16 / INT8）,内存布局优化等优化操作
- Core ML 会分析算子，根据支持度和性能，将任务分配到不同硬件上

#### 2.3.2 Differences with TVM

- Core ML 是**“封闭式 AI Compiler”**+与**”硬件绑定的Runtime“**
- Core ML 不支持自定义算子、改 lowering 规则、接受新硬件
- 简单来说，Core ML 是工业闭源实现，适合产品开发者



## 3. Tensor Program

### 3.1 Primitive Tensor Function

- 元张量函数：最基本的张量计算单元，个人认为就是我们平时说的”算子“

  - 许多不同的抽象能够实现同样的元张量函数
  - 许多机器学习编译过程中，会将元张量函数变为**更加专门的、针对特定工作和部署环境的函数**

  <img src="assets/image-20260203155848348.png" alt="image-20260203155848348" style="zoom:50%;" />

- 简单提及一些**算子内部优化**的例子

  - 将函数映射到专门的算子库
  - 上面例子里展示的并行计算实现
  - ......

### 3.2 Tensor Program Abstraction

Tensor Program 更关注于循环的优化和数据的排布

元张量函数实现的抽象往往包含以下几个部分：

- 存储数据的多维数组（Multi-dimensional buffers)
- 驱动张量计算的循环嵌套（Loop nests）
- 计算部分本身的语句（Computations)

<img src="assets/image-20260203162901340.png" alt="image-20260203162901340" style="zoom:50%;" />

为什么要写成特定形式的张量程序？为什么不直接用C语言/Cuda去实现呢？

- 当前理解：方便实现优化的自动化
- 可以应用一些**Program-based transformation**,用于加速张量程序的执行
- 张量程序中额外的结构能够为程序变换提供更多的信息（如上图中的spatial）

loop optimization三板斧（）：

1. Fusion
2. Tiling
3. vectorization

## Reference

1. [模型加速与 AI compiler 介绍](https://zhuanlan.zhihu.com/p/617043119)

2. [Machine Learning Compilation 课程](https://book-zh.mlc.ai/)

3. [Apache TVM 中文文档](https://tvm.hyper.ai/docs/)

4. [Relax: TVM 的下一代图层级 IR](https://zhuanlan.zhihu.com/p/523395133)

5. [TVM，MLIR，LLVM各自有哪些缺陷？](https://zhuanlan.zhihu.com/p/1990662498647574219)

6. [CoreML 的优势与性能](https://zhuanlan.zhihu.com/p/1927420389815988747)

7. [CoreML简体中文文档](https://developer.apple.com/cn/documentation/coreml/)

8. [人工智能编译器MLIR-官方入门教程讲解](https://www.bilibili.com/video/BV1Hd4y1U7mb/?spm_id_from=333.1387.favlist.content.click&vd_source=47b9e94682446eba3bcd8ada1d947692)

9. [MNN文档](https://mnn-docs.readthedocs.io/en/latest/)

10. [AI编译器和推理引擎的区别](https://bbs.huaweicloud.com/blogs/398747)

11. [MNN: A UNIVERSAL AND EFFICIENT INFERENCE ENGINE将模型适配到各种终端硬件的解决方案，加速，量化，保精度](https://blog.csdn.net/weixin_43424450/article/details/144449776)

    