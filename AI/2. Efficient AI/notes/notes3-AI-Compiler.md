# AI Compiler

## 1. AI Compiler 核心概念

AI Compiler 是： **把“高层数学计算（张量/算子）” 逐层降低（lowering）成 “具体硬件上最高效的执行形式”**

### IR (Intermediate Representation)

不同于LLVM IR，AI Compiler IR更加关注：

- 更关注张量/算子 对象
- 关注数据流，而非LLVM IR中的控制流
- 典型优化是Fusion/Layout
- 核心原因在于**神经网络是“张量计算图”**

#### AI Compiler  IR层级

AI Compiler IR是分层的：

```
Graph IR  →  Tensor IR  →  Loop IR  →  Instruction IR
```

不同层IR具体关注点如下：

| 层级      | 关注点      | 典型 IR          |
| --------- | ----------- | ---------------- |
| Graph IR  | 算子 & 拓扑 | Relay / FX / HLO |
| Tensor IR | 张量计算    | Linalg / TIR     |
| Loop IR   | 循环结构    | Affine / SCF     |
| Low IR    | 指令        | LLVM IR          |

### 计算图（Computation Graph）

计算图基本结构：

- **节点**：算子（Conv / MatMul / ReLU）
- **边**：张量（Tensor)

```
Input → Conv → ReLU → MatMul → Output
```

需要注意这是 **静态数据流图**：

- 没有 if / while（推理时）
- 执行顺序由数据依赖决定 （有些可以并行执行）

**张量**： 带 shape 的多维数组 + 语义

- 不能将张量简单理解成数组，其还包含
- 数据
- 维度（shape）
- 数据类型（dtype）
- 语义（batch / channel / spatial）

### 算子 (Operator)

算子: 高层数学原语

- Conv2D
- MatMul
- LayerNorm

需要注意算子**更多是一个数学定义，而不是实现**；具体如何实现是**编译器的实现**

- **kernel 是编译器生成的，不是调用的**

整体分层可以如下理解：

```
模型作者：用哪些算子（Conv / MatMul / LN）
编译器前端：确认算子语义
编译器中端：决定算子如何组合 / 变形
编译器后端：决定算子如何实现
```

结合具体IR层级:

| 层级                    | 决定什么                       |
| ----------------------- | ------------------------------ |
| **Frontend / Graph IR** | 用哪些算子（Conv？GEMM？）     |
| **Middle IR**           | 算子是否 fusion、是否替换实现  |
| **Backend**             | 每个算子对应哪个 kernel / 指令 |

### Lowering

Lowering指**将“抽象高的表示” 转成“更接近硬件的表示”**

例如：

```
Conv2D
 ↓
Linalg Ops
 ↓
Nested Loops
 ↓
Vector Instructions
```

### Optimization

#### Operator Fusion（算子融合）

```
Conv → ReLU → Add
↓
ConvReLUAdd
```

目的：

- 减少内存访问
- 提升数据局部性
- 并不是简单的子函数的融合，而是**要确认中间值不会重新写内存（存在于寄存器/SRAM)**

#### Layout Transformation（数据布局）

```
NCHW ↔ NHWC
```

不同硬件偏好不同布局：

- GPU：NHWC
- CPU：NCHW
- NPU：自定义

#### Tiling / Blocking

```
for i in N:
  for j in M:
    ...
```

→ 拆成 cache / SRAM 友好的小块

#### Memory Planning

- 中间张量是否复用？
- buffer 多大？
- on-chip 还是 DRAM？

### Others

除了上面概念之外，还有如下概念，在此先不展开：

- **Schedule（调度）**：显示控制整个流程的顺序
- **Hardware Mapping（硬件映射）**
- **Auto-Tuning（自动搜索）**

## 2. MLIR

MLIR: **一个“可以同时存在多种 IR，并且明确描述 lowering 路径”的 IR 框架**

MLIR的三个核心设计思想：

- **IR 是可扩展的（Dialect）**: 无需等LLVM官方支持你的算子

  - 定义自己的 Dialect
  - 定义自己的 Op
  - 定义自己的 Type

- **多层 IR 共存**：

  ```
  Torch Dialect
     ↓
  Linalg Dialect
     ↓
  SCF / Affine
     ↓
  LLVM Dialect
  ```

- **Lowering 是显式的**

  - Lowering = Rewrite Pattern + Pass Pipeline

### MLIR基本构成

- **Module / Operation / Region / Block**

- 简单类比如下：

| MLIR      | 类比        |
| --------- | ----------- |
| Module    | 翻译单元    |
| Operation | 指令 / 算子 |
| Region    | 函数体      |
| Block     | 基本块      |