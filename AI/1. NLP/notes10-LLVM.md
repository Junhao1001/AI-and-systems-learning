# LLVM

 “Getting Started with LLVM”：https://llvm.org/docs/GettingStarted.html



## LLVM在编译器中的角色

### 完整编译器

一个完整的编译器的流程如下所示：

```mathematica
源代码 (C / C++ / Rust / Swift ...)
   ↓
词法分析 (Lexer)
   ↓
语法分析 (Parser)
   ↓
语义分析 (Semantic Analysis)
   ↓
【中间表示 IR】
   ↓
中间代码优化
   ↓
指令选择 / 寄存器分配
   ↓
目标机器代码 (x86 / ARM / RISC-V)
```

#### 编译器的前端、中端、后端

**前端：**

- 从“源代码文本” → “中间表示（IR）”
- 核心职责：
  - 理解**语言本身**
  - 判断：
    - 语法是否正确？
    - 类型是否匹配？
    - 名字绑定是否正确？
  - **把“语言语义”准确映射到 IR**

- 完全语言相关，不关心CPU是x86还是ARM

```
源代码
  ↓
词法分析 (Lexer)
  ↓
语法分析 (Parser)
  ↓
抽象语法树 (AST)
  ↓
语义分析 (类型检查、作用域、符号表)
  ↓
中间表示 IR
```

**中端**：

- 作用：**对 IR 做“与语言无关、与硬件弱相关”的优化**
- 例如：
  - 常量折叠
  - 死代码消除
  - 循环优化
  - 函数内联
  - 公共子表达式消除
  - ...
- 核心职责：
  - 提高 **执行效率**
  - 降低 **资源消耗**
  - 保证 **语义等价**

- 更多关心**控制流、数据依赖、内存访问模式**

**后端**：

- 作用：**IR → 具体硬件的机器代码**

- 核心职责：
  - 利用硬件特性
  - 生成 **高性能、合法** 的指令序列
- 硬件相关，强烈依赖：
  - 指令集
  - 寄存器数量
  - 内存模型
  - SIMD / GPU 架构

```
LLVM IR
  ↓
指令选择 (Instruction Selection)
  ↓
寄存器分配 (Register Allocation)
  ↓
指令调度 (Scheduling)
  ↓
机器码 / 汇编
```

### LLVM IR的作用

在没有LLVM时：

- 编译器的前端、中端、后端强耦合
- 每种语言 × 每种硬件 = 一个编译器，优化逻辑要重复N次
- LLVM核心作用为：
  - **LLVM 不是编译器，而是编译器基础设施**
  - **LLVM 的核心价值在 IR + 优化 + 后端**
  - **LLVM 解决的是“多语言 × 多硬件”的工程复杂度问题**



## How LLVM works

### Some Concepts

- **常量传播（Constant Propagation）**: 如果某个值在编译期就能确定是常量，就把“用它的地方”直接替换成这个常量
- **DCE（Dead Code Elimination，死代码消除）**：删掉“永远不会被用到，或者永远不会执行”的代码

### SSA(Static Single Assignment)

- 静态单赋值：**每个名字只被赋值一次，且一旦定义就不可变**

  - 这是**中间表示层面**的规则，不是源语言的规则

- 举例来说：

  ```
  c语言：
  a = 1;
  a = a + 1;
  a = a * 2;
  
  SSA表示：
  a1 = 1
  a2 = a1 + 1
  a3 = a2 * 2
  ```

  - 值不等于变量
  - **每个SSA名字为一个不可变值**

- 其优势在于：

  - use-def链天然存在：
    - 每个值知道自己来自哪里，被谁使用
    - 常量传播、DCE直接进行图遍历
  - 没有别名写入的歧义：可以进行大胆的优化

- **φ（phi）节点**：

  - 关于分支合流的情况
  - C代码如下：

  ```c
  int y;
  if (cond)
      y = 1;
  else
      y = 2;
  return y;
  
  ```

  - SSA表示如下处理：

  ```makefile
  entry:
    br cond, then, else
  
  then:
    y1 = 1
    br merge
  
  else:
    y2 = 2
    br merge
  
  merge:
    y3 = φ(y1, y2)
    ret y3
  
  ```

  - 如果我是从 then 来的，就取 y1； 如果我是从 else 来的，就取 y2
  - φ **不是运行时判断**，而是**基于 CFG 的值选择**

### CFG（Control Flow Graph，控制流图）

**CFG 是一个有向图，用来描述程序中“执行路径的可能性”**

#### Basic Block

基本块：**单入口、单出口、内部无跳转的指令序列**

- 只能从第一条指令进入，从最后一条指令**（terminator）**离开
- 中间不能有分支、跳转

CFG描述的是**结构，而非值**

```
C代码：
if (x > 0)
    y = 1;
else
    y = 2;
    
CFG结构：
        [entry]
           |
      x > 0 ?
       /     \
   [then]   [else]
       \     /
        [merge]

```

- **CFG是优化的基础**，因为所有的优化都在问：
  - 这条路径**一定会执行吗**？
  - 这条代码**永远到达不了吗**？
  - 这个循环**执行多少次**？

- 另外，LLVM IR需要把“**隐含的控制结构”变成“显式跳转”**，才能做通用分析

### SSA 和 CFG如何结合

CFG 解决“程序怎么走”， SSA 解决“数据怎么流”

- **CFG决定φ 节点出现的位置**：只出现在有多个前驱的Basic Block
- **SSA的值选择依赖执行路径**

SSA和CFG在LLVM中可以满足如下需求：

| 需求     | CFG  | SSA  |
| -------- | ---- | ---- |
| 控制分析 | ✅    | ❌    |
| 数据分析 | ❌    | ✅    |
| 简化优化 | ⚠️    | ✅    |
| 可组合性 | ✅    | ✅    |



## What LLVM includes

### Some Concepts

- **语法糖**：只为了“让人更好写 / 更好读”，对程序语义没有本质影响的语法
- **side-effect**：除了“产生返回值”以外，对程序状态产生的影响
- **use-def**：描述一个 Value 在哪里被定义（def），又在哪里被使用（use）

LLVM IR的总体层级结构如下所示：

```
Module
 ├── Global Variables
 ├── Function
 │    ├── BasicBlock
 │    │     ├── Instruction
 │    │     └── Instruction
 │    └── BasicBlock
 └── Function

```

- LLVM IR就是一个 Module，内部是函数
- 函数由基本块构成
- 基本块由指令组成

### Module

**一个 Module 表示一个“编译单元（translation unit）”**

- 可以类比为一个`.c`文件

Module里往往包含：

- **函数定义 / 声明**

- **全局变量**

- **类型信息**

- **目标平台信息（DataLayout / TargetTriple）**

设计原因（不懂）

- 优化需要**跨函数**
- 链接前就能做分析
- 支持 LTO（Link Time Optimization）

### Function

LLVM IR的函数约等于源语言中的函数

其组成为：

- 函数签名：返回类型、参数列表
- **属性**：
  - readonly：函数只读取内存，不写内存
  - noalias：这个指针参数不会与其他指针指向同一块内存
  - nounwind：函数不会抛出异常
  - ......
- Basic Blocks

其硬约束为：

- 必须有**entry block**
  - entry block 是一个 Function 的“第一个 BasicBlock”，程序从这里开始执行
- 所有BasicBlock必须属于一个Fucntion
- Function 构成一个CFG

- **SSA不跨Function**
  - 两个function里可以有同名变量

### BasicBlock

BasicBlock 是**一段“顺序执行、无中间跳转”的指令序列**

- 单入口、单出口
- 只能在末尾跳转
- **Basic Block是CFG的节点**，**是控制流里的最小稳定单元**

### Instruction

- Instruction 是**LLVM IR的最小执行单位**

- **每一条 Instruction 本身就是一个 Value**

不同类别：

- **非终结指令（produces value）**

  ```
  %x = add i32 %a, %b
  %y = mul i32 %x, 2
  ```

  - 都有结果
  - 都是 SSA value

- **终结指令（terminator）**

  ```
  br label %next
  ret i32 %x
  ```

  - 不产生值
  - 决定 CFG 结构

Instruction 的关键属性：

- 有类型
- 有操作数
- 有use-def 关系
- 可能有side-effect



## MEM（内存模型）

### some concepts

- **alias**：两个（或多个）不同的指针，在运行时**可能指向同一块内存**

  - LLVM的alias结论一般是四值逻辑

  | 结果           | 含义                        |
  | -------------- | --------------------------- |
  | `NoAlias`      | 绝对不会指向同一内存        |
  | `MayAlias`     | 可能指向同一内存（最保守）  |
  | `MustAlias`    | 一定指向同一内存            |
  | `PartialAlias` | 部分重叠（如 struct field） |



内存（地址）不是SSA的，内存读出来的值是SSA的

- **内存 = 可变状态**
- **寄存器值（SSA Value）= 不可变**

因此LLVM 使用**显示内存指令**，分别表示SSA value和内存：

```
SSA Value 世界        内存世界
----------------    ----------------
%x = add ...        store i32 %x, i32* %p
%y = mul ...        %y = load i32, i32* %p

```



### alloca：栈上分配内存

```
%p = alloca i32
```

**在当前函数的栈帧里分配一块 i32 大小的内存**

- `%p` 是一个 **地址**
- 类型是 `i32*`

**特点**：

- 生命周期是**整个Function**
- 一般只出现在entry block

### store：往内存中写值

```
store i32 %v, i32* %p
```

- 把 SSA 值 `%v` 写入内存 `%p`

- **不产生返回值**

- 有 side-effect，体现在如下方面

  | 优化      | 是否受 store 影响        |
  | --------- | ------------------------ |
  | DCE       | 不能删“可能被读”的 store |
  | LICM      | 不能随便 hoist           |
  | Reorder   | 不能跨越 aliasing store  |
  | Vectorize | 内存依赖可能阻止         |

### load：从内存读值

```
%x = load i32, i32* %p
```

- 从 `%p` 读
- 读出来的 `%x`：
  - 是 SSA Value
  - 不可变



仅用上面三个命令，会有如下问题：

- 同一个变量：被反复load/store
- 编译器必须考虑
  - alias
  - 内存顺序

### mem2reg

其核心思想为：如果一块内存

- 只在函数内使用

- 没有被取地址逃逸

 那就把它“从内存提升到 SSA 寄存器

- mem2reg删除了alloca/load/store
- **引入了SSA Value 和 φ 节点**

例子，C语言为：

```c
int foo(int a) {
    int x = a;
    x = x + 1;
    return x;
}

```

引入mem2reg前为

```
%x = alloca i32
store i32 %a, i32* %x
%v1 = load i32, i32* %x
%v2 = add i32 %v1, 1
store i32 %v2, i32* %x
%ret = load i32, i32* %x

```

引入后为：

```
%x1 = %a
%x2 = add i32 %x1, 1
ret i32 %x2
```

#### mem2reg的进一步理解

- **mem2reg 不是一种变量**； **mem2reg 是一个 优化Pass，对 LLVM IR 的一种“变换”** 
- mem2reg 把“用 memory 表示的局部变量”提升为“SSA value + φ”
- LLVM IR 语言本身并不要求“必须 mem2reg “
  - 前端一定会生成 alloca + load/store，这是最简单、最保守、最正确的 lowering 方式
  - mem2reg 是“可选的 Pass”
    - 你 **可以不跑**
    - 在 `-O0` 时：
      - mem2reg **通常不跑**
    - 在 `-O1+`：
      - mem2reg **几乎一定跑**
- **LLVM IR 语言本身允许 memory-based 表达； 在优化流水线中，通常会尽早运行 mem2reg，使局部标量变量转为 SSA，从而方便后续优化**