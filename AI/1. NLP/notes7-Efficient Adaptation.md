# Efficient Adaptation

## Prompting

- Prompting 是通过“精心设计或学习输入提示”，在不（或几乎不）更新模型参数的情况下，引导大模型完成特定任务的方法
- 个人理解：**一种在推理阶段，通过设计输入文本来引导模型完成特定任务的方式；它本质上是一种特定的提问 / 指令表达方法**

```
预训练 → 表示 &语言建模
          ↓
Instruction Following（SFT）
          ↓
模型能把 prompt 当任务
          ↓
Zero-shot 能力显现
          ↓
In-context learning
          ↓
Few-shot 能力显现
```

**Discrete Prompting**：

- **zero-shot**
- **few-shot**

**Soft Prompting**：

Prompt 不再是文本，而是**可学习向量**：
$$
[x_{\text{prompt}}, x_1, x_2, \dots]
$$

- Prompt 向量维度 = embedding 维度
- 只训练 prompt

**Prefix Tuning（Attention 级别 Prompt）**：

- 在 **每一层 attention** 的 K/V 前加前缀
- 比输入层 prompt 表达力更强

## PEFT (Parameter-Efficient Fine-Tuning)

核心思想：在微调大模型时，只训练“极少量新增或选定参数”，而**冻结绝大多数原模型参数**

```
PEFT
├── Additive（加参数）
│   ├── Adapter
│   ├── LoRA
│   └── Prefix / Prompt Tuning
├── Selective（选参数）
│   ├── BitFit
│   └── Partial Fine-tuning
└── Re-parameterization（重参数化）
    └── LoRA（也属于）
```

### Adapters

- 核心思想：在冻结大模型参数的前提下，**在每一层网络中插入一个小型可训练模块，用极少参数实现对新任务的适配**
  - 插入（insert）
  - 小模块（bottleneck）
  - 主干冻结（freeze backbone）

- 在Transformer中的位置
  - **FFN 后（最经典）**
  - Attention 后
  - 两者都插（表达能力更强，但参数稍多）

```
Self-Attn → Add&Norm → Adapter
FFN       → Add&Norm → Adapter
```

- **Basic Architecture**:

  - 输入维度：`d`（模型隐藏维度）
  - Adapter bottleneck 维度：`r`（r ≪ d）
  - 数学形式：

$$
\text{Adapter}(h) = W_{\text{up}} \, \sigma(W_{\text{down}} h)
$$

  - 并通过 **残差连接** 加回原表示。

<img src="./images/notes7-Efficient Adaptation/image-20251223171857701.png" alt="image-20251223171857701" style="zoom:80%;" />

### Sparse Subnetworks （Pruning）

- **Core Idea**: 指在一个大模型中，只使用部分神经元或权重来执行任务，从而在保留大部分参数的前提下，实现高效训练或推理

- 假设原始模型权重矩阵为 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$

- **定义稀疏掩码**：

$$
M \in \{0,1\}^{d_{\text{out}} \times d_{\text{in}}}
$$

  - 1 表示激活权重
  - 0 表示冻结权重（不更新、不计算）

- **稀疏子网络计算**：

$$
\tilde{W} = W \odot M
$$

  - 只有 $\tilde{W}$ 的非零部分参与前向和反向传播
  - 梯度只流向激活部分

#### 不同实现方式：

- **Static Sparsity**：

  - 在训练前确定哪些权重激活（例如 LTH）

  - 优点：训练中稳定，推理效率高

  - 缺点：需要预先找到合适子网络

- **Dynamic Sparsity**：

  - 每轮训练/每步迭代选择活跃子网络

  - 典型算法：SET、RigL

  - 优点：灵活，可能找到更好子网络

  - 缺点：实现复杂，需要动态 mask

- **Mixture of Experts (MoE)** 

  - 每次只激活部分专家网络

  - 这也是 Sparse Subnetwork 的一种“任务级稀疏”

Pruning 和 Sparse Subnetwork的关系：

- Sparse Subnetwork = Task-aware Pruning + 子网络选择

#### Pruning的分类

**Magnitude-based Pruning**：

- 核心思想：权重越小，对输出影响越小 → 可以剪掉
- 具体步骤：
  1. 训练模型
  2. 对权重按绝对值排序
  3. 保留 top-k %，剪掉剩余
- 特点：简单、高效
- 典型应用：LTH（Lottery Ticket Hypothesis）

**Gradient / Sensitivity-based Pruning**：

- 根据权重对任务 loss 的敏感性选择保留/剪掉
- 方法：

$$
\text{score}(w_i) = \left|\frac{\partial L}{\partial w_i} \cdot w_i \right|
$$

- 优点：更任务相关
- 缺点：计算梯度信息较昂贵

**Structured Pruning**：

- 不剪单个权重，而剪**整个神经元、通道或注意力头**
- 优点：推理加速明显
- 缺点：对性能影响较大，需要精细调节



### LoRA (Low-Rank Adaptation)

#### 原理：

微调一个线性层：
$$
y = W x
$$

- 全量微调：更新整个 $W$
- **LoRA**：认为“任务差异只需要一个低秩的增量”

于是改成：
$$
y = (W + \Delta W) x
$$
并且 **强约束**：
$$
\Delta W = B A,\quad A \in \mathbb{R}^{r \times d},\; B \in \mathbb{R}^{d' \times r},\; r \ll d
$$
**只训练 A、B，冻结 W**

#### 初始化：

- A 随机初始化
  - 标准正态分布
  - Xavier / Kaiming （用在FFN层，希望与原线性层尺度匹配）

- $B$ 初始化为 0

#### 常见注入点：

- Attention 层
  - $W_q, W_k, W_v, W_o$

- FFN 层
  - $W_{up}, W_{down}$

- 经验规律：

  - **Attention LoRA → 更偏任务行为 / 对齐**

  - **FFN LoRA → 更偏知识 / 表达能力**

#### LoRA和Adapter的区别

- 经典 Adapter： 是一个**显式插入的网络模块**；改变了 forward graph，有非线性
- LoRA：**不插新层**，不改变原 forward 结构，是对原权重的“参数化重写”
- LoRA 在 PEFT 的**抽象层面上可以被视为一种 Adapter**，因为它通过引入少量可训练参数来适配冻结的主干模型

QLoRA：冻结的低比特量化主干模型（通常 4-bit） + 全精度 LoRA 微调 (后续学习Quantization)



### Prex-Tuning

核心思想：冻结模型参数，仅在每一层 Transformer 的 attention 中，**引入一小段可训练的“前缀向量（prefix）**”

标准 attention：
$$
\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$
在 每一层 attention 中，拼接一段**可训练的前缀 key / value**：
$$
K' = [K_{\text{prefix}}; K_{\text{input}}], \quad
V' = [V_{\text{prefix}}; V_{\text{input}}]
$$

- prefix 不来自输入 token
- 是 **直接学习的连续向量**
- 通常 **只作用在 K / V，不作用在 Q**

#### 和Adapter的区别

- Adapter、LoRA、Prefix 是同一“位置”的不同实现选择 它们都发生在 SFT / task adaptation 阶段

| 能力类型   | Adapter | Prefix   |
| ---------- | ------- | -------- |
| 行为对齐   | 中      | **强**   |
| 风格控制   | 中      | **很强** |
| 新知识注入 | **强**  | 弱       |
| 推理稳定性 | 高      | 较低     |

#### Others

- 因为 K / V 决定“能被关注什么”，而 Q 决定“我在找什么”；
  - Prefix 的目标是提供“可被模型检索的上下文记忆”，而不是改变输入 token 本身的语义
- Prefix-Tuning 本质上就是：**在 self-attention 中人为制造一个“可控的 cross-attention memory**

### Prompt Tuning

- prompt 从**离散 token**变成**连续向量**：

  ```
  [p1, p2, p3, ..., pk, x1, x2, ..., xn]
  ```

- 其中：

  - `pi`：**可学习的向量**
  - `xi`：真实输入 token embedding
  - **模型参数完全冻结**

- Prompt Tuning 就是：**学习这组 pi**

推理阶段：

- Prompt embedding 会作为 **上下文条件**，通过 self-attention 影响后续 token 的隐藏状态

- 输入文本：

  ```
  x = "Translate this sentence into French"
  ```

- 经过 tokenizer：

  ```
  [x1, x2, ..., xn]
  ```

- 查 embedding 表：
  $$
  E(x) = [e_1, e_2, ..., e_n], \quad e_i \in \mathbb{R}^d
  $$

- 拼接 Prompt + Input Embeddings（关键一步）

  ```
  [ p1, p2, ..., pm, e1, e2, ..., en ]
  ```

- 也就是：
  $$
  E'(x) = [P ; E(x)] \in \mathbb{R}^{(m+n)\times d}
  $$

- **没有新增 token，没有 tokenizer 变化**；prompt 不对应任何“真实词”

- 模型**不会区分**哪些 embedding 来自 prompt，哪些来自输入

  
