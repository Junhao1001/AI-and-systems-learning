# System Optimization

# FlashAttention

## Concepts

- **GPU HBM（High Bandwidth Memory）**： HBM是一种贴近GPU的高带宽显存（off-chip DRAM)

  - 带宽极高、延迟较高、容量大
  - 是GPU的主存
  - 任何kernel都要从HBM中读数据

- **GPU内存体系内容**：

  ```
  HBM (off-chip DRAM)
    ↓
  L2 Cache (on-chip, shared by all SMs)
    ↓
  Shared Memory (per-SM, software managed)
    ↓
  Registers (per-thread, fastest)
  ```

  | 层级          | 位置       | 特点                   |
  | ------------- | ---------- | ---------------------- |
  | Registers     | SM 内      | 最快，最小             |
  | Shared Memory | SM 内      | 快，小，可控           |
  | L2 Cache      | GPU 芯片上 | 中等，自动             |
  | **HBM**       | 芯片外     | **慢，但容量和带宽大** |

- SRAM：主要指 GPU 上的 on-chip 存储， 包括register/shared memory

## 标准Attention

标准attenttion形式如下：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

1. 计算 $S = QK^T$
2. 把 **整个 $S$（大小是 $N \times N$**）写回 **HBM**
3. 对 $S$ 做 softmax（读 HBM）
4. 再乘 $V$（再读 HBM）

### 数据流：

- **Step 0: Q / K / V 初始都在 HBM**
- **Step 1: 计算 S = QKᵀ**
  - 启动kernel，每个thread block进行如下操作
  - **从 HBM 读取一部分 Q 和 K**
  - 放入 **register / shared memory**
  - 计算 dot product, 放回HBM
  - 此时S的尺寸是N*N，显存占用暴涨
- **Step 2: Softmax(S)**
  - 启动新Kernel，从**HBM读S**
  - 计算后，将softmax写回HBM
- **Step 3: O = softmax(S) · V**
  - 启动第3个kernel
  - 从 **HBM 读 softmax(S) 和 V**
  - 做矩阵乘法
  - 将输出写回HBM



整个流程有如下问题：

- 其中$QK^T$ 是 $O(N^2)$，对于长序列并不友好
- GPU的计算很快，但是瓶颈在于**读写显存（HBM）**
- **对cache的利用率较低**：数据计算完成后就丢弃，下次再从HBM读回

## FlashAttention

**核心思想：不把 $QK^T$ 和 softmax 结果写回 HBM，而是“边算边 softmax，边算边乘 V”**

### IO-aware(以访存为中心)

标准attention的问题在于：

- GPU HBM很慢
- SRAM很快但是很小

因此设计思想为：

- **Attention 拆成小块（Tiles）**
- **每一小块在 SRAM 里完成全部计算**
- **算完立刻丢，不写回 HBM**

具体步骤为：

- 将 Attention 拆成：

  - Q block：$Q_i$

  - K/V block：$K_j, V_j$

- 对每个 block pair：
  - 计算 $Q_i K_j^T$
  - 在 **SRAM 中做 softmax 的局部累积**
  - 立即乘 $V_j$
  - 累加到输出
- block并非是token, 而是指**一次能被加载进 SRAM（shared memory + registers）的一小段 K/V**

### Online Softmax(流式 softmax)

对每个 block：

- 维护：
  - 当前最大值 $m$
  - 当前归一化因子 $l = \sum e^{x - m}$
  - o：当前输出向量（已经加权过的）

当新 block 来时：
$$
m^{\prime}=\max(m,m_{new}) \\
l^{\prime}=l\cdot e^{m-m^{\prime}}+\sum e^{x_{new}-m^{\prime}}
$$
**同时更新输出**：
$$
o \leftarrow o \cdot \frac{l_{\text{old}}}{l_{\text{new}}}
     + \sum_j \frac{e^{x_j - m_{\text{new}}}}{l_{\text{new}}} V_j
$$

- 注意，这里的j求和并不是对历史所有的block j进行求和
- 对**当前 block 内的 j ∈ J_t** （第t个block内的index集合$J_t$）求和 （如果block里只有一个向量，这里也就不需要有求和符号）

### 收益

这种方法的代价是：

- **增加了一点计算**
- **大幅减少了内存访问**

- 在GPU上，是比较划算的trade-off

性能上：

- 通常 **2–4× speedup**

- 对长序列收益更明显

显存上：

- Attention 显存从：
  - **O(N²)** → **O(N)**
- 直接解锁：
  - 长上下文
  - 更大 batch size



# vLLM

vLLM 是一个面向大语言模型推理（Inference）的高性能服务引擎，**核心创新是用一种叫做 *PagedAttention* 的机制来高效管理 KV Cache，从而显著提升吞吐和显存利用率**

## Concepts

- **KV cache**: 在自回归推理时，把“已经算过的 Key / Value 存下来，下一个 token 生成时直接复用”，避免重复计算
- OOM：Out Of Memory，内存（通常是 GPU 显存）不够用了

## 传统框架的问题

- KV cache的大小为：`batch_size × seq_len × hidden_dim`

  - batch_size: 同一时间“一起送进模型算”的请求（或样本）数量

  - 假设：

    - batch_size = 4（4 个用户）
    - 每个用户已经生成 1000 token

    那么：

    - KV cache 里实际存的是：

      ```
      4 条序列 × 1000 个 token
      ```

传统框架的问题：

| 问题            | 本质原因                     |
| --------------- | ---------------------------- |
| 显存浪费严重    | KV cache 必须是 **连续内存** |
| 动态 batch 困难 | 不同请求 seq_len 不同        |
| 低吞吐          | padding + 同步执行           |
| 频繁 OOM        | 长上下文直接炸               |

## PagedAttention

借鉴了 **操作系统的虚拟内存分页（paging）** 思想

| OS 虚拟内存 | vLLM           |
| ----------- | -------------- |
| Page        | KV Block       |
| Page Table  | Block Table    |
| 虚拟地址    | Token 序列位置 |
| 物理页      | GPU 显存块     |

**vLLM：**

```
Block 0: [K0 K1 K2 K3]
Block 1: [K4 K5 K6 K7]
Block 2: [K8 K9 ...]
```

- 每个 block 是固定大小（例如 16 tokens）
- 不同请求共享一个 **KV Block Pool**

### Block Table

对于每个request：

```
token index  →  block id  →  GPU memory address
```

- **避免 padding**

- **允许不同长度请求共存**

- **支持动态增长 / 回收 KV cache**

**KV Block Pool**:

- KV Block Pool 通常是「预先分配的一大块连续显存」，然后在这块显存里做“子块管理”（block-level allocation）

**Block Table:**

- 在 vLLM 中, KV cache **不再连续**, 一个请求的 token 可能分布在不同block中

- block table **负责把「逻辑 token 序号」映射到「物理 KV block」**

- 每个request，都有一张表：

  ```
  BlockTable[request_id] = [
    block_id_for_tokens_0_to_15,
    block_id_for_tokens_16_to_31,
    block_id_for_tokens_32_to_47,
    ...
  ]
  ```

- 所以一个token访问KV的路径通常如下所示：

  ```
  token_index
     ↓
  logical_block_index = token_index / block_size
     ↓
  physical_block_id = BlockTable[logical_block_index]
     ↓
  GPU address = base_addr + physical_block_id * block_bytes
  
  ```

**Padding**: 为了让不同长度的序列“对齐成同样长度”，人为填充的无效 token

- 在 **传统 inference batching** 中：

  - 一个 batch 里的序列, 长度必须一致

    - 因为其KV cache的形态为：

      ```
      K: [batch_size, seq_len, num_heads, head_dim]
      V: [batch_size, seq_len, num_heads, head_dim]
      ```

    - seq_len是一个单一的维度，并不能动态调整

    - 张量框架（PyTorch / CUDA kernel）**不支持“ragged tensor”**

  - 于是短序列被 padding 到最长序列

  - 对 `<pad>` token：也要算 attention，也要占 KV cache； 造成了**算力浪费 + 显存浪费**

- 在vLLM中：

  - **KV cache 是按 token/block 分配的**
  - **不存在“对齐到最长序列”的要求**

  - 短序列：只分配需要的 block
  - 长序列：动态增长
  - padding 在 KV cache 层面被“彻底消灭”

### Attention计算

在attention计算时，会进行如下操作：

- Q：当前 token
- K/V：从 **多个离散 blocks** 中 gather
- Attention kernel 内部完成：
  - block-wise load
  - on-the-fly address translation
  - reduction

**Attention kernel 被“重新写过”**

## 调度相关

### Continuous Batching

传统 batching：

- 等一批请求 → 一起跑 → 等结束

vLLM：

- 新请求可以 **随时插入**
- 旧请求可以继续生成
- **token-level interleaving**

可以认为：**GPU 永远在干活**

### 请求生命周期管理

每个 request：

```
prefill → decode → finished
```

vLLM 调度器会：

- 优先让 GPU 满载
- 动态分配 KV blocks
- 回收已完成请求的 blocks





# Pipeline Parallelism(需要实践理解)

Pipeline Parallelism 是一种 **模型并行（Model Parallelism）** 的方式，用于在 **多块 GPU 或计算设备之间分摊模型的计算任务**

- 将模型按照 **层（Layer）** 或 **模块（Module）** 划分成多个连续的 **分段（Stage）**。
- 不同的 GPU 负责不同的 Stage。
- 数据（通常是一个 batch 的样本）像流水线一样 **从第一个 Stage 传递到最后一个 Stage**

## 基本原理

假设有一个模型有 12 层，训练 batch 为 1 个大 batch：

1. **切分模型：**
   - 4 张 GPU → 每张 GPU 负责 3 层
   - Stage0: 层1~3
   - Stage1: 层4~6
   - Stage2: 层7~9
   - Stage3: 层10~12
2. **切分数据：**
   - 大 batch 可以进一步切成 **micro-batch**（小批量）
   - 每个 micro-batch 依次进入流水线
3. **流水线执行：**
   - Timestep 1：GPU0 开始处理 micro-batch 1
   - Timestep 2：GPU0 处理 micro-batch 2，GPU1 处理 micro-batch 1
   - Timestep 3：GPU0 处理 micro-batch 3，GPU1 处理 micro-batch 2，GPU2 处理 micro-batch 1
   - …

这样可以让 GPU **几乎不空闲**，即使模型太大放不下一个 GPU，也能高效训练

## 部分挑战

**微批次调度（Micro-batch scheduling）**

- 流水线要保持饱满，否则会有 **bubble（空闲时间）**
- Bubble 会降低效率

**梯度计算与反向传播（Backpropagation）**

- 传统训练：正向 → 反向一次完成
- Pipeline：不同 micro-batch 的正向和反向交错，需要 **保存每个 micro-batch 的中间激活**（1F1B（One Forward One Backward）调度）
  - GPU 内部本质是 **SIMD（单指令多数据）并行**，适合一次执行大量矩阵运算
  - 如果你在 **同一个进程 / CUDA 流** 上顺序调用：
    - GPU **不会真正同时执行**，是 **串行** 执行，等第一个完成才执行第二个
  - 如果使用 **多个 CUDA 流（stream）**：
    - GPU 可以**重叠执行某些操作**，尤其是计算和通信（比如 memcpy 或 kernel 执行不同内核）。
    - 但是 **相同类型的矩阵运算核函数仍然是抢占式调度**，并不保证完全并行
    - 在1F1B调度下：一个 GPU 同时“**看似处理两个 batch**”，其实是 **交错**：
      - 当前 micro-batch 正向的部分 kernel 在运行
      - 前一个 micro-batch 的反向在 **不同 kernel 或不同 stream 上排队执行**

**激活存储开销（Memory）**

- 每个 Stage 需要缓存多个 micro-batch 的中间结果
- 激活 checkpointing 可以缓解

**通信开销（Communication）**

- Stage 之间要传递激活和梯度
- 如果网络带宽低，可能成为瓶颈

## 优势

**训练阶段**：

- 大量计算、显存消耗大
- 使用 **微批次 + 正向/反向交错** 来 **充分利用多卡 GPU**
- 核心目的是 **提高 GPU 利用率 / 加速训练**

**推理阶段**：

- **大批量推理（High Throughput Inference）**

  - 批量处理大量请求时（batch size 很大）

  - 可以用 Pipeline 将 batch 拆分 micro-batch，多卡协同处理

- **大模型推理**

  - 如果模型太大，单卡放不下
  - Pipeline 可以分布到多卡进行正向传播