# Quantization

**Quantization = 用低精度数值（如 int8 / int4）来近似表示高精度数值（如 float32）**

- 更小的模型体积

- 更低的内存带宽

- 更快的推理速度

- 更低的功耗（尤其在移动端 / 边缘设备）

## Concepts

- **FP**: float point, 浮点数，一般由三部分组成：
  - Sign：正负号
  - Exponent：数值范围
  - Mantissa：精度

$$
x = (-1)^{sign} \times mantissa \times 2^{exponent}
$$



- **BF**：Brain Floating Point

| 格式 | Exponent | Mantissa |
| ---- | -------- | -------- |
| FP32 | 8        | 23       |
| BF16 | **8**    | **7**    |
| FP16 | 5        | 10       |

- BF和FP的差别：
  - 累加精度（accumulation precision）：数学算子一样，但**中间累加精度被“偷偷提高”了**（BF16 × BF16 → FP32 accumulate）
  - 硬件支持差异：GPU / TPU / NPU 对 BF16 和 FP16 的支持**不是对称的**
- **clipping/percentile**: 
  - 动机：在统计时发现，**绝大多数值很小**，**极少数 outlier 特别大**
  - clipping：裁剪，人为设定一个阈值，把超过阈值的数“截断”
  - $T$：clipping threshold
  - 超过的值直接变成 ±T

$$
x_{clip} = \text{clip}(x, -T, T)
$$

- **Percentile**: 
  - **Percentile(p)** = 一个阈值，使得  **p% 的数据落在这个值之内**
  - 通常为99%或99.99%

## 流程位置

并非是之前提到的训练周期（如pre-training / SFT / RLHF）里，而是在**训练完成之后、推理之前**

```
Pre-training
   ↓
SFT / Instruction tuning
   ↓
Alignment (RLHF / DPO / IPO)
   ↓
Final FP16 / BF16 Model
   ↓
Quantization
   ↓
Inference / Deployment
```



## 基本数学形式

最常见的是 **Uniform Quantization**：
$$
x_{int} = \text{round}\left(\frac{x_{fp}}{s}\right) + z
$$

- `s`：scale（缩放因子）
- `z`：zero point（零点）
- `x_int`：int8 / int4
- `x_fp`：float32 / float16

## 具体步骤

1. 拿到一个训练完成的 FP 模型（FP32/FP16/BF16）
2. 选择量化策略
   - 量化哪些部分:
     - 权重/激活/KV Cache
   - 用什么精度：
     - INT8/INT4/FP8
   - 量化粒度：
     - Per-tensor：整个 tensor 用一个 scale；快；精度较差
     - Per-channel：每个 output channel 一个 scale；精度明显更好；推理稍慢

3. **Calibration（校准 / 统计分布）**: 对模型跑 **少量样本**（几十～几百）：
   - 收集权重 / 激活的 **数值分布**
   - 记录 min / max / percentile / histogram
4. 计算 Quantization 参数（scale / zero-point）:
   - symmetric（z = 0）/ asymmetric（z ≠ 0）/ clipping / percentile
   - **通过统计得到的分布范围和选用的量化大小，来确定scale**
5. 执行量化：

$$
W_q = \text{round}\left(\frac{W}{s}\right)
$$

并存储：

- `W_q`（int）
- `s`（fp16 / fp32）

激活量化：推理时动态计算 activation scale；或使用静态 scale

推理时计算形式变化：**数值语义保持，表示方式改变**
$$
y = s \cdot (W_q x)
$$

6. **精度验证 & 推理评估**

- 验证内容：

  - Perplexity

  - 下游任务 accuracy

  - 推理速度 / 内存 / 延迟

- 满足需求才能部署

**Quantization 是一种推理阶段的数值近似技术，其目标是在不改变模型计算语义的前提下，通过低精度表示与 scale 机制，尽量还原原始 FP 模型的推理结果**







# Distillation

概括：用一个**大模型（Teacher）** 的行为，来训练一个**小模型（Student）**， 让 Student 在**参数量更小、计算更快**的情况下，尽量接近 Teacher 的性能

Distillation 的目标：

- **性能 **：小模型学到大模型的泛化能力
- **成本 **：推理快、内存小
- **部署友好**：适合移动端、边缘设备

## 流程位置

在LLM中，Distiallation Stage往往**不是一个单独的stage**; 而是融入到：

- SFT数据构造
- 监督信号设计
- 目标函数中

```
(1) Pre-training
     ↓
(2) Post-training
     ├─ SFT（Instruction / Chat）
     ├─ Preference Alignment（RLHF / DPO / IPO）
     └─ Distillation（数据 / 行为层面）
     ↓
(3) Compression & Deployment
     ├─ Quantization
     ├─ Pruning
     └─ Compiler / Kernel
```

## 训练方式

1. Teacher先进行训练：
   - 大模型、性能强、不更新参数
2. Student 学 Teacher

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{KD} + (1-\alpha) \cdot \mathcal{L}_{CE}
$$

- **KD Loss**：Student vs Teacher
- **CE Loss**：Student vs Ground Truth

### 数据蒸馏

```
Pre-trained LM
   ↓
[Teacher 生成指令 / 对话 / 推理数据]
   ↓
SFT（Student）

```

- 用大模型生成：
  - Instruction-response
  - Chain-of-Thought
  - 多轮对话
  - 并作为**SFT数据**
- 小模型一般是重新训练的模型，而不是大模型的一部分
- 小模型的“多任务泛化能力会弱一些”

| 能力         | 大模型 | 小模型     |
| ------------ | ------ | ---------- |
| Seen tasks   | 强     | 强         |
| Unseen tasks | 强     | 较弱       |
| 复杂推理     | 强     | 明显下降   |
| 工具调用     | 强     | 部分可保留 |

#### 为什么Teacher生成的SFT数据更便宜、更有效

- **数据可控**：Teacher 可以：

  - 统一格式（chat / role）
  - 控制长度
  - 控制难度
  - 控制覆盖任务类型

- 隐式知识被显示化：Teacher输出的是

  - 已对齐的行为

  - 已学会的推理路径

    - Teacher可以输出**chain-of-Thought**，让SFT学习
    - 推理路径被当做token序列

    ```
    Question: ...
    Let's think step by step:
    1. ...
    2. ...
    3. ...
    Answer: ...
    
    ```

    - **Structured Reasoning / Tool Trace**：更为工程化的模式
    - **行为蒸馏（policy distillation）**，还学习了决策模式

    ```
    <analysis>
    Step 1: ...
    Step 2: ...
    </analysis>
    <final>
    Answer
    </final>
    
    ```

    - 隐式行为蒸馏：即使不写推理，Student仍能学习到一部分回答结构、语言风格

  - 规范的语言风格

  - Student学的是压缩过的知识

- 真实数据常有模糊、歧义、标签噪声；大模型生成的数据语义清晰、推理步骤明确

- 总体来说人工真实数据的成本更高、质量更难控制

- **但是Teacher推理成本高，蒸馏等于用一次较为贵的推理，换多次便宜的推理**

### Logits / Sequence 蒸馏

#### Logits 蒸馏

**让 Student 去拟合 Teacher 在每一个 token 位置的 概率分布（logits / softmax）**
$$
\mathcal{L}_{KD} = \mathrm{KL}(
\text{softmax}(z_t / T)
\;\|\;
\text{softmax}(z_s / T)
)
$$

- $z_t$：Teacher logits
- $z_s$：Student logits
- $T$：Temperature（温度）



**普通的CE Loss(Cross-Entropy loss):**
$$
\mathcal{L}_{CE}
= - \sum_i y(i)\log p_s(i)
$$
这里的**y(i)是one-hot的**，它能告诉哪个是“对的”，却不能说明哪些是差点“对的”

因此引入**Temperature T**：
$$
p_s^{(T)}(i)
= \frac{e^{z_{s,i}/T}}{\sum_j e^{z_{s,j}/T}}
$$

- $T > 1$：

  - 拉平分布
  - 放大“非最大类”的信息

- $T \to \infty$：接近 uniform

- $T = 1$：普通 softmax

  

**KD Loss(KL Divergence )**:

定义:
$$
\mathcal{L}_{KD}
= \mathrm{KL}(p_t^{(T)} \;\|\; p_s^{(T)})
$$
展开：
$$
= \sum_i p_t^{(T)}(i)
\log \frac{p_t^{(T)}(i)}{p_s^{(T)}(i)}
$$
忽略与 Student 无关的常数项：
$$
\mathcal{L}_{KD}
\equiv - \sum_i p_t^{(T)}(i)\log p_s^{(T)}(i)
$$
**总结来说：在 Logits Distillation 中，Student 的目标函数是一个联合目标：一部分约束其对真实 one-hot 标签的拟合，保证正确性；另一部分约束其对 Teacher 下一个 token 概率分布的拟合，从而学习到更平滑、更信息密集的决策边界**

#### Sequence Distillation

Teacher 直接生成**完整输出序列**，Student 把它当作“ground truth”来学

```
x → Teacher → ŷ
(x, ŷ) → Student
```

和数据蒸馏有一定的重合点

数据蒸馏仅是关心数据的来源，Teacher可以不输出序列，只输出label等；

Sequence 蒸馏更关心**监督信号的形式（loss）**，将teacher输出的序列当做ground truth来拟合。

一般来说：数据蒸馏包含Sequence蒸馏



### Task-specific Student

**Task-specific Student 是一个“只为某一类（或少数几类）任务而设计和训练的小模型”**

和前面数据蒸馏中提到的**通用 LLM student**的区别如下，其更像是一个专用工具：

| 维度       | Task-specific Student | 通用小 LLM Student |
| ---------- | --------------------- | ------------------ |
| 任务范围   | 单一或少数            | 多任务             |
| 输入形式   | 固定 schema           | Instruction / Chat |
| 输出形式   | Label / span / score  | 自由文本           |
| 推理方式   | 非生成式（多）        | 自回归生成         |
| 延迟       | 极低                  | 较高               |
| 部署       | 工业级稳定            | 灵活但重           |
| 是否像 LLM | ❌                     | ✅                  |

- 一般用于分类、QA、排序、匹配

#### 和SFT的区别

两者似乎都是为了“特定任务”微调：

- **SFT让同一个模型学会某个任务**
  - 其监督信号往往为人工标注数据、为真实ground-truth
- **Task-specific Student 是“训练一个专门为该任务设计/选择的小模型，用大模型当老师来教”**
  - 其监督信号为Teacher LLM的输出
  - 损失函数可以是logits、sequence
