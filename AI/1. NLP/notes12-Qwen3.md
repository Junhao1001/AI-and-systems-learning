# Qwen3

## 0. 本地环境配置

### git clone transformers

```
git clone https://github.com/huggingface/transformers.git
```

### PyCharm创建项目

- 打开pycharm，open之前clone的`transformer`目录
- 创建虚拟环境：
  - 右下角打开Python interpreter
  - 点击 **Add Interpreter**
  - 选择 **Virtualenv Environment**
  - 选择：Base interpreter：`Python 3.10`

- 安装成功后：
  - 右下角显示：`Python 3.10 (venv)`
  - `transformers/venv/` 目录出现
  - PyCharm Terminal 自动激活 venv

### 用 PyCharm Terminal 安装依赖

- 安装 PyTorch（CPU 版本即可）

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- Editable 模式安装 Transformers（🔥关键）

```
pip install -e .
```

这一步非常重要，它意味着：`import transformers` 用的是 **你正在编辑的源码**

- 安装 Qwen3 相关依赖

```
pip install accelerate sentencepiece safetensors einops
```

### 验证 transformers 可运行

- 根目录创建`run_test_qwen.py`文件

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

print("transformers import ok")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True
)

print("Qwen3 loaded")

```

- 运行后有对应输出



## 0.5 Some Concepts

### HF (Hugging Face)

**Hugging Face Transformers 提供的一整套「模型工程规范」**

HF定义了：

- 模型应该长什么样
- forward 应该接收什么参数
- generate 怎么统一调用不同模型
- 输出格式（dataclass）

### helper function

**helperr function: 不是模型本体，但让模型更好写 / 更好用的小函数**

其不是一个结构，而是一些**工具**

模型中常用的**helper函数**：

- 构造 attention mask
- 处理 KV cache
- reshape / expand tensor
- 处理 rotary embedding 的 index

### Pytorch层 （nn.Module）?

pytorch层是**一个带参数+可向前计算的函数**

数学角度：

```
y = f(x; θ)
```

在Qwen3 中，有如下pytorch层：

| PyTorch 层          | 数学意义           |
| ------------------- | ------------------ |
| `Qwen3Attention`    | Attention 映射     |
| `Qwen3MLP`          | FFN                |
| `RMSNorm`           | 归一化             |
| `Qwen3DecoderLayer` | 一整层 Transformer |

### 不带任务头的纯语言模型

其作用是：**把 token 序列 → 映射为“上下文语义表示”**

```
input_ids
  ↓
embedding
  ↓
Transformer layers
  ↓
hidden_states

```

- 它**不会计算词表概率和选下一个token**
- 它**只输出表示**

### lm_head

用于将**表示转换为词表概率**

作用如下：

```
hidden_states
  ↓
lm_head
  ↓
logits
  ↓
softmax
  ↓
token probability
```

lm_head **不是 Transformer 的一部分**

它是一个 **任务头（task head）**

同一个 backbone：

- 可以接 LM head
- 也可以接 classification head

### Others

- **past_key_values**: Attention 中缓存的 Key / Value
- **hidden_states**: 每个 token 对应的“语义向量表示”



## 1. 整体结构分析

`modeling_qwen3.py`大致可以分为7个模块：

```markdown
1. imports + 工具函数
2. 辅助小模块（Norm / MLP / Rotary）
3. Attention 实现
4. Decoder Layer
5. Backbone Model（Qwen3Model）
6. Task Head（Qwen3ForCausalLM）
7. HF 注册 & 文档相关代码
```

### 1.1 imports + 通用工具

- 常用工具，如：`torch`,`nn`,`F`
- HF 的：
  - `PreTrainedModel`
  - `BaseModelOutputWithPast`
  - `CausalLMOutputWithPast`
- 一些 helper 函数（mask / cache）

### 1.2 基础组件(Building Blocks)

- 将论文里的数学模块变成PyTorch层
- 定义了**Transformer的一些小模块**，可能会被Attention/ DecoderLayer 调用
  - `RMSNorm`
  - `Qwen3MLP`
  - Rotary Embedding 相关函数

### 1.3 Qwen3Attention

该模块负责：

- Q / K / V 投影
- RoPE（旋转位置编码）
- GQA / MQA
- KV Cache（past_key_values）
- causal mask

### 1.4 Qwen3DecoderLayer

实现**一层标准decoder block:**

其基本结构为：

```
x
 ├─ RMSNorm
 ├─ Attention
 ├─ Residual
 ├─ RMSNorm
 ├─ MLP
 └─ Residual
```

- 需要关注各层的顺序
- residual 如何进行增加
- 如何插入attention / nlp

### 1.5 Qwen3Model (Backbone)

依次处理如下事情：

1. embedding input_ids
2. 依次跑 N 层 `Qwen3DecoderLayer`
3. 管理：
   - attention_mask
   - position_ids
   - past_key_values
4. 最后做一个 norm

`Qwen3Model` 约为**“不带任务头的纯语言模型”**，其输出为

- hidden_states
- past_key_values

### 1.6 Qwen3ForCausalLM

代码里调用得到是：

```
AutoModelForCausalLM → Qwen3ForCausalLM
```

其会执行如下命令：

- 调用 `Qwen3Model.forward`
- 接一个 `lm_head`
- 计算 logits / loss

### 1.7 HF glue代码

包括：

- `_CONFIG_FOR_DOC`
- `@add_start_docstrings`
- `register_for_auto_class`

**作用**：

- 文档
- AutoModel 识别
- HuggingFace 生态兼容

### 整体调用链

从`generate()`接口开始，模型的调用链为：

```scss
model.generate()
  ↓
GenerationMixin
  ↓
Qwen3ForCausalLM.forward()
  ↓
Qwen3Model.forward()
  ↓
for layer in layers:
      Qwen3DecoderLayer.forward()
          ↓
          Qwen3Attention.forward()
          Qwen3MLP.forward()
```



## 2. Qwen3ForCausal

### 2.1 初始化

```python
self.model = Qwen3Model(config)
self.vocab_size = config.vocab_size
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# Initialize weights and apply final processing
self.post_init()
```

- `self.qwen3`：Decoder-only Transformer的主题

- `self.lm_head`: 任务头，将hidden_states 映射到词表

- `self.post_init()`: **HuggingFace `PreTrainedModel` 统一的“模型初始化收尾钩子”**

  - 权重初始化
  - 权重tying （之后再来理解含义，这里先不深入）
  - 注册 gradient checkpoint/ flash attention 等后处理逻辑

- `post_init()` 定义在 **`PreTrainedModel`** 里。

  ```
  Qwen3ForCausalLM
   └── Qwen3PreTrainedModel
       └── PreTrainedModel
  ```

  

### 2.2 调用backbone

- 调用model(Qwen3Model)

```python
outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
```

- 输入：

  - `input_ids` ： token id

    - 这一轮 forward 要处理的所有 token
    - 长度不定，第一轮是prompt的所有token
    - 后续就是新生成的一个token

  - `attention_mask` ：padding / causal mask

    - 避免attention 看见padding
    - 1为可见，0为padding/mask

  - `position_ids`：token的位置编号，用于**RoPE / rotary embedding**

  - `past_key_values` → KV cache，用于加速生成

    ```python
    past_key_values = Tuple[
        layer_0(k, v),
        layer_1(k, v),
        ...
    ]
    ```

    - key: `[batch, heads, past_len, head_dim]`
    - value: 同上

  - `input_embeds`: 可以绕过embedding lookup,一般不和`input_ids`同时上传

    - 在多模态模型中，可能有的embedding没有token id
    - Prompt tuning / Soft prompt, 提前处理了token embeddings

  - `use_cache`: bool，用于确认是否返回`past_key_values`

    - 推理时: True    训练时:  False

  - `cache_position`: qwen3中较新的，**显示告诉模型当前 token 在“全序列中的绝对位置”**

    - 常用于静态 KV cache
    - Flashattention v2等
    - Long context

- 输出：

  - `hidden_states` → 每个 token 的表示
  - `past_key_values` → 更新后的 KV cache

### 2.3 hidden_states 到 logits

- logits：每个 token 对词表的“打分”，还没 softmax

```python
    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
```

- 进行一个线性映射，从`hidden_dim → vocab_size`

- `slice_indices`: 算力优化

  - 假设输入长度 = 4096，但你只想预测最后一个 token (一般在推理时)
  - 如果直接

  ```
  logits = lm_head(hidden_states)
  会得到
  [batch, 4096, vocab]
  ```

  - 通常

  ```
  slice_indices = [-1]
  ```

  - 只保留最后一个token的 hidden state
  - **在训练时，仍然需要计算所有的logits，用于loss计算**

### 2.4 计算loss

```python
loss = None
if labels is not None:
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
```

- 训练阶段提供 labels：
  - logits 会和 labels 对齐
  - 计算标准 **cross-entropy loss**
- 在推理阶段：
  - `labels=None`
  - 所以不会计算 loss

### 2.5 输出结构

```python
return CausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=outputs.past_key_values,
    hidden_states=outputs.hidden_states,
    attentions=outputs.attentions,
)
```

- HF 使用统一 dataclass 来封装输出

- 包含：

  - `logits`: 预测概率前的向量

  - `past_key_values`: KV cache

  - `hidden_states`: 中间表示

  - `loss` : 训练损失（optional）

  - `attention`: 每一层的attention map, 默认是none

    ```python
    attentions[layer] =
        [batch_size, num_heads, tgt_len, src_len]
    ```

    

## 3. Qwen3Model

- `Qwen3Model`就是**纯Transformer主干**，其只负责**输出hidden states**
- Qwen3Model的整体结构如下所示：

```
Qwen3Model
│
├── Embedding
│   ├── token embedding
│   └── rotary position embedding（RoPE）
│
├── N × DecoderLayer
│   ├── Self-Attention
│   ├── MLP / FFN
│   └── RMSNorm
│
├── Final RMSNorm
│
├── KV Cache 管理
│
└── Forward 输出组织
```

### 3.1 init

```python
def __init__(self, config: Qwen3Config):
    super().__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
        [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
    )
    self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = Qwen3RotaryEmbedding(config=config)
    self.gradient_checkpointing = False
    self.has_sliding_layers = "sliding_attention" in self.config.layer_types

    # Initialize weights and apply final processing
    self.post_init()
```

- `self.padding_idx`: padding_idx 并不是整个序列的 mask，而只是一个 **单一 token id**

  - 当 `input_ids == padding_idx`：
    - embedding 输出 **全 0 向量**
    - 该向量不会被梯度更新

- `self.embed_tokens`：

  - PyTorch 会自动**创建 weight 张量**：

    ```
    shape = [vocab_size, hidden_size]
    ```

    - 每个 token id 对应一行
    - 初始值通常是 **均匀/正态分布随机数**（训练前没有语义）

- `self.layers`：获取L层Transformer Decoder

- `self.norm`: Norm层，让hidden_states在送入lm_head前稳定；这里使用的是**RMSNorm**

- `self.rotary_emb`: 位置编码相关

- `self.gradient_checkpointing`: 一个**训练期专用、和推理无关**的机制, 用“多算一次”换“少存激活值”

  - 不开时，`forward`保存每一层的hidden states；`backward`直接使用保存的值
  - 开时，`forward`不保存中间激活，`backward`重新计算一次

- `self.sliding_layers`: **Sliding Window Attention**思想

  - 当context很长时，全部考虑会导致显存爆炸
  - Qwen3的策略是只使用**后M层 sliding window**

  ```
  sliding_layers = 指定使用 sliding window attention 的层索引
  ```

  - 每个 token 只 attend 一个固定窗口（如最近 4k）
  - KV cache 可丢弃更早部分

### 3.2 输入处理

**获取输入token**:

```python
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
```

- 将token_id映射为对应的embeddings
- 或者直接使用 `input_embeds`

### 3.3 Attention Mask & Cache初始化

**初始化cache**:

```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)
```

- `use_cache=True` 表示在生成时要保存过去的 key/value
- `past_key_values=None` 表示第一次调用 forward（没有缓存）

**token位置索引**

```python
if cache_position is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
```

- **计算每个 token 的全局位置索引**，用于 RoPE（Rotary Position Embedding）或 KV cache
- 每次加上当前batch的长度

**position_id初始化**

```python
if position_ids is None:
    position_ids = cache_position.unsqueeze(0)
```

- 初始化position_ids
- 增加一个batch维度

**attention_mask 处理**

```python
if not isinstance(causal_mask_mapping := attention_mask, dict):
    # Prepare mask arguments
    mask_kwargs = {
        "config": self.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
    "full_attention": create_causal_mask(**mask_kwargs),
}
```

- **判断 attention_mask 是否已经被预处理**
- 没有的话在这里进行初始化

**sliding layers初始化**：

```python
# The sliding window alternating layers are not always activated depending on the config
if self.has_sliding_layers:
    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
```

### 3.4 位置编码

- Qwen3使用的是**Rotary Positional Embedding（RoPE）**
- 在attention中对Q/K做旋转

```python
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

### 3.5 Decoder Layer循环

```python
for decoder_layer in self.layers[: self.config.num_hidden_layers]:
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
```

具体变化过程就是：

```
h₀ → layer₀ → h₁ → layer₁ → ... → h_L
```

内部操作在Decoder层进行解析

### 3.6 Final Norm & Output

```pytHon
hidden_states = self.norm(hidden_states)
return BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=past_key_values if use_cache else None,
)
```

- 最后进行一层norm，将Transformer 输出 → 稳定化

- 返回的输出**最核心的就是last_hidden_state**
- 其他都可选



## 4. Qwen3DecoderLayer

前面提到多次，Qwen3使用的Decoder Transformer Block架构如下所示：

- 是一个**Pre-norm Transformer**

```
input hidden_states
        │
        ▼
   RMSNorm (pre-norm)
        │
        ▼
 Self-Attention (+ KV cache)
        │
        ▼
 residual add
        │
        ▼
   RMSNorm (pre-norm)
        │
        ▼
        FFN
        │
        ▼
 residual add
        │
        ▼
 output hidden_states
```

### 4.1 Decoder的输入

```python
    (self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Unpack[TransformersKwargs],)
```

- 大多数前面都解释过，此处不再进行复述

### 4.2 整体分析

- **保存residual**

  ```
  residual = hidden_states
  ```

  - 后面attention的输出会加回这个值
  - 保证Transformer的核心稳定性

- **第一层RMSNorm(Attention前)**

  ```python
  hidden_states = self.input_layernorm(hidden_states)
  ```

  - 归一化 hidden_states 的 **幅度**
  - RMSNorm不减均值（比 LayerNorm 更快）

- **Self-Attention（核心）**

  ```python
  hidden_states, _ = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=past_key_values,
      use_cache=use_cache,
      cache_position=cache_position,
   	position_embeddings=position_embeddings,
      **kwargs,
  )
  ```

  - 主要完成了如下工作

    - 计算Q/K/V
    - QK^T → mask → softmax → × V
    - RoPE（使用 position_ids）
    - KV cache 追加 / 读取
    - GQA / MQA 头共享
    - 下一章节分析

  - 输出为如下形式

    ```
    attn_output, attn_weights, present_kv
    ```

    - `attn_output`即更新的hidden_states
    - `attn_weights` 默认不保存
    - `present_kv` 用于 cache

- **Attention + Residual**

  ```
  hidden_states = residual + attn_output
  ```

- **FFN**

  ```python
      residual = hidden_states
      hidden_states = self.post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states
  ```

  - 第二次残差准备
  - 在FFN前，再进行一次Norm
    - 防止FFN放大数值
    - 保持梯度稳定
  - FFN层：**token内部的变换**
    - 引入非线性变换
    - 通道维度扩展（通常 4× hidden_size）（Linear1先扩展，Linear2再回缩）
  - FFN + Residual：完成decoder layer

## 5. Qwen3Attention

主要完成**Q / K / V 投影，GQA / MQA，RoPE，KV cache 结构**工作

整体架构为：

```
hidden_states
   │
   ├──> Q/K/V 投影 (GQA / MQA)
   │
   ├──> RoPE 应用 position_ids
   │
   ├──> 拼接 past_key_values / 更新 cache
   │
   ├──> Attention Scores = Q·K^T / sqrt(d_k)
   │
   ├──> Apply causal_mask / sliding_mask
   │
   ├──> softmax → attention_probs
   │
   ├──> Context = attention_probs × V
   │
   └──> Output projection → hidden_states
```

核心输入为：

```
hidden_states         # [B, T, H]  # 输入的 hidden_states
attention_mask        # causal / sliding mask
position_ids          # [B, T]    # 用于 RoPE
past_key_value        # KV cache
use_cache             # bool
cache_position        # token positions
```

### 5.1 Q/K/V投影

```pytHon
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

- `.view()` 本质是 **reshape**，不改变数据，只改变张量的维度
- 这里的作用：
  - `[B, T, H] → [B, T, num_heads, head_dim]`
  - 然后 transpose → `[B, num_heads, T, head_dim]`
- 方便每个head独立计算注意力
- 为什么q/k需要norm：
  - Q 和 K 的缩放有利于数值稳定和 softmax 的梯度
  - RoPE 位置编码需要在 Q/K 上施加旋转和归一化
  - V 不需要做这些，因为 V **本质上只参与加权求和**，不会进入 softmax

- Decoder层里的norm：整个 token 向量尺度统一，为 Attention / FFN 做准备

- Q/K的norm:

  ```python
  q = q * (1 / sqrt(head_dim))  # 或者使用 RoPE 归一化
  k = k * (1 / sqrt(head_dim))
  ```

  - **缩放 dot-product**：`softmax(Q·K^T / sqrt(d_k))`
  - 避免 dot-product 数值过大或过小，保证 softmax 不饱和
  - 与 RoPE 一起使用时，保证旋转位置编码不会放大数值
  - V 不需要做这些，因为 V **本质上只参与加权求和**，不会进入 softmax

### 5.2 RoPE (旋转位置编码)

#### 数学本质（不深入）

- 将向量当做“二维复数对”

```
[x0, x1,  x2, x3,  x4, x5,  x6, x7]
 ↓   ↓    ↓   ↓    ↓   ↓    ↓   ↓
(0,1)   (2,3)   (4,5)   (6,7)
```

- 位置编码：在二维平面中对每个维度对进行旋转：

- 对于 **位置 p**，在第 i 个维度对上，旋转角度是：

$$
\theta_{p,i} = p \cdot \omega_i
$$

- 其中：

$$
\omega_i = 10000^{-2i/d}
$$

- 最终编码形式为：

$$
\text{RoPE}(x, p) =
\begin{pmatrix}
x_0 \cos \theta_0 - x_1 \sin \theta_0 \\
x_0 \sin \theta_0 + x_1 \cos \theta_0 \\
x_2 \cos \theta_1 - x_3 \sin \theta_1 \\
x_2 \sin \theta_1 + x_3 \sin \theta_1 \\
\vdots
\end{pmatrix}
$$

- **为什么这么编码**？

  - **内积天然依赖 p − q，而不是 p 或 q 本身**，能编码相对位置
  - **Attention score**：

  $$
  \langle \text{RoPE}(q_i, i), \text{RoPE}(k_j, j) \rangle
  $$

  - can be derived as:

  $$
  = \langle q_i, R_{j-i} k_j \rangle
  $$




#### 代码分析

```python
   cos, sin = position_embeddings
   query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

- `position_embeddings`是在Qwen3Model层已经转换好了

  ```python
  position_embeddings = self.rotary_emb(hidden_states, position_ids)
  ```

- 对应的维度变换为

  ```
  position_ids        [B, T]
  cos, sin ∈ [B, T, D/2]
  ```

- `apply_rotary_pos_emb`执行旋转编码

### 5.3 拼接past_key_values更新cache

```pytHon
if past_key_values is not None:
    # sin and cos are specific to RoPE models; cache_position needed for the static cache
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

```

- cache_kwargs记录当前K/V在序列中的位置，用于后续RoPE/attention mask对齐
- 将历史 key/value 拼接，避免重复计算 **(KV cache 只在「同一个 batch 内、同一条序列上」累积)**
- 更新了cache，并返回attention所需的K/V
- 当前 token 的 Q **只和“同一 batch、同一序列的过去 tokens 的 KV”做 attention**

### 5.4 attention score计算

```python
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
```

- 指定“用哪一种attention实现函数”
- `eager_attention_forward`选择朴素的attention实现

`eager_attention_forrward`的实现：

```python
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

- `repeat_kv`：解决 GQA / MQA 的 head 数不一致
  - 把K/V的head 复制扩展到与Q head数一致
- QKᵀ + scaling（attention logits）
  - **外层Q/K norm： 控制每个向量的模长**
  - **scaling：控制点积随维度增长的统计尺度**
- 加 causal / padding mask
  - 允许位置：0
  - 禁止位置：-∞（或极小负数）
  - 确保：不能看 未来/padding/sliding window之外
- softmax
- dropout：
  - 防止强依赖某个token,导致泛化能力变差
- attention × V：获取attn_outputs
- 输出：`attn_output, attn_weights`
  - `attn_output`：真正送入后续 FFN / residual 的结果
  - `attn_weights`：**仅 eager 返回**（flash 通常不返回）（推理时不需要）

### 5.5 拼接多头并映射

```python
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
```

- `reshape`拼接多头信息，维度变换回`[B,T,hidden_size]`
- `.contiguous`: 保证内存布局连续
- `output projection`：输出投影
  - 融合不同 head 的信息
  - 学习 head 之间的组合方式
  - 把 multi-head attention 结果重新映射回模型空间

## 6. 功能模块

### 6.1 RMSNorm

#### 数学定义

**普通LayerNorm**:
$$
\text{LN}(x)_i =
\gamma_i \cdot
\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
+ \beta_i
$$

**RMSNorm**:

给定：
$$
x = (x_1, x_2, \dots, x_d)
$$
RMS（均方根）：
$$
\text{RMS}(x) =
\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
$$
RMSNorm 输出：
$$
\text{RMSNorm}(x)_i =
\gamma_i \cdot
\frac{x_i}{\text{RMS}(x) + \epsilon}
$$
**Why RMSNorm**：

- 不减均值：
  - 减均值会改变向量方向
  - 在 Transformer 中：**方向**（token 表示的语义方向）比**绝对中心** 更重要
- 数值更稳定：RMS对异常值不那么敏感
- 省算力：少一次`mean`、少一次减法，少一个bias参数

**$\gamma 和 \epsilon$ 什么作用**？



#### 代码简析

```python
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```

- `self.weight`: 即 $\gamma 参数$
  - 用于缩放，重新引入**尺度**
  - 如果没有，RMS=1，能量恒定；
  - $\gamma$ 是一个训练得到的值
- `self.variance_epsilon`: 
  - $\epsilon$ 防止除0
- `hidden_states.to(torch.float32)`:
  - RMS / sqrt 在 fp16 下不稳定
  - **softmax / norm 一律用 fp32 算**
- `variance`： 均方根
- 后续归一化计算、scale并恢复input_type
- ` extra_repr`：看起来是个打印，调试用

### 6.2 Qwen3MLP

- Qwen3MLP一个**现代 LLM 标准的 Gated MLP**

```
x
├── gate_proj ──> act_fn ──┐
│                          × ──> down_proj ──> output
└── up_proj  ─────────────┘
```

代码实现为：

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

**数学形式**：

设：

- $x \in \mathbb{R}^{d}$
- $W_g, W_u \in \mathbb{R}^{d \times d_{\text{ff}}}$
- $W_d \in \mathbb{R}^{d_{\text{ff}} \times d}$

则：
$$
\begin{aligned}
g &= W_g x \\
u &= W_u x \\
y &= W_d \big( \phi(g) \odot u \big)
\end{aligned}
$$
其中：

- $\phi$：`hidden_act`（Qwen3 通常是 **SiLU / Swish**）
- $\odot$：逐元素乘法（Hadamard product）

**Why Gated MLP**:
$$
\phi(W_g x) \odot W_u x
$$
它的作用可以理解为：

> **用一条分支学“要不要开”，
>  用另一条分支学“开什么内容”。**

**直觉解释**

- `gate_proj + act_fn`： 学一个 **soft mask**
- `up_proj`：提供真实内容
- `*`：内容是否通过，由 gate 决定

相比来，普通的FFN:

- 非线性是 **逐元素、单通道**

- 所有维度 **无条件激活**

- 表达能力受限

` intermediate_size`的作用

- 一般是`hidden_size`的4倍
- 扩维后，参数更多，非线性更强
- 后续再降维回到`hidden_state`

