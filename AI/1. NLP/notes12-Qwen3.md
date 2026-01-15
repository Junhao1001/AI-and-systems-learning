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

    