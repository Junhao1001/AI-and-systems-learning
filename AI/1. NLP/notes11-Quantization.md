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