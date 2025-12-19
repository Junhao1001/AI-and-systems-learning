# RNN

## Language Modeling

- **Language Modeling**: the task of predicting **what word comes next**
- Formally: given a sequence of words $x^{(1)}, x^{(2)},...,x^{(t)}$ , compute the **probability distribution of the next word** $x^{(t+1)}$:

$$
P(\boldsymbol{x}^{(t+1)}|\boldsymbol{x}^{(t)},\ldots,\boldsymbol{x}^{(1)})
$$

- The probability of a piece of text:

$$
\begin{aligned}P(\boldsymbol{x}^{(1)},\ldots,\boldsymbol{x}^{(T)})&=P(\boldsymbol{x}^{(1)})\times P(\boldsymbol{x}^{(2)}|\boldsymbol{x}^{(1)})\times\cdots\times P(\boldsymbol{x}^{(T)}|\boldsymbol{x}^{(T-1)},\ldots,\boldsymbol{x}^{(1)})\\&=\prod_{t=1}^TP(\boldsymbol{x}^{(t)}|\boldsymbol{x}^{(t-1)},\ldots,\boldsymbol{x}^{(1)})\end{aligned}
$$

### n-gram LM

- N-gram LM:  depends on the preceding n-1 words
- how to get these probabilities: **counting them in some large corpus of text**
- problems:
  - **Sparsity**: 
    - some answers may never occur in data
    - even conditions (the preceding n-1 words) never occur in data
  - **Storage problems**:
    - need to store all n-grams in the corpus; increase model size

### Evaluating Language Models

- **perplexity**: 

$$
\text{perplexity}=\prod_{t=1}^T\left(\frac{1}{P_{\mathrm{LM}}(\boldsymbol{x}^{(t+1)}|\boldsymbol{x}^{(t)},\ldots,\boldsymbol{x}^{(1)})}\right)^{1/T} \\
=\exp\left(\frac{1}{T}\sum_{t=1}^{T}-\log\hat{y}_{\boldsymbol{x}_{t+1}}^{(t)}\right)=\exp(J(\theta))
$$

- equal to cross-entropy loss $J(\theta)$
- Lower perplexity is better

## Neural language model 

### fixed-window neural Language Model

- like the model that predict the center word in word2Vec 

  ![image-20251219122741136](./images/notes4-RNN/image-20251219122741136.png)

- no sparsity problem
- But fixed window is too small; enlarge window also enlarges model (W)
- No symmetry in how the inputs are processed

## RNN

### Basic Architecture

- The calculation of the current moment depends on **the hidden state of the previous moment (Memory)**

  

<img src="./images/notes4-RNN/image-20251219123203554.png" alt="image-20251219123203554" style="zoom:80%;" />

- Advantages: 
  - can process **any length** input , and won't increase model size
  - Can use information from many steps back
  - Symmetry: Same weights applied on every timestep
- Disadvantages:
  - computation is slow

### Training of RNN LM

- **Loss Function**:  **cross-entropy** between predicted probability distribution , and the true next word

$$
J^{(t)}(\theta)=CE(\boldsymbol{y}^{(t)},\hat{\boldsymbol{y}}^{(t)})=-\sum_{w\in V}\boldsymbol{y}_{w}^{(t)}\log\hat{\boldsymbol{y}}_{w}^{(t)}=-\log\hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}
$$

- Overall loss for entire training set:

$$
J(\theta)=\frac{1}{T}\sum_{t=1}^{T}J^{(t)}(\theta)=\frac{1}{T}\sum_{t=1}^{T}-\log\hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}
$$

- **Stochastic Gradient Descent**:

  - computing across entire corpus is too expensive
  - SGD: **use one or several sentences to compute the gradient and update the parameters**

  - $\theta$ is global sharing, for the whole corpus not for only one sentence

#### Backpropagation Through Time (BPTT)

- gradient:

$$
\frac{\partial \mathcal{L}}{\partial W_h}
=
\sum_{t=1}^{T}
\frac{\partial \mathcal{L}}{\partial h_t}
\frac{\partial h_t}{\partial W_h}
$$

- Loss: not only the Loss(t)
  $$
  \mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2 +..+ \mathcal{L}_t
  $$

### Vanishing/Exploding gradient

- **Gradient is multiplied in time**

$$
\frac{\partial h_t}{\partial h_{t-k}}
=
\prod_{i=t-k+1}^{t}
W_h^\top \cdot f'( \cdot )
$$

- if $W_h$ is small, the gradient is close to 0 when the sequence is long (t is big)
- if $W_h$ is large, the gradient will become very large when the sequence is long (t is big)

- **Truncated BPTT**:

  - Calculate backpropagation on only  k time steps
  - But it will miss long dependencies

- **Gradient Clipping**:

  - if the norm of the gradient is greater than some threshold, **scale it down** before applying SGD update
  - $\tau$：threshold
  - **scaling after a whole BPTT not during the BPTT**

$$
g \leftarrow g \cdot \min\left(1, \frac{\tau}{\|g\|}\right)
$$

- **LSTMs**: 
  - use gate mechanism to separate long memory and short memory



## LSTMs

### Basic Architecture:

<img src="./images/notes4-RNN/featured.png" alt="Insights into LSTM architecture | Thorir Mar Ingolfsson" style="zoom:67%;" />

- **Forget Gate f(t)**: controls what is kept vs forgotten, from previous cell state

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

- **Input Gate i(t)**: controls what parts of the new cell content are written to cell

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

- **Output Gate o(t)**: controls what parts of cell are output to hidden state

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

- **New cell content**:  the new content to be written to the cell **(key)**

$$
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$

- **Cell state**: erase (“forget”) some content from last cell state, and write (“input”) some new cell content

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

- **Hidden state**: read (“output”) some content from the cell

$$
h_t = o_t \odot \tanh(c_t)
$$

### How does LSTM  solve vanishing gradients?

- The key is: 
  $$
  c_{t-1} \;\xrightarrow{\times f_t}\; c_t
  $$

  - if  $f_t \approx 1$， preserve information over many timesteps **(still confuse)**

- Maybe try to derivate the BPTT of LSTM ?

### Other methods

#### ResNet (simple)

- Residual connections (or skip-connections)

<img src="./images/notes4-RNN/image-20251219151422011.png" alt="image-20251219151422011" style="zoom:100%;" />

- **Core concept**: **the model learn the "variations" rather than "full mappings"**

#### Dense connections

- Each layer "sees" the output (or input?) of all previous layers.

$$
xl=[x0,x1,…,xl−1]
$$

- **concatenation**

$$
y_l = H_l([x_0, x_1, \dots, x_{l-1}])
$$

#### Highway connections

$$
y = T(x) \odot H(x) + (1 - T(x)) \odot x
$$

- $H(x)$：normal linear transformation
- $T(x) = \sigma(W_T x + b_T)$：**transform gate**
- $1 - T(x)$：**carry gate**
- Replaced by ResNet in practice



## Bidirectional and multi-layer RNNs

### Bidirectional RNNs

- one-way RNN can only depend on **left context** (message in the past)
- In NLP, many predictions rely on both "left context + right context"
- **Bidirectional RNN**: a Forward RNN, a Backward RNN, then fusing their states
  - forward:   $\overrightarrow{h_t}
    = f(x_t, \overrightarrow{h_{t-1}})$
  - backward:  $\overleftarrow{h_t}
    = f(x_t, \overleftarrow{h_{t+1}})$
  - Concatenated hidden states :  $h_t = [\overrightarrow{h_t} \, ; \, \overleftarrow{h_t}]$

- need **entire input sequence**;
  - not applicable to Language Modeling, because in **LM you only have left context available**

### Multi-layer RNNS

<img src="./images/notes4-RNN/image-20251219153841377.png" alt="image-20251219153841377" style="zoom:67%;" />

- The lower RNNs should compute lower-level features
- higher RNNs should  compute higher-level features
- High-performing RNNs are usually multi-layer
