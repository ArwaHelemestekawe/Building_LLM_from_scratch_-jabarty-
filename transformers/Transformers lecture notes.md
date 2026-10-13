2026-10-12 23:45

# Transformers lecture notes
![[Pasted image 20261013002440.png]]

### Difference between positional embeddings and embeddings 

In Transformers, **Embeddings** and **Positional Embeddings** behave differently during training.

### 1. Token Embeddings (Word Embeddings)

These are the vectors representing words/tokens.

Example:

"cat" -> [0.2, -0.7, 1.3, ...]

"dog" -> [1.1, 0.4, -0.2, ...]

✅ **Trainable**

- Initialized randomly (usually).
- Updated by backpropagation during training.
- The model learns better representations for tokens.

So:

Token Embedding Matrix = Learned during training

---

### 2. Positional Embeddings

Transformers need position information because attention alone does not know word order.

There are **two main types**:

#### A. Fixed Positional Encoding (Original Transformer Paper)

Uses sine and cosine functions:

![[Pasted image 20261012234759.png]]
Example:

Position 1 -> fixed vector

Position 2 -> fixed vector

Position 3 -> fixed vector

✅ **Not trainable**

- Computed mathematically.
- Never changes during training.

---

#### B. Learned Positional Embeddings (BERT, GPT, etc.)

Instead of sine/cosine:

Position 1 -> random vector

Position 2 -> random vector

Position 3 -> random vector

These vectors are parameters of the model.

✅ **Trainable**

- Initialized randomly.
- Updated during training like normal embeddings.

---

### Quick Summary

|Component|Trainable?|
|---|---|
|Token Embeddings|✅ Yes|
|Sinusoidal Positional Encoding|❌ No|
|Learned Positional Embeddings|✅ Yes|

---

### What about modern LLMs?

Many modern models no longer use the original fixed positional encoding. They use methods such as:

- **Learned Positional Embeddings** (BERT)
- **RoPE (Rotary Positional Embeddings)** (LLaMA, Qwen, DeepSeek)
- **ALiBi**

==For **RoPE**, the positional information itself is **fixed (not learned)**. The model learns the attention weights, but the rotation formula used to inject position is not trained.==

So:

|Method|Trainable?|
|---|---|
|Sinusoidal PE|❌|
|Learned PE|✅|
|RoPE|❌|
|ALiBi|❌|

A good rule is:

> **Token embeddings are almost always trainable. Positional embeddings may be trainable or fixed depending on the architecture.**

# Multi head attention 

![[Pasted image 20261012235720.png]]

![[Pasted image 20261013002213.png]]
# Multi-Head Attention

## Assumptions

- Sequence length: $n = 4$
- Model dimension: $d_{\text{model}} = 512$
- Number of heads: $h = 8$

Input matrix:

$$
X \in \mathbb{R}^{4 \times 512}
$$

Each row represents one token, and each token is represented by a vector of length 512.

---

## 1. Create Q, K, and V

The input $X$ is projected using three different trainable weight matrices.

$$
Q = XW^Q
$$

$$
K = XW^K
$$

$$
V = XW^V
$$

Weight matrix dimensions:

$$
W^Q, W^K, W^V \in \mathbb{R}^{512 \times 512}
$$

Result:

$$
Q, K, V \in \mathbb{R}^{4 \times 512}
$$

### Why?

- **Query (Q):** What information am I looking for?
- **Key (K):** What information do I contain?
- **Value (V):** What information should be passed along?

Each projection learns a different role.

---

## 2. Why Transpose K?

For one attention head:

$$
Q_i \in \mathbb{R}^{4 \times 64}
$$

$$
K_i \in \mathbb{R}^{4 \times 64}
$$

We calculate the attention scores using:

$$
Q_i K_i^T
$$

The dimensions are:

$$
(4 \times 64)(64 \times 4) = (4 \times 4)
$$

Therefore:

$$
Q_i K_i^T \in \mathbb{R}^{4 \times 4}
$$

Each entry represents a score between a query token and a key token.

We transpose K because matrix multiplication requires the inner dimensions to match.

---

## 3. Split into Multiple Heads

The original matrices have dimensions:

$$
Q, K, V \in \mathbb{R}^{4 \times 512}
$$

Number of heads:

$$
h = 8
$$

Dimension of each head:

$$
d_k = \frac{d_{\text{model}}}{h}
$$

$$
d_k = \frac{512}{8} = 64
$$

After splitting the feature dimension:

$$
Q_i, K_i, V_i \in \mathbb{R}^{4 \times 64}
$$

There are eight heads, each processing a smaller feature dimension.

### Why multiple heads?

Different heads can learn different relationships, such as:

- Subject–verb relationships
- Nearby words
- Long-range dependencies
- Negation
- Other contextual relationships

These are possibilities learned by the model, not fixed assignments.

---

## 4. Compute Attention Inside Each Head

The scaled dot-product attention formula is:

$$
\operatorname{head}_i =
\operatorname{softmax}
\left(
\frac{Q_i K_i^T}{\sqrt{d_k}}
\right)V_i
$$

### Step 1: Calculate attention scores

$$
Q_i K_i^T
$$

Dimensions:

$$
(4 \times 64)(64 \times 4) = (4 \times 4)
$$

### Step 2: Scale the scores

$$
\frac{Q_i K_i^T}{\sqrt{d_k}}
$$

Since $d_k = 64$:

$$
\sqrt{d_k} = \sqrt{64} = 8
$$

Therefore:

$$
\frac{Q_i K_i^T}{8}
\in \mathbb{R}^{4 \times 4}
$$

Scaling prevents dot-product scores from becoming excessively large.

### Step 3: Apply softmax

$$
A_i =
\operatorname{softmax}
\left(
\frac{Q_i K_i^T}{\sqrt{d_k}}
\right)
$$

Result:

$$
A_i \in \mathbb{R}^{4 \times 4}
$$

Each row contains the attention weights for one query token.

### Step 4: Multiply by V

$$
\operatorname{head}_i = A_i V_i
$$

Dimensions:

$$
(4 \times 4)(4 \times 64) = (4 \times 64)
$$

Therefore:

$$
\operatorname{head}_i \in \mathbb{R}^{4 \times 64}
$$

Each output token is a weighted combination of value vectors.

---

## 5. Concatenate the Heads

There are eight attention heads:

$$
\operatorname{head}_1, \ldots, \operatorname{head}_8
$$

Each head has shape:

$$
\operatorname{head}_i \in \mathbb{R}^{4 \times 64}
$$

Concatenate them along the feature dimension:

$$
H =
\operatorname{Concat}
\left(
\operatorname{head}_1,
\ldots,
\operatorname{head}_8
\right)
$$

The resulting shape is:

$$
H \in \mathbb{R}^{4 \times 512}
$$

Because:

$$
8 \times 64 = 512
$$

### Why concatenate?

Each head learns different features. Concatenation combines these features into one representation.

---

## 6. Final Linear Projection

The output projection matrix has dimensions:

$$
W^O \in \mathbb{R}^{512 \times 512}
$$

The final output is:

$$
\operatorname{MultiHead}(Q,K,V) = HW^O
$$

Dimensions:

$$
(4 \times 512)(512 \times 512) = (4 \times 512)
$$

Therefore:

$$
\operatorname{MultiHead}(Q,K,V)
\in \mathbb{R}^{4 \times 512}
$$

### Why do we need this projection?

Concatenation places the outputs of all heads side by side. The learned matrix $W^O$ mixes and transforms those features into the final representation.

Think of it as combining the results of eight experts.

---

## 7. Complete Flow

**Input**

$$
X \in \mathbb{R}^{4 \times 512}
$$

**Step 1: Project the input**

$$
Q = XW^Q,\qquad
K = XW^K,\qquad
V = XW^V
$$

**Step 2: Split into eight heads**

$$
Q_i, K_i, V_i \in \mathbb{R}^{4 \times 64}
$$

**Step 3: Compute attention**

$$
\operatorname{head}_i =
\operatorname{softmax}
\left(
\frac{Q_i K_i^T}{\sqrt{64}}
\right)V_i
$$

**Step 4: Concatenate**

$$
H =
\operatorname{Concat}
\left(
\operatorname{head}_1,
\ldots,
\operatorname{head}_8
\right)
$$

$$
H \in \mathbb{R}^{4 \times 512}
$$

**Step 5: Output projection**

$$
\operatorname{Output} = HW^O
$$

$$
\operatorname{Output} \in \mathbb{R}^{4 \times 512}
$$

---

## 8. Shape Summary

| Operation | Shape |
|---|---|
| Input X | $4 \times 512$ |
| Q, K, V | $4 \times 512$ each |
| Q, K, V for each head | $4 \times 64$ each |
| Attention scores | $4 \times 4$ |
| Attention weights | $4 \times 4$ |
| Output of each head | $4 \times 64$ |
| Concatenated heads | $4 \times 512$ |
| Final output | $4 \times 512$ |

---

## 9. Important Correction: What Is a Head?

An imprecise explanation is:

> Divide the columns into groups and call each group a head.

A more accurate explanation is:

> Split the feature dimension into smaller subspaces. Each head uses its own query, key, and value projections and performs a complete attention calculation.

A head is not simply a group of columns. It calculates attention scores, converts them into weights, and uses those weights to combine value vectors.

**Key idea:** Multi-head attention lets a model learn different relationships between tokens simultaneously.


# What is the difference between multi head attention in encoder and MASKED multi head attention in decoder and why we need the MASKED one ?


![[Pasted image 20261013003042.png]]


# Encoder Attention vs. Decoder Masked Attention

## 1. Encoder: Normal Multi-Head Attention

Suppose the input sentence is:

> I love machine learning.

The encoder receives the entire input sentence at once. Therefore, each token can attend to every token in the input, including tokens that appear later in the sentence.

For example:

- **I** can attend to: I, love, machine, learning
- **love** can attend to: I, love, machine, learning
- **machine** can attend to: I, love, machine, learning
- **learning** can attend to: I, love, machine, learning

### Encoder Attention Matrix

Here, each row represents a query token, and each column represents a key token.

| Query \ Key | I | love | machine | learning |
|---|---|---|---|---|
| I | ✓ | ✓ | ✓ | ✓ |
| love | ✓ | ✓ | ✓ | ✓ |
| machine | ✓ | ✓ | ✓ | ✓ |
| learning | ✓ | ✓ | ✓ | ✓ |

Nothing is hidden by a causal mask.

### Why is this okay?

The encoder is responsible for understanding the input sentence. Since the entire input is already available, it can use information from both earlier and later tokens.

For example, the representation of a word can incorporate context from the words that follow it.

---

## 2. Decoder: What Is the Problem?

Suppose the decoder is generating:

> I love machine learning.

During autoregressive generation, the decoder produces tokens step by step.

**Step 1:**

```text
I
```

**Step 2:**

```text
I love
```

**Step 3:**

```text
I love machine
```

**Step 4:**

```text
I love machine learning
```

When predicting the next token, the decoder must not use future tokens that have not yet been generated.

### What happens without a mask?

During training, suppose the decoder receives:

```text
I love machine learning
```

The model is trained to predict the next token from the preceding tokens. In a standard next-token training setup, the input and target are shifted:

| Decoder input | Target |
|---|---|
| I | love |
| I love | machine |
| I love machine | learning |

If the decoder could attend to future positions, it might access the answer it is supposed to predict.

For example, when predicting `machine`, unrestricted attention could allow the model to access future tokens such as `learning`.

**This creates information leakage during training. The model could exploit information unavailable during actual generation.**

**Important:** In a correctly shifted causal language model, the prediction at position 2 must not access the token at position 3 or any later input position.

---

## 3. Masked Multi-Head Attention

To prevent future information from leaking into earlier positions, the decoder uses a **causal mask**, also called a **look-ahead mask**.

For a sequence of length 4, the allowed attention positions are:

| Query position | Key 1 | Key 2 | Key 3 | Key 4 |
|---|---|---|---|---|
| 1 | ✓ | X | X | X |
| 2 | ✓ | ✓ | X | X |
| 3 | ✓ | ✓ | ✓ | X |
| 4 | ✓ | ✓ | ✓ | ✓ |

Where:

- ✓ = allowed
- X = forbidden

The rule is:

**A token can attend to its own position and earlier positions, but not later positions.**

This produces a lower-triangular attention pattern.

### Example

Suppose the decoder input is:

```text
I love machine learning
```

At position 3, the token `machine` can attend to:

- I
- love
- machine

It cannot attend to:

- learning

The future token is blocked by the causal mask.

---

## 4. How the Causal Mask Works Mathematically

First, attention scores are calculated using the query and key matrices:

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

For a sequence of length 4:

$$
S \in \mathbb{R}^{4 \times 4}
$$

Suppose the attention score matrix is:

$$
S =
\begin{bmatrix}
2 & 1 & 3 & 5 \\
4 & 2 & 1 & 6 \\
3 & 5 & 4 & 7 \\
1 & 2 & 3 & 4
\end{bmatrix}
$$

These values are illustrative attention scores.

### Step 1: Apply the causal mask

Future positions are replaced with negative infinity, written as $-\infty$.

The masked score matrix becomes:

$$
S_{\text{masked}} =
\begin{bmatrix}
2 & -\infty & -\infty & -\infty \\
4 & 2 & -\infty & -\infty \\
3 & 5 & 4 & -\infty \\
1 & 2 & 3 & 4
\end{bmatrix}
$$

### Step 2: Apply softmax

The attention weights are calculated as:

$$
A = \operatorname{softmax}(S_{\text{masked}})
$$

Softmax is defined as:

$$
\operatorname{softmax}(x_i)
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

Since:

$$
e^{-\infty} = 0
$$

every masked position receives an attention weight of zero.

For example, the first row becomes:

$$
[1,\ 0,\ 0,\ 0]
$$

The first token can only attend to itself.

The second row assigns weights only to the first two positions. The third row assigns weights only to the first three positions.

Therefore, future tokens cannot contribute information to earlier positions.

### Step 3: Multiply by the value matrix

Finally, the attention weights are multiplied by the value matrix:

$$
\operatorname{Attention}(Q,K,V) = AV
$$

Because the masked positions have zero weights, their value vectors do not contribute to the output at that position.

---

## 5. A Decoder Block Contains Two Different Attention Mechanisms

In a standard encoder-decoder Transformer, a decoder block contains two attention sublayers.

### A. Masked Self-Attention

The queries, keys, and values all come from the decoder's own representations.

$$
Q = X_{\text{dec}}W^Q
$$

$$
K = X_{\text{dec}}W^K
$$

$$
V = X_{\text{dec}}W^V
$$

A causal mask prevents each position from accessing future positions.

**Purpose:** Allow the decoder to use its existing context without seeing future target tokens.



### B. Cross-Attention

![[Pasted image 20261013004233.png]]

Cross-attention allows the decoder to use information from the encoder's output.

The queries come from the decoder:

$$
Q = X_{\text{dec}}W^Q
$$

The keys and values come from the encoder output:

$$
K = H_{\text{enc}}W^K
$$

$$
V = H_{\text{enc}}W^V
$$

Here:

- $X_{\text{dec}}$ = decoder representations
- $H_{\text{enc}}$ = encoder output representations

==**The decoder uses its queries to retrieve relevant information from the encoded input.**==

### Why is a causal mask usually unnecessary for cross-attention?

The encoder has already processed the complete source sentence. The source tokens are available to the decoder, so attending to later source positions does not reveal future target tokens.

Cross-attention can therefore attend to all valid source positions, although padding masks may still be required.

---

## 6. Quick Comparison

| Feature | Encoder Self-Attention | Decoder Masked Self-Attention | Decoder Cross-Attention |
|---|---|---|---|
| Queries come from | Encoder | Decoder | Decoder |
| Keys come from | Encoder | Decoder | Encoder |
| Values come from | Encoder | Decoder | Encoder |
| Can attend to future positions? | Yes, within the input | No | Source positions are not causally masked |
| Uses a causal mask? | No | Yes | Usually no |
| Main purpose | Understand input context | Model target sequence causally | Retrieve information from the input |

Note: Padding masks can be used in addition to causal masks where necessary.

---

## 7. One-Sentence Intuition

**Encoder attention** is like reading an entire book before answering questions.

**Decoder masked attention** is like writing a story one word at a time without peeking at words you have not written yet.

**Decoder cross-attention** is like consulting the book while writing your answer.

---

## 8. The Most Important Takeaway

The causal mask does not stop the decoder from using previous tokens. It only blocks access to future positions.

Mathematically, the attention computation is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}} + M
\right)V
$$

where $M$ is the causal mask:

$$
M_{ij} =
\begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

Here, $i$ is the query position and $j$ is the key position.

The condition $j \leq i$ allows the current and previous positions, while $j > i$ blocks future positions.

**In short:** The encoder can use the whole input context, while decoder self-attention must preserve the left-to-right order needed for autoregressive generation.


# How to generate text: using different decoding methods for language generation with Transformers
link: [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)

![[Pasted image 20261013014448.png]]


 ==🔴**auto-regressive** means **"self-repeating"** or **"step-by-step."** Instead of generating an entire paragraph all at once, the model generates text **one single word (or token) at a time**, feeding its previous outputs back into itself as input to generate the next one.==


$$P(w_{1:T} \mid W_0) = \prod_{t=1}^{T} P(w_t \mid w_{1:t-1}, W_0)$$Here is what each piece means in plain English:

$W_0$ (The Context): The initial prompt or starting text you give the model (e.g., "I enjoy walking with my cute dog").

$w_{1:T}$ (The Full Sequence): The complete generated text consisting of $T$ words ($w_1, w_2, w_3, \dots, w_T$).

$\prod_{t=1}^{T}$ (The Product): A mathematical symbol meaning "multiply everything together." <mark style="background: #FF5582A6;">It signifies that the overall probability of the whole sentence is the product of the probabilities of each individual step.</mark>

$P(w_t \mid w_{1:t-1}, W_0)$ (The Conditional Probability): At any given step $t$, the model calculates the probability of the next word ($w_t$)<mark style="background: #FF5582A6;"> based on everything that came before it—which includes both the initial prompt ($W_0$) and all the words generated so far ($w_{1:t-1}$)</mark>.


______________________________________________________
## Greedy Search

Greedy search is the simplest decoding method. It selects the word with the highest probability as its next word: 

![[Pasted image 20261013193002.png]]


<mark style="background: #FF5582A6;">Starting from the word "The","The", the algorithm greedily chooses the next word of highest probability "nice""nice" and so on, so that the final generated word sequence is ("The","nice","woman")("The","nice","woman") having an overall probability of 0.5×0.4=0.20.5×0.4=0.2 .</mark>


![[Pasted image 20261013194053.png]]
### The major drawback of greedy search though is that it misses high probability words hidden behind a low probability word as can be seen in our sketch above:


The word "has""<mark style="background: #FF5582A6;"> has" with its high conditional probability of 0.90.9 is hidden behind the word "dog"</mark> " dog", which has only the second-highest conditional probability, <mark style="background: #FF5582A6;">so that greedy search misses the word sequence "The","dog","has"</mark>


## Beam search

Beam search solves a major flaw of greedy search: <mark style="background: #FF5582A6;">**it doesn't blindly lock into the single most probable word at every step.** Instead, it keeps track of multiple promising paths simultaneously.
</mark>
Setting **`num_beams=2`** means the model preserves the **top 2 most likely hypotheses** at every single time step rather than just 1.

### Step-by-Step Illustration (`num_beams=2`)

Imagine the model is building a sentence starting with the context _"The"_, and we want to generate the next two words.

#### Step 1: Generating the First Word ($t = 1$)

The model evaluates all possible next words and ranks them by probability:

1. _"nice"_ (probability: $0.5$) — _Top Hypothesis #1_
    
2. _"dog"_ (probability: $0.4$) — _Top Hypothesis #2_
    
3. _"car"_ (probability: $0.2$) — _Lower-ranked, discarded by beam search_
    

Because `num_beams=2`, the model keeps both paths alive:

- **Beam Path 1:** `("The", "nice")`
    
- **Beam Path 2:** `("The", "dog")`
    

#### Step 2: Generating the Second Word ($t = 2$)

Now, the model looks ahead for _both_ paths independently to find their next options:

- **Expanding Path 1 (`("The", "nice")`):**
    
    - *Next word "woman" gives a cumulative probability of $0.5 \times 0.4 = \mathbf{0.2}$.*
        
    - Next word _"house"_ gives a lower probability.
        
- **Expanding Path 2 (`("The", "dog")`):**
    
    - Next word _"has"_ gives a cumulative probability of $0.4 \times 0.9 = \mathbf{0.36}$. _(Notice that _"has"_ itself has a high local probability of $0.9$, which greedy search missed in Step 1 because it threw away _"dog"_)._
        

#### Step 3: Choosing the Overall Winner

The model evaluates all generated paths across all steps:

- Path 1 (`"The", "nice", "woman"`) = **0.2** total probability
    
- Path 2 (`"The", "dog", "has"`) = **0.36** total probability
    

==🟡Beam search selects **Path 2** because it has the highest overall cumulative probability ($\mathbf{0.36 > 0.2}$), successfully recovering a hidden high-probability phrase that a greedy search would have missed.==


### Beam search is not guaranteed to find the absolute highest-probability output sequence for two main reasons:

- **It explores a limited subset of paths:** By setting `num_beams=2` (or any fixed number), the model strictly drops every other path at every single time step. If a sequence has a temporarily low probability early on but would have ballooned into an extremely high-probability ending later, beam search permanently loses it the moment it falls outside the top $N$ beams.
    
- **The search space grows exponentially:** Language generation expands combinatorially at every word step ($V^T$, where $V$ is the vocabulary size and $T$ is length). Checking _every_ possible combination to find the absolute mathematical maximum is computationally impossible, so beam search relies on a heuristic shortcut that sacrifices global optimality for speed.


![[Pasted image 20261013195531.png]]

An **n-gram** is simply a sequence of $n$ consecutive words in a text (for example, a 2-gram or "bigram" is a pair of two words like "walk with", and a 3-gram or "triggin" is three words like "walk with my").

<mark style="background: #FF5582A6;">When language models generate text using beam search, they often get stuck in repetitive loops (repeating the exact same phrase over and over). To fix this, an **n-gram penalty** acts as a strict guardrail:</mark>

1. **Tracking History:** As the model generates words step by step, it keeps track of every $n$-gram phrase it has already outputted in the current text.
    
2. **Blocking Repeats:** Before picking the next word, the algorithm checks: _"Will adding this candidate word complete an $n$-gram that already exists earlier in the text?"_
    
3. **Setting Probability to Zero:** If the answer is yes, the model **manually forces the probability of that word to $0$** (completely eliminating it from the choice pool), <mark style="background: #FF5582A6;">forcing the model to choose a different, non-repetitive word instead.
</mark>



![[Pasted image 20261013200258.png]]

### Another important feature about beam search is that we can compare the top beams after generation and choose the generated beam that fits our purpose best

To understand why beam search struggles with open-ended generation tasks like storytelling or chatting compared to structured tasks like translation, it helps to look at how length impacts the algorithm:

### 1. Predictable Length vs. Open-Ended Length

- **Machine Translation & Summarization:** These tasks have a **tightly bounded output length**. If you are translating a 10-word sentence from French to English, you expect roughly a 10-word output. Because the finish line is near, beam search can efficiently score and rank paths without the probabilities degrading over time.
    
- **Dialog & Story Generation:** These tasks are **open-ended**. A story could be 50 words or 500 words long. <mark style="background: #FF5582A6;">Because beam search continuously multiplies probabilities at every single step, longer sequences naturally accumulate lower cumulative probabilities.</mark> This causes the algorithm to favor prematurely cutting off the text or opting for short, safe, and bland sentences just to keep the cumulative score mathematically high.
    

### 2. The Trap of Repetitive Safety

Because beam search aggressively optimizes for the highest mathematical probability, <mark style="background: #FF5582A6;">it tends to gravitate toward common phrases and boilerplate structures. In open-ended creative tasks where users expect surprise, variation, and dynamic turns, beam search often produces dry, repetitive text loops that lack human-like creativity</mark>


# Sampling

### How Basic Sampling Works

Unlike greedy search (which _always_ picks the single highest probability word) or beam search (which tracks top paths), basic sampling treats the next-word probability distribution like a **lottery wheel**:

1. **Proportional Chances:** Every word is given a chance of being selected based on its exact probability score. Words with higher probabilities have larger "slices of the pie," while lower probability words have smaller slices.
    
2. **Random Selection ($w_t \sim P(w \mid w_{1:t-1})$):** The model rolls a virtual dice, and whichever word the random roll hits is chosen.



![[Pasted image 20261013201931.png]]

The graphic walks through a two-step random generation process starting from the context `"The"`:

- **Step 1 (Choosing the first word after `"The"`):**
    
    - The model evaluates the options: `"nice"` ($0.5$), `"dog"` ($0.4$), and `"car"` ($0.1$).
        
    - Instead of deterministically picking `"nice"` (the highest score), the random sample rolls and hits **`"car"`** (highlighted in red with a probability of $0.1$). Even though it is unlikely, it gets chosen because of the random element.
        
- **Step 2 (Choosing the next word after `"car"`):**
    
    - The context is now `"The car"`, and the model evaluates new options: `"drives"` ($0.5$), `"is"` ($0.3$), and `"turns"` ($0.2$).
        
    - The random roll hits **`"drives"`** (highlighted in red with a probability of $0.5$).
        
- **Final Sequence:** The resulting generated phrase becomes `("car", "drives")`.
    

### The Main Takeaway & Trade-off

- **Why it's used:** It makes text generation **non-deterministic** and much more creative, allowing the model to produce varied, human-like phrasing.
    
- **The drawback:** Because it can randomly pick low-probability words (like `"car"` after `"The"`), basic sampling can sometimes produce incoherent gibberish or nonsensical jumps if left completely unfiltered.


### 1. Temperature (TTT) — _Controls the Sharpness of the Distribution_

- **What it does:** <mark style="background: #FF5582A6;">Temperature scales the probability distribution before sampling by dividing the logits.</mark>
    
- **High Temperature (e.g., 1.01.01.0 or higher):**<mark style="background: #FF5582A6;"> Flattens the distribution out, making words have closer probabilities. This increases randomness, creativity, and unexpected phrasing</mark>.
    
- **Low Temperature (e.g., 0.60.60.6):** Sharpens the distribution, making high-probability words even more dominant and suppressing unlikely choices.
    
- **The Limit (T→0T \to 0T→0):**<mark style="background: #FF5582A6;"> As temperature approaches zero, it mathematically collapses into **greedy decoding**,</mark> losing all randomness and suffering from the exact same repetitive loops.
    

### 2. Top-K — _Controls the Pool Size_

- **What it does:** Acts as a hard boundary cutoff on vocabulary choices. It forces the model to only consider the **KKK most likely next words** and completely throws away everything else.
    
- **Small KKK (e.g., `top_k=10`):** Keeps choices tightly restricted to the safest, most probable options, yielding highly predictable text.
    
- **Large KKK or Zero (e.g., `top_k=50` or `top_k=0`):** Expands or removes the cutoff, allowing the model to look deeper into less common vocabulary (though setting it to 0 without other filters can introduce gibberish).
    

### How They Combine in Practice

In Hugging Face generation, you usually use them together to get the best of both worlds—creative flavor backed by quality control:

Python

```
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,          # Keeps only the top 50 choices
    temperature=0.7    # Smooths/adjusts the probabilities within those 50 choices
)
```

By balancing **temperature** (how wild or safe the probabilities are) and **top-k** (how many candidates are allowed in the race), you can fine-tune your model to write anything from creative stories to structured conversational responses.


# Top k sampling

Top-K sampling is a text-generation technique that solves the issue of basic sampling (which can sometimes produce completely random, incoherent text) by restricting the model's choices to only the most sensible options.

### How Top-K Sampling Works

1. **Ranking Candidates:** At every step, the model ranks all words in its vocabulary based on their conditional probability distribution.
    
2. **Filtering the Pool:** It keeps only the **$K$ most likely next words** (e.g., the top 50 words when `top_k=50`) and completely discards everything else.
    
3. **Redistributing Probability Mass:** The remaining probability mass is rescaled so that it totals $100\%$ among _only_ those $K$ words, and the model randomly samples its next choice from this filtered pool.


#### We need to redistribute the probabilities so that the filtered set of words remains a **valid probability distribution** (where all remaining probabilities add up to exactly 1.01.01.0, or 100%100\%100%).

### The Mathematical Reason

1. **What Happens During Filtering:** When Top-K sampling keeps only the top KKK words and sets all other vocabulary words to 000, the sum of the remaining probabilities shrinks. For example, if you chop off the long tail of low-probability words, your remaining KKK words might now only add up to 0.70.70.7 (70%70\%70%) instead of 1.01.01.0.
    
2. **The Rules of Probability:** A proper sampling process requires the total probability mass to equal 1.01.01.0 so that the random "lottery roll" works correctly across the available options.
    
3. **Rescaling (Redistribution):** To fix this, the algorithm divides each kept word's score by the new total sum (renormalizing). This scales the remaining probabilities up proportionally so they fill the entire 100%100\%100% pool again, ensuring every word in the filtered list maintains its fair, relative weight to the others.

### 2. Distorted Relative Weighting

If you keep the old, unscaled probabilities (e.g., $0.5$, $0.3$, and $0.2$) without renormalizing them to fill the 1.0 space, those specific numbers lose their correct proportional relationship to one another within the newly restricted pool.


![[Pasted image 20261013203341.png]]

### Step-by-Step Breakdown of the Visual

1. **The First Step ($P(w \mid \text{"The"})$ — Left Side):**
* **The Distribution Shape:** This is a **flat distribution** (the probabilities are spread out relatively evenly across many words like *nice*, *dog*, *car*, *woman*, etc.).


* **Applying $K=6$:** The algorithm keeps only the top 6 most likely words (*nice*, *dog*, *car*, *woman*, *guy*, *man*) and cuts off the rest (*people*, *big*, *house*, *cat*).


* **Probability Mass:** Even though it keeps 6 words, those top words together only encompass about **$68\%$** ($\sum_{w \in V_{\text{top-K}}} P(w \mid \text{"The"}) = 0.68$) of the total probability mass because the distribution is so flat.





---

2. **The Second Step ($P(w \mid \text{"The"}, \text{"car"})$ — Right Side):**
* **The Distribution Shape:** This is a **sharp distribution** (a couple of words like *drives* and *is* dominate heavily, while everything else drops off fast).


* **Applying $K=6$:** The algorithm keeps the top 6 words (*drives*, *is*, *turns*, *stops*, *down*, *a*) and cuts off the long tail of weird, low-probability words (*not*, *the*, *small*, *told*).


* **Probability Mass:** Because the top words are so dominant, those 6 words capture nearly **$99\%$** ($\sum_{w \in V_{\text{top-K}}} P(w \mid \text{"The"}, \text{"car"}) = 0.99$) of the total probability mass.





---

### Key Takeaway from the Graphic

The image highlights both the power and the limitation of Top-K sampling 

* **The Good:** It successfully cleans up erratic choices (like eliminating *not*, *the*, *small*, *told* in the second step).


* **The Flaw:** Because $K$ is fixed to a static number (6 in this illustration), i<mark style="background: #FF5582A6;">t can be too restrictive on flat distributions (dropping reasonable words like *people* or *cat* in step 1) while being too loose on sharp distributions (keeping weirdly fitted words like *down* or *a* in step 2). This exact limitation is what motivates **Top-p (nucleus) sampling**.</mark>




Spot on! You hit the nail on the head.

This code snippet is essentially standard sampling, but with one critical safeguard added:

```python
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50  # Restricts the pool to the 50 most likely candidates
)

```

### Why it differs from basic sampling:

* In basic sampling (`top_k=0`), the model chooses from *every single word* in its entire dictionary, which often results in weird typos or incoherent gibberish.
* By setting **`top_k=50`**, you tell the model: *"You can still use random sampling (`do_sample=True`), but you are only allowed to roll the dice among the **top 50 most sensible choices** at any given moment."*

It gives you the creative freedom of sampling while cutting off the wild, low-probability tail of the vocabulary to keep the text coherent and natural!

### ==🔴1. The Core Problem: Static Size vs. Dynamic Context==

When a language model predicts text, the shape of its probability distribution changes constantly from one word to the next:

- **Flat Distributions:** Sometimes the model is uncertain, meaning many words are equally valid (e.g., after `"The"`, words like _nice_, _dog_, _car_, _people_, _house_ are all plausible).
    
- **Sharp Distributions:** Other times the model is very certain, meaning one or two words dominate completely (e.g., after `"The car"`, words like _drives_ or _is_ take up almost all the probability mass).
    

### 2. Why Fixed $K$ Fails in Both Scenarios

Because Top-K uses a strict, fixed number (like $K = 6$), it causes opposite problems depending on the distribution type:

- **On Flat Distributions (Too Restrictive):** If you enforce a fixed cutoff like $K=6$ on a flat distribution, you accidentally cut off other totally reasonable candidate words (like `"people"`, `"big"`, `"house"`, or `"cat"`) simply because they ranked 7th or lower. This limits the model's creative potential.
    
- **On Sharp Distributions (Too Loose):** Conversely, if you keep $K=6$ on a sharp distribution where only the top 2 words actually make sense, the method is forced to drag in 4 extra, poorly-fitting words (like `"down"` or `"a"`) just to fill up the quota of 6. This risks introducing random gibberish





### Top-p (nucleus) sampling

Top-p (nucleus) sampling solves the rigidity of ==🔴**Top-K by dynamically changing the size of the word pool based on how predictable or unpredictable the context is**==.

Instead of picking a fixed number of words ($K$), Top-p looks at the cumulative probability and asks: **"What is the smallest group of top words that add up to $p\%$ (e.g., $92\%$) of the total probability mass?"**

---

### How It Works Visually (Comparing the Two Scenarios)

* **Scenario A: Flat / Unpredictable Distribution (Left side of the blog graphic)**
* When the model is uncertain (like after the word `"The"`), the probabilities are spread out evenly among many plausible words.
* To reach the $92\%$ cumulative threshold, Top-p dynamically expands its pool to include **9 different words**. It gives the model wide creative freedom precisely when the sentence can go in many directions.




* **Scenario B: Sharp / Predictable Distribution (Right side of the blog graphic)**
* When the model is very confident (like after `"The car"`), the top choices (*drives*, *is*) dominate almost the entire probability space.
* To reach that same $92\%$ threshold, Top-p shrinks its pool down to just **3 words**. It cuts off all the weird tail-end words instantly, locking down on safety and coherence when predictability is high.


![[Pasted image 20261013210230.png]]
### Why It's Better

Unlike Top-K (which forces a rigid size whether the model is guessing wildly or is 100% certain), Top-p **stretches when the model needs creativity** and **shrinks when the model needs precision**.



#### While in theory, _Top-p_ seems more elegant than _Top-K_, both methods work well in practice. _Top-p_ can also be used in combination with _Top-K_, which can avoid very low ranked words while allowing for some dynamic selection.


**`num_return_sequences=3`**: Tells the model to run the generation process three separate times independently, returning **3 distinct variations** of the text using the randomness of sampling



# References
