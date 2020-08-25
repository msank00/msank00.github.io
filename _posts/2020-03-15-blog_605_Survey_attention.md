---
layout: post
title:  "Survey - Attention"
date:   2020-03-15 00:00:10 -0030
categories: jekyll update
mathjax: true
---

<link rel="stylesheet" href="/assets/css/markdownstyle.css">

**DISCLAIMER:** I am not the original author of these contents. I compile from various sources to understand a concept and for future reference as I believe in DRY (Don’t Repeat Yourself !!). This blog is nothing but a SCRIBE for me. Solely for education purposes. I try my best to put all the due credits in the reference. If you find any content is incorrect or credit is missing, please contact me. I will be happy to add/edit them. 
{: .olive}

----

# Content

1. TOC
{:toc}

---

# Introduction

From the blog of [Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. Google Translate started using such a [model](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/) in production in late 2016. These models are explained in the two pioneering papers ([Sutskever et al. 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)).

In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words:

<center>
<figure class="video_container">
  <iframe src="https://jalammar.github.io/images/seq2seq_2.mp4" frameborder="0" allowfullscreen="true" width="100%" height="150"> </iframe>
</figure>
</center>

Under the hood, the model is composed of an `encoder` and a `decoder`.

The `encoder` processes each item in the input sequence, it compiles the information it captures into a vector (called the `context`). After processing the entire input sequence, the encoder sends the context over to the `decoder`, which begins producing the output sequence item by item.


<center>
<figure class="video_container">
  <iframe src="https://jalammar.github.io/images/seq2seq_4.mp4" frameborder="0" allowfullscreen="true" width="100%" height="150"> </iframe>
</figure>
</center>

The context is nothing but a **vector of floats**.  It is basically the number of hidden units in the encoder RNN. 


The `context` vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). These papers introduced and refined a technique called **Attention**, which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.

An attention model differs from a classic sequence-to-sequence model in two main ways:

- First, the `encoder` passes a lot more data to the `decoder`. Instead of passing the last hidden state of the encoding stage, the encoder **passes all the hidden states to the decoder**:


<center>
<figure class="video_container">
  <iframe src="https://jalammar.github.io/images/seq2seq_7.mp4" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>


**Attention mechanism takes a linear combination of all the encoder hidden states**
{: .red}


- Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:

- Look at the set of encoder hidden states it received – each encoder hidden states is most associated with a certain word in the input sentence
- Give each hidden states a score (let’s ignore how the scoring is done for now)
- Multiply each hidden states by its softmaxed score, thus amplifying hidden states with high scores, and drowning out hidden states with low scores

<center>
<figure class="video_container">
  <iframe src="https://jalammar.github.io/images/attention_process.mp4" frameborder="0" allowfullscreen="true" width="100%" height="350"> </iframe>
</figure>
</center>


<center>
<figure class="video_container">
  <iframe src="https://jalammar.github.io/images/attention_tensor_dance.mp4" frameborder="0" allowfullscreen="true" width="100%" height="350"> </iframe>
</figure>
</center>


>  The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important.


Neural networks, in particular recurrent neural networks (RNNs), are now at the core of the leading approaches to language understanding tasks such as **language modeling, machine translation and question answering**. In [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the authors introduce the **Transformer**, a novel neural network architecture based on a self-attention mechanism that we believe to be particularly well suited for language understanding.

In the paper, the author shows that the Transformer outperforms both recurrent and convolutional models on academic `English to German` and `English to French` translation benchmarks. On top of higher translation quality, the Transformer requires **less computation to train** and is a much better fit for modern machine learning hardware, speeding up training by up to an order of magnitude.

The **sequence-to-sequence** neural network models are widely used for NLP. A popular type of these models is an **encoder-decoder**. There, one part of the network — encoder — encodes the input sequence into a fixed-length context vector. This vector is an internal representation of the text. This context vector is then decoded into the output sequence by the decoder. See an example:

<center>
<img src="https://miro.medium.com/max/900/1*1ui7iDq956eDs-mAZHEdIg.png" width="600">
</center>


However, there is a catch with the common encoder-decoder approach: a neural network compresses all the information of an input source sentence into a fixed-length vector. It has been shown that this leads to a decline in performance when dealing with long sentences. The attention mechanism was introduced by Bahdanau in “Neural Machine Translation by Jointly Learning to Align and Translate” to alleviate this problem.

:paperclip: **Reference:**

- [Attention Explained by Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) :fire: :fire: MUST read

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Transformer - Visual Understanding

Must Watch :fire:

<center>

<figure class="video_container">
  <iframe src="https://www.youtube.com/watch?v=mV8c9dJpxQE" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>

</center>

_*If above link is broken, click [here](https://www.youtube.com/watch?v=mV8c9dJpxQE)_

- The Transformer uses attention to boost the speed with which these models can be trained.
- The biggest benefit, however, comes from how The Transformer lends itself to **parallelization**.
- At the core of Transformer, there is `encoder-decoder` block

The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there’s nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.

<center>
<img src="https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png" width="500" alt="image">
</center>

**The encoders are all identical in structure (yet they do not share weights)**.
{: .purple}

- Each `encoder` block is a combination of a FFNN and Self-Attention


<center>
<img src="https://jalammar.github.io/images/t/Transformer_encoder.png" width="500" alt="image">
</center>

- The encoder’s inputs first flow through a self-attention layer

**A layer that helps the encoder look at other words in the input sentence as it encodes a specific word.**
{: .orange}

- The FFNNs are all independent, this helps to run the code in parallel.

As multiple encoders are stacked on top of each other, **ENCODER#0** receives the `regular word embedding` which passes through self-attention layer followed by FFNN. The output of these FFNN are form of `enriched word embedding` which act as input to the **ENCODER#1**.

The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence

<center>
<img src="https://jalammar.github.io/images/t/Transformer_decoder.png" width="500" alt="image">
</center>


## Self-Attention in Detail

Must Watch :fire:

<center>

<figure class="video_container">
  <iframe src="https://www.youtube.com/watch?v=flHtAUGB0PU&t=1s" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>

</center>

_*If above link is broken, click [here](https://www.youtube.com/watch?v=flHtAUGB0PU&t=1s)_


The **first step** in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a `Query vector`, a `Key vector`, and a `Value vector`. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

<center>
<img src="https://jalammar.github.io/images/t/transformer_self_attention_vectors.png" width="500">
</center>

- Multiplying $x_1$ by the $W^Q$ weight matrix produces $q_1$, the `query` vector associated with that word. We end up creating a `query`, a `key`, and a `value` projection of each word in the input sentence. 

**What are the “query”, “key”, and “value” vectors?**
{: .red}

- They’re abstractions that are useful for calculating and thinking about attention. 


The **second step** in calculating self-attention is to calculate a `score`. Say we’re calculating the self-attention for the first word in this example, `Thinking`. We need to **score each word of the input sentence against this word**. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The **third and forth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients and then normalize them by taking Softmax of these scores.

This softmax score determines how much each word will be expressed at this position.

The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to `keep intact the values` of the word(s) we want to focus on, and `drown-out irrelevant words` (by multiplying them by tiny numbers like $0.001$, for example).

The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at **this position (for the first word)**.

<center>
<img src="https://jalammar.github.io/images/t/self-attention-output.png" width="500" alt="image">
</center>

The resulting vector is one we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing.


## Matrix Calculation of Self-Attention

The first step is to calculate the `Query`, `Key`, and `Value` matrices. We do that by packing our embeddings into a matrix $X$, and multiplying it by the weight matrices we’ve trained ($W^Q$, $W^K$, $W^V$).


<center>
<img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation.png" width="300" alt="image">
</center>

- Every row in the X matrix corresponds to a word in the input sentence.

Finally, since we’re dealing with matrices, we can condense **steps two** through **six** in one formula to calculate the outputs of the self-attention layer.

<center>
<img src="https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png" width="500" alt="image">
</center>

## Multi Headed Attention 


Must Watch :fire:

<center>

<figure class="video_container">
  <iframe src="https://www.youtube.com/watch?v=N9AGY-Z6tbM&t=1s" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>

</center>

_*If above link is broken, click [here](https://www.youtube.com/watch?v=N9AGY-Z6tbM&t=1s)_

The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. This improves the performance of the attention layer in two ways:

1. It **expands** the model’s ability to `focus on different positions`. In the example above, $z_1$ contains a little bit of every other encoding, but it could be **dominated by** the the **actual word itself**.
2. It gives the attention layer multiple `representation subspaces`.
   1. With multi-headed attention, we maintain **separate** $Q_i/K_i/V_i$ weight matrices for **each head** resulting in different $Q_i/K_i/V_i$ matrices. As we did before, we multiply $X$ by the $W_i^Q/W_i^K/W_i^V$ matrices to produce $Q/K/V$ matrices. Here $i$ represents the $i^{th}$ head.
   2. In the original paper they used $8$ heads.

All the different heads give different $z_i$ matrices. Concat them and multiply with weight matrix $W^O$ (trained jointly with the model) to get the final $z$ matrix that captures information from all the attention heads. This final $z$ can be sent to the FFNN.

Summarizing all here it is:

<center>
<img src="https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png" width="800" alt="image">
</center>

## Positional Encoding

One thing that’s missing from the model as we have described it so far is a way **to account for the order of the words** in the input sequence.

To address this, the transformer **adds a vector** to each input embedding. These positional encoding vectors are also **learnt**. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence.

<center>
<img src="https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png" width="600" alt="image">
</center>

## The Residuals

One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a [layer-normalization](https://arxiv.org/abs/1607.06450) step.

If we’re to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

<center>
<img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" width="400" alt="image">
</center>

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:


## The Decoder Side

Now that we’ve covered most of the concepts on the encoder side, we basically know how the components of decoders work as well. But let’s take a look at how they work together.


![image](https://jalammar.github.io/images/t/transformer_decoding_1.gif)


:paperclip: **Reference:**

- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) :fire: :fire: 
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) :fire: :fire: 

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Transformer - Deeper Understanding

The following content has been borrowed from
[D2L: Alex Smola, Transformer](https://d2l.ai/chapter_attention-mechanisms/transformer.html) for educational purpose.

Let’s recap the pros and cons of CNN and RNN:

- **CNNs** are **easy to parallelize** at a layer but cannot capture the variable-length sequential dependency very well.
- **RNNs** are able to capture the long-range, variable-length sequential information, but suffer from inability to parallelize within a sequence.

To combine the advantages from both CNNs and RNNs, [Vaswani et al., 2017](https://d2l.ai/chapter_references/zreferences.html#vaswani-shazeer-parmar-ea-2017) designed a novel architecture using the `attention mechanism`. This architecture, which is called as **Transformer**, achieves 
1. `Parallelization` by capturing recurrence sequence with `attention` 
2. At the same time `encodes` each item’s `position` in the sequence. 

As a result, Transformer leads to a compatible model with significantly shorter training time.

Similar to the `seq2seq` model, Transformer is also based on the `encoder-decoder` architecture. However, Transformer differs to the former by 
1. **Replacing the recurrent layers** in seq2seq with **multi-head attention layers**
2. Incorporating the `position-wise` information through **position encoding**
3. Applying **layer normalization**. 

We compare Transformer and seq2seq side-by-side in the below figure

<center>
<img src="https://d2l.ai/_images/transformer.svg" width="600" alt="image">
</center>


Overall, these two models are similar to each other: 
- The source sequence embeddings are fed into $n$ repeated blocks. The outputs of the last block are then used as **attention memory** for the decoder. 
- The target sequence embeddings are similarly fed into $n$ repeated blocks in the decoder, and the final outputs are obtained by applying a dense layer with vocabulary size to the last block’s outputs.

On the flip side, Transformer differs from the seq2seq with attention model in the following:

- **Transformer block:** a recurrent layer in seq2seq is replaced by a Transformer block. This block contains a `multi-head attention layer` and a network with two `position-wise` feed-forward network layers for the **encoder**. For the **decoder**, another `multi-head attention` layer is used to take the encoder state.
- **Add and norm:** the inputs and outputs of both the multi-head attention layer or the position-wise feed-forward network, are processed by two `add and norm` layer that contains a **residual structure** and a `layer normalization` layer.
- **Position encoding:** Since the self-attention layer does not distinguish the item order in a sequence, a `positional encoding` layer is used to add sequential information into each sequence item.

## Multi-Head Attention

Before the discussion of the multi-head attention layer, let’s quick express the `self-attention` architecture. The self-attention model is a normal attention model, with its `query`, its `key`, and its `value` being copied exactly the same from each item of the sequential inputs. As we illustrate in the below figure, self-attention outputs a same-length sequential output for each input item. Compared with a recurrent layer, output items of a self-attention layer can be **computed in parallel** and, therefore, it is easy to obtain a highly-efficient implementation.

<center>
<img src="https://d2l.ai/_images/self-attention.svg" alt="image" width="400">
</center>


The `multi-head attention` layer consists of $h$ **parallel self-attention layers**, each one is called a `head`. For each head, before feeding into the attention layer, we project the `queries`, `keys`, and `values` with three dense layers with hidden sizes $p_q$, $p_k$, and $p_v$, respectively. The outputs of these $h$ attention heads are concatenated and then processed by a final dense layer.

![image](https://d2l.ai/_images/multi-head-attention.svg)


Assume that the dimension for a `query`, a `key`, and a `value` are $d_q$, $d_k$, and $d_v$, respectively. Then, for each head $i=1, \dots ,h$, we can train learnable parameters $W^{(i)}_q \in \mathbb{R}^{p_q \times d_q}$, $W^{(i)}_k \in \mathbb{R}^{p_k \times d_k}$ , and $W^{(i)}_v \in \mathbb{R}^{p_v \times d_v}$. Therefore, the output for each head is

<center>

$
o^{(i)} = attention(W^{(i)}_q q,W^{(i)}_k k, W^{(i)}_v v )
$

</center>


where **attention** can be any attention layer, such as the `DotProductAttention` and `MLPAttention`. 

After that, the output with length $p_v$ from each of the $h$ attention heads are concatenated to be an output of length $h p_v$, which is then passed the final dense layer with $d_o$ hidden units. The weights of this dense layer can be denoted by $W_o \in \mathbf{R}^{d_o \times hp_v}$. As a result, the multi-head attention output will be

<center>

$
\mathbf o = \mathbf W_o 
\begin{bmatrix}
\mathbf o^{(1)}\\\vdots\\\mathbf o^{(h)}
\end{bmatrix}
$

</center>

## Position-wise Feed-Forward Networks

Another key component in the Transformer block is called position-wise feed-forward network (FFN). It accepts a $3$-dimensional input with shape (`batch size`, `sequence length`, `feature size`). The position-wise FFN consists of two dense layers that applies to the last dimension. Since the same two dense layers are used for each position item in the sequence, we referred to it as position-wise. Indeed, it is **equivalent to applying two** $1 \times 1$ convolution layers.

## Add and Norm

Besides the above two components in the Transformer block, the `add and norm` within the block also plays a key role to **connect** the `inputs` and `outputs` of other layers `smoothly`. To explain, we add a layer that contains a **residual structure** and a **layer normalization** after both the multi-head attention layer and the position-wise FFN network. 

Layer normalization is similar to batch normalization in Section 7.5. One difference is that the mean and variances for the layer normalization are calculated along the last dimension, e.g `X.mean(axis=-1)` instead of the first batch dimension, e.g., X.mean(axis=0). 

Layer normalization prevents the range of values in the layers from changing too much, which means that **faster training** and **better generalization** ability.

## Positional Encoding

Unlike the recurrent layer, both the multi-head attention layer and the position-wise feed-forward network compute the output of each item in the sequence `independently`. This feature enables us **to parallelize the computation**, but it **fails to model the sequential information** for a given sequence. 

To better capture the sequential information, the Transformer model uses the **positional encoding to maintain the positional information** of the input sequence.

To explain, assume that $X \in \mathbb{R}^{l×d}$
is the embedding of an example, where $l$ is the `sequence length` and $d$ is the `embedding size`. This positional encoding layer encodes $X$’s position $P \in \mathbb{R}^{l \times d}$ and outputs $P+X$.


The position $P$ is a $2$-D matrix, where 
- $i$ refers to the **order in the sentence**
- $j$ refers to the **position along the embedding vector dimension**. 

In this way, each value in the origin sequence is then maintained using the equations below:

<center>

$
P_{i, 2j} = \sin(i/10000^{2j/d})
$

</center>

<center>

$
P_{i, 2j+1} = \cos(i/10000^{2j/d})
$

</center>

for $i=0,\ldots, l-1$ and $j=0,\ldots,\lfloor(d-1)/2\rfloor$.

<center>
<img src="https://d2l.ai/_images/positional_encoding.svg" alt="image" width="500">
</center>

## Encoder

Armed with all the essential components of Transformer, let’s first build a Transformer encoder block. This encoder contains a 
1. `Multi-head attention` layer
2. A `position-wise` feed-forward network
3. Two `add and norm` connection blocks. 

As shown in the [code (section 10.3.5)](https://d2l.ai/chapter_attention-mechanisms/transformer.html), for both of the attention model and the positional FFN model in the `EncoderBlock`, their outputs’ dimension are equal to the `num_hiddens`. 

This is due to the nature of the residual block, as we need to add these outputs back to the original value during `add and norm`.

## Decoder

The Transformer **decoder** block looks similar to the Transformer encoder block. However, besides the two sub-layers (1. the `multi-head attention` layer and 2. the `positional encoding` network), the decoder Transformer block contains a **third sub-layer**, which applies **multi-head attention on the output of the encoder stack**. 

Similar to the Transformer encoder block, the Transformer decoder block employs `add and norm`, i.e., the residual connections and the layer normalization to connect each of the sub-layers.

To be specific, at time-step $t$, assume that $x_t$ is the current input, i.e., the `query`. As illustrated in the below figure, the `keys` and `values` of the self-attention layer consist of the current query with all the past queries $x_1 \ldots ,x_{t−1}$.

<center>
<img src="https://d2l.ai/_images/self-attention-predict.svg" width="400" alt="image">
</center>

During training, the output for the $t$-`query` could observe all the previous `key-value` pairs. It results in an different behavior from prediction. Thus, during prediction we can **eliminate the unnecessary information by specifying** the valid length to be $t$ for the $t^{th}$ query.

**Reference:**

- [D2L: Alex Smola, Transformer](https://d2l.ai/chapter_attention-mechanisms/transformer.html)
- [IMP: The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How does Attention Work ?

<center>
<img src="/assets/images/image_20_Attention_7.png" alt="image" width="500">
</center>

- At each time step we come up with a distribution on the input words. 

- In the above image at time $t_3$, input `ja raha` gets the attention and it's corresponding distributoin is $[0,0,0.5, 0.5, 0]$. This helps in modelling because now input `jaa raha` is linked to output `going` with more attention as in the attention vector other words got $0$. 
- This distribution tells us how much **attention** to pay to some part of the input but not all parts.
- At each time step we should feed this relevant information (i.e. encoding of relevant words in the form of attention distribution) to the decoder.

In reality this distribution is not available beforehand and is **learnt** through the model. 

[_reference: Prof. Mikesh, Padhai Lecture, Encoder-Decoder_]

 >> The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important.

 When reading a text, you always focus on the word you read but at the same time your mind still holds the important keywords of the text in memory in order to provide context.

An attention-mechanism works similarly for a given sequence. For our example with the human Encoder and Decoder, imagine that instead of only writing down the translation of the sentence in the imaginary language, the Encoder also writes down keywords that are important to the semantics of the sentence, and gives them to the Decoder in addition to the regular translation. Those new keywords make the translation much easier for the Decoder because it knows what parts of the sentence are important and which key terms give the sentence context.

## Attention

The basic idea: each time the model predicts an output word, it only uses parts of an input where the most relevant **information is concentrated** instead of an entire sentence. In other words, it only pays **attention to some input words**. Let’s investigate how this is implemented.


<center>
<img src="https://miro.medium.com/max/710/1*9Lcq9ni9aujScFYyyHRhhA.png" width="600">
</center>

> An illustration of the attention mechanism (RNNSearch) proposed by [Bahdanau, 2014]. Instead of converting the entire input sequence into a single context vector, we create a separate context vector for each output (target) word. These vectors consist of the weighted sums of encoder’s hidden states.


Encoder works as usual, and the difference is only on the decoder’s part. As you can see from a picture, the **decoder’s hidden state is computed with a context vector**, the previous output and the previous hidden state. But now we use not a single context vector $c$, but a **separate context vector** $c_i$ for **each target word**.

**References:**

- [Paper: Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
- [Visualizing A Neural Machine Translation Model by Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Important Blog](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- [Imp: Attention in NLP](https://medium.com/@joealato/attention-in-nlp-734c6fa9d983)
- [Imp: Attention and Memory in NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

## Encoder-Decoder with Attention

<center>
<img src="/assets/images/image_20_Attention_8.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_20_Attention_9.png" alt="image" width="500">
</center>


In the Decoder equation their is a correction for $e_{jt}$. Instead of $W_{attn}s_t$, it will be $W_{attn}s_{t-1}$


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


## More on Attention


<center>
<img src="/assets/images/image_20_Attention_1.png" alt="image" width="500">
</center>

- Attention provides a solution to the `bottleneck problem`.
- `Core idea`: on each step of the decoder, use direct connection to the encoderto focus on a particular part of the source sequence


<center>
<img src="/assets/images/image_20_Attention_2.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_20_Attention_3.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_20_Attention_4.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_20_Attention_5.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_20_Attention_6.png" alt="image" width="500">
</center>



**Resource:**

- [cs224n-2019-lecture08-nmt](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


## What is Transformer and it's pros and cons?

From the Author:

>> In paper “Attention Is All You Need”, we introduce the Transformer, a novel neural network architecture based on a self-attention mechanism that we believe to be particularly well suited for language understanding.  

Natural Language Understanding (NLU):  language modeling, machine translation and question answering

- Transformer outperforms both recurrent and convolutional models on academic English to German and English to French translation benchmarks. 
- On top of higher translation quality, the Transformer requires less computation to train and is a much better fit for modern machine learning hardware, speeding up training by up to an order of magnitude.

The paper ‘Attention Is All You Need’ describes transformers and what is called a sequence-to-sequence architecture. Sequence-to-Sequence (or Seq2Seq) is a neural net that transforms a given sequence of elements, such as the sequence of words in a sentence, into another sequence.

<center>
<img src="https://cdn-images-1.medium.com/max/800/1*BHzGVskWGS_3jEcYYi6miQ.png" width="500">
</center>

- One interesting point is, even if it's used for **seq2seq generation**, **but there is no** `recurrence` part inside the model like the  `vanilla rnn` or `lstm`.
- So one slight but important part of the model is the **positional encoding** of the different words. Since we have no recurrent networks that can remember how sequences are fed into a model, we need to somehow give every word/part in our sequence a relative position since a sequence depends on the order of its elements. These positions are added to the embedded representation (n-dimensional vector) of each word. 

**Pros:**

- Faster learning. More GPU efficient unlike the `vanilla rnn`

The animation below illustrates how we apply the Transformer to machine translation. Neural networks for machine translation typically contain an encoder reading the input sentence and generating a representation of it. A decoder then generates the output sentence word by word while consulting the representation generated by the encoder.

<center>
<img src="https://3.bp.blogspot.com/-aZ3zvPiCoXM/WaiKQO7KRnI/AAAAAAAAB_8/7a1CYjp40nUg4lKpW7covGZJQAySxlg8QCLcBGAs/s640/transform20fps.gif" width="600">
</center>

**References:**

- [Paper: Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Google AI Blog](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- [Important Blog](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- [Important: The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) 

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Attention is All You Need


<center>
<img src="/assets/images/image_38_attention_1.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_38_attention_2.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_38_attention_3.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_38_attention_4.png" alt="image" width="500">
</center>

## Positional Encoding

<center>
<img src="/assets/images/image_38_attention_5.png" alt="image" width="500">
</center>


## Transformer Machine Translation

<center>
<img src="/assets/images/image_38_attention_6.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_38_attention_7.png" alt="image" width="500">
</center>


## Transformer Language Pre-training


<center>
<img src="/assets/images/image_38_attention_8.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_38_attention_9.png" alt="image" width="500">
</center>

## Multi-head attention


<center>
<img src="/assets/images/image_38_attention_10.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_38_attention_11.png" alt="image" width="500">
</center>

_Read the excellent slides from the below reference_

**Reference:**

- [CS.Toronto.Lecture16](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec16.pdf)
- [Excellent Slide by Alex Smola](https://icml.cc/media/Slides/icml/2019/halla(10-09-15)-10-15-45-4343-a_tutorial_on.pdf)
- [Good summery (even UMLFit)](https://www.student.cs.uwaterloo.ca/~mjksmith/DSC_Transformer_Presentation.pdf)
- [Imp slides on Transformer](https://www.slideshare.net/DaikiTanaka7/attention-is-all-you-need-127742932)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


# Machine Translation, seq2seq model

## Encoder-Decoder Model

**Approach 1:**

$h_T$ passed to the $s_0$ of the decoder only.

<center>
<img src="/assets/images/image_10_seq2seq_1.png" alt="image" width="500">
</center>

**Approach 2:**

$h_T$ passed to the every state $s_i$ of decoder.

<center>
<img src="/assets/images/image_10_seq2seq_2.png" alt="image" width="500">
</center>


- Padhai, DL course, Prof. Mikesh, IIT M, Lecture: Encoder Decoder

## Neural Machine Translation (NMT)

- `Neural Machine Translation` (NMT)is a way to do Machine Translation with a `single neural network`.
- The neural network architecture is called sequence-to-sequence(aka `seq2seq`) and it involves two RNNs.


<center>
<img src="/assets/images/image_19_NMT_1.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_19_NMT_2.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_19_NMT_3.png" alt="image" width="500">
</center>

<center>
<img src="/assets/images/image_19_NMT_4.png" alt="image" width="500">
</center>


<center>
<img src="/assets/images/image_19_NMT_5.png" alt="image" width="500">
</center>

**Resource:**

- [cs224n-2019-lecture08-nmt](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)


----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
