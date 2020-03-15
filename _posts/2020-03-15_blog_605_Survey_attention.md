---
layout: post
title:  "Survey: Attention"
date:   2020-03-15 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

# Introduction

>  The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important.

Neural networks, in particular recurrent neural networks (RNNs), are now at the core of the leading approaches to language understanding tasks such as **language modeling, machine translation and question answering**. In [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the authors introduce the **Transformer**, a novel neural network architecture based on a self-attention mechanism that we believe to be particularly well suited for language understanding.

In our paper, we show that the Transformer outperforms both recurrent and convolutional models on academic `English to German` and `English to French` translation benchmarks. On top of higher translation quality, the Transformer requires **less computation to train** and is a much better fit for modern machine learning hardware, speeding up training by up to an order of magnitude.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How does Attention Work ?


![image](/assets/images/image_20_Attention_7.png)

- At each time step we come up with a distribution on the input words. 

- In the above image at time $t_3$, input `ja raha` gets the attention and it's corresponding distributoin is $[0,0,0.5, 0.5, 0]$. This helps in modelling because now input `jaa raha` is linked to output `going` with more attention as in the attention vector other words got $0$. 
- This distribution tells us how much **attention** to pay to some part of the input but not all parts.
- At each time step we should feed this relevant information (i.e. encoding of relevant words in the form of attention distribution) to the decoder.

In reality this distribution is not available beforehand and is **learnt** through the model. 

[_reference: Prof. Mikesh, Padhai Lecture, Encoder-Decoder_]

 >> The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important.

 When reading a text, you always focus on the word you read but at the same time your mind still holds the important keywords of the text in memory in order to provide context.

An attention-mechanism works similarly for a given sequence. For our example with the human Encoder and Decoder, imagine that instead of only writing down the translation of the sentence in the imaginary language, the Encoder also writes down keywords that are important to the semantics of the sentence, and gives them to the Decoder in addition to the regular translation. Those new keywords make the translation much easier for the Decoder because it knows what parts of the sentence are important and which key terms give the sentence context.

**References:**

- [Paper: Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
- [Visualizing A Neural Machine Translation Model by Jay Alammar](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Important Blog](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- [Imp: Attention in NLP](https://medium.com/@joealato/attention-in-nlp-734c6fa9d983)
- [Imp: Attention and Memory in NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Encoder-Decoder with Attention

![image](/assets/images/image_20_Attention_8.png)
![image](/assets/images/image_20_Attention_9.png)

In the Decoder equation their is a correction for $e_{jt}$. Instead of $W_{attn}s_t$, it will be $W_{attn}s_{t-1}$


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


## More on Attention


![image](/assets/images/image_20_Attention_1.png)

- Attention provides a solution to the `bottleneck problem`.
- `Core idea`: on each step of the decoder, use direct connection to the encoderto focus on a particular part of the source sequence

![image](/assets/images/image_20_Attention_2.png)
![image](/assets/images/image_20_Attention_3.png)
![image](/assets/images/image_20_Attention_4.png)
![image](/assets/images/image_20_Attention_5.png)
![image](/assets/images/image_20_Attention_6.png)


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
<img src="https://cdn-images-1.medium.com/max/800/1*BHzGVskWGS_3jEcYYi6miQ.png" width="600">
</center>

- One interesting point is, even if it's used for **seq2seq generation**, **but there is no** `recurrence` part inside the model like the  `vanilla rnn` or `lstm`.
- So one slight but important part of the model is the **positional encoding** of the different words. Since we have no recurrent networks that can remember how sequences are fed into a model, we need to somehow give every word/part in our sequence a relative position since a sequence depends on the order of its elements. These positions are added to the embedded representation (n-dimensional vector) of each word. 

Pros:
- Faster learning. More GPU efficient unlike the `vanilla rnn`

**References:**

- [Paper: Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Google AI Blog](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- [Important Blog](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- [Important: The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) 

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----


# Machine Translation, seq2seq model

## Encoder-Decoder Model

**Approach 1:**

$h_T$ passed to the $s_0$ of the decoder only.

![image](/assets/images/image_10_seq2seq_1.png)

**Approach 2:**

$h_T$ passed to the every state $s_i$ of decoder.

![image](/assets/images/image_10_seq2seq_2.png)

- Padhai, DL course, Prof. Mikesh, IIT M, Lecture: Encoder Decoder

## Neural Machine Translation (NMT)

- `Neural Machine Translation` (NMT)is a way to do Machine Translation with a `single neural network`.
- The neural network architecture is called sequence-to-sequence(aka `seq2seq`) and it involves two RNNs.

![image](/assets/images/image_19_NMT_1.png)
![image](/assets/images/image_19_NMT_2.png)
![image](/assets/images/image_19_NMT_3.png)
![image](/assets/images/image_19_NMT_4.png)
![image](/assets/images/image_19_NMT_5.png)


**Resource:**

- [cs224n-2019-lecture08-nmt](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)


----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
