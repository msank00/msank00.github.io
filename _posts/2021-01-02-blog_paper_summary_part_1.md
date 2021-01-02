---
layout: post
title:  "Paper Summary: Part 1"
date:   2021-01-02 00:00:10 -0030
categories: papersummary
mathjax: true
---



# Content

1. TOC
{:toc}
---


# Improving Zero-Shot Translation by Disentangling Positional Information

- From FacebookAI   

## Multilingual Neural Machine Translation (MultiNMT)

Idea: Say you have paied translation corpus `english - french`, `german - english` and you have trained a model jointly on the corpus. Now your target is to translate `german` to `french` directly using the trained model. See, in the training corpus to `german - french` available. So this is a form of `zero-shot-learning`. Zero because, it's unseen in training corpus. 

## Disentangling Positional Information

Recent approaches are based on `Transformer` architecture, where `positional encoding` is heavily used to retain the positional information of the token. This makes the translation language specific. 

> :bulb: ...but ideal MultiNMT needs language agnostic behavior... 

**Two potential causes of this positional correspondence:**

1. Residual connections
2. Encoder self-attention alignment
   1. Via the self-attention transform,  each position is a weighted sum from all input positions. 

So if current Transformer architecture is used blindly, it will give poor performance. 

:six_pointed_star: **Solution:** Relax this strong positional architecture and make the architecture flexible for MultiNMT. Relax this structural constraint and offer the model some freedom of word reordering in the encoder already.

**Transformaer Architecture**

<center>

<img src="https://lilianweng.github.io/lil-log/assets/images/transformer-encoder.png" width="300">

</center>


<center>

<img src="https://lilianweng.github.io/lil-log/assets/images/transformer.png" width="600">

</center>

<center>
<img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png" width="400" alt="image">
</center>


_[Transformer MUST READ](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#a-family-of-attention-mechanisms)_ :rocket:


**Author's modification**

![image](/assets/images/image_42_papersummary_1.png)

_the above self attention (SA) schematic diagram is similar to the right **zoom in** image of the above original transformer architecture image above_

- **Modify Residual Connection:** Author achieve considerable gains on zero-shot translation quality by only **removing residual connections** once in a middle encoder layer.
  - Set **one encoder layer free** from these constraints, so that it could create its own output ordering instead of always following a one-to-one mapping with its input.
-  **Position-Based Self-Attention Query:**



> Zero-shot inference relies on a model’s general-izability to conditions unseen in training.


**Open Question:**

- :smirk: No mathematical support, emperical behavior
- :thinking: Why only one encoder layer is relaxed? Any reason? 
  - From author: to minimize the impact on the model architecture and ensure gradient flow, we limit this change to only one encoder layer,and only its multihead attention laye

**Reference:**

- [arxiv paper](https://arxiv.org/pdf/2012.15127.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>