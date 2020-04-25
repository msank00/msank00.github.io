---
layout: post
title:  "Deep Learning: Natural Language Processing (Part 2)"
date:   2019-07-13 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

Quick Refresher: Neural Network in NLP

# What is Syntactic and Semantic analysis?

Syntactic analysis (syntax) and semantic analysis (semantic) are the two primary techniques that lead to the understanding of natural language. Language is a set of valid sentences, but what makes a sentence valid? Syntax and semantics.

- **Syntax** is the grammatical structure of the text 
- **Semantics** is the meaning being conveyed. 

A sentence that is syntactically correct, however, is not always semantically correct. 
- **Example,** “cows flow supremely” is grammatically valid (subject — verb — adverb) but it doesn't make any sense.

## SYNTACTIC ANALYSIS

Syntactic analysis, also referred to as syntax analysis or parsing. 

> It is the process of analyzing natural language with the rules of a formal grammar. 

Grammatical rules are applied to categories and groups of words, not individual words. Syntactic analysis basically assigns a semantic structure to text.

For example, a sentence includes a subject and a predicate where the subject is a noun phrase and the predicate is a verb phrase. Take a look at the following sentence: “The dog (noun phrase) went away (verb phrase).” Note how we can combine every noun phrase with a verb phrase. Again, it's important to reiterate that a sentence can be syntactically correct but not make sense.

## SEMANTIC ANALYSIS

The way we understand what someone has said is an unconscious process relying on our intuition and knowledge about language itself. In other words, the way we understand language is heavily based on meaning and context. Computers need a different approach, however. The word `semantic` is a **linguistic term** and means `related to meaning or logic`.

:paperclip: **Reference:**

- [Blog](https://builtin.com/data-science/introduction-nlp)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# What is Natural Language Underrstanding?

It can be easily understood by the syllabus topic of the course CS224U by Standford. Though over the years the definition has been changed.

**2012**

- WordNet
- Word sense disambiguation
- Vector-space models
- Dependency parsing for NLU
- Relation extraction
- Semantic role labeling
- Semantic parsing
- Textual inference
- Sentiment analysis
- Semantic composition withvectors
- Text segmentation
- Dialogue

**2020**

- Vector-space models
- Sentiment analysis
- Relation extraction
- Natural LanguageInference
- Grounding
- Contextual wordrepresentations
- Adversarial testing
- Methods and metrics

:paperclip: **Reference:**

- [CS224u course website](https://web.stanford.edu/class/cs224u/)
- [CS224u slide](https://web.stanford.edu/class/cs224u/materials/cs224u-2020-intro-handout.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Motivation: Why Learn Word Embeddings?

Image and audio processing systems work with rich, high-dimensional datasets encoded as vectors of the individual raw pixel-intensities for image data, or e.g. power spectral density coefficients for audio data. For tasks like object or speech recognition we know that all the information required to successfully perform the task is encoded in the data (because humans can perform these tasks from the raw data). However, **natural language processing** systems **traditionally treat words as discrete atomic symbols**, and therefore `cat` may be represented as `Id537` and `dog` as `Id143`. These encodings are **arbitrary, and provide no useful information** to the system regarding the relationships that may exist between the individual symbols. This means that the model can leverage very little of what it has learned about ‘cats’ when it is processing data about ‘dogs’ (such that they are both animals, four-legged, pets, etc.). Representing words as unique, discrete ids furthermore leads to **data sparsity**, and usually means that we may need more data in order to successfully train statistical models. Using vector representations can overcome some of these obstacles.

**Vector space models** (**VSM**s) represent (`embed`) words in a continuous vector space where **semantically similar** (meaningfully similar) words are mapped to nearby points (`are embedded nearby each other`). VSMs have a long, rich history in NLP, but all methods depend in some way or another on the **Distributional Hypothesis**, 

> which states that words that appear in the same contexts share semantic meaning. 

The different approaches that leverage this principle can be divided into two categories: 

- **Count-based methods** (e.g. Latent Semantic Analysis),
- **Predictive methods** (e.g. neural probabilistic language models like `word2vec`).

This distinction is elaborated in much more detail by [Baroni et al. 2014](https://www.aclweb.org/anthology/P14-1023.pdf) in his great paper **Don’t count, predict!**, where he compares the `context-counting` vs `context-prediction`. In a nutshell: 
  - **Count-based** methods first compute the statistics of how often some word co-occurs with its neighbor words in a large text corpus, and then map these count-statistics down to a small, dense vector for each word. 
  - **Predictive models** `directly try to predict` a word from its neighbors in terms of learned small, dense embedding vectors (considered parameters of the model).

**Word2vec** is a particularly **computationally-efficient predictive model** for learning word embeddings from raw text. It comes in two flavors, the Continuous Bag-of-Words model (CBOW) and the Skip-Gram model. 
  - Algorithmically, these models are similar, except that CBOW predicts target words (e.g. ‘mat’) from source context words (‘the cat sits on the’), while the skip-gram does the inverse and predicts source context-words from the target words. 

This inversion might seem like an arbitrary choice, but statistically it has different effect.
- **CBOW** **smoothes over a lot of the distributional information** (by treating an entire context as one observation). For the most part, this turns out to be a **useful thing for smaller datasets**. 
- **Skip-gram** treats each context-target pair as a new observation, and this tends to **do better when we have larger datasets**. 

:paperclip: **Reference:**

- [Tensorflow: Vector Representations of Words](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/tutorials/word2vec/index.md) :fire:
- [Baroni et al. 2014](https://www.aclweb.org/anthology/P14-1023.pdf)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to design a basic vector space model?

- [Youtube](https://www.youtube.com/watch?v=gtuhPq0Xyno&feature=youtu.be)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# What is PMI: Point-wise Mutual Information?

The idea behind the NLP algorithm is that of transposing words into a vector space, where each word is a D-dimensional vector of features. By doing so, we can compute some quantitative metrics of words and between words, namely their cosine similarity.

**Problem:** How to understand whether two (or more) words actually form a unique concept?

**Example:** Namely, consider the expression ‘social media’: both the words can have independent meaning, however, when they are together, they express a precise, unique concept.

Nevertheless, it is not an easy task, since if both words are frequent by themselves, their co-occurrence might be just a chance. Namely, consider the name ‘Las Vegas’: it is not that frequent to read only ‘Las’ or ‘Vegas’ (in English corpora of course). The only way we see them is in the bigram Las Vegas, hence it is likely for them to form a unique concept. On the other hand, if we think of ‘New York’, it is easy to see that the word ‘New’ will probably occur very frequently in different contexts. How can we assess that the co-occurrence with York is meaningful and not as vague as ‘new dog, new cat…’?

The answer lies in the **Pointwise Mutual Information (PMI)** criterion. The idea of PMI is that we want to 

> quantify the likelihood of co-occurrence of two words, taking into account the fact that it might be caused by the frequency of the single words. 

Hence, the algorithm computes the ($\log$) probability of co-occurrence scaled by the product of the single probability of occurrence as follows:


$$
PMI(w_a, w_b) = \log  \left( \frac{p(w_a, w_b)}{p(w_a) p(w_b)} \right) = \log \left( \frac{p(w_a, w_b)}{p(w_a)} \times \frac{1}{p(w_b)} \right) \\ = \log \left( p(w_b \vert w_a) \times \frac{1}{p(w_b)} \right) = \log \left( p(w_a \vert w_b) \times \frac{1}{p(w_a)} \right) 
$$

where $w_a$ and $w_b$ are two words.

Now, knowing that, when $w_a$ and $w_b$ are independent, their joint probability is equal to the product of their marginal probabilities, when the ratio equals 1 (hence the log equals 0), it means that the two words together don’t form a unique concept: they co-occur by chance.

On the other hand, if either one of the words (or even both of them) has a **low probability of occurrence if singularly considered**, but **its joint probability together with the other word is high**, it means that the two are likely to express a **unique concept**.

> PMI is the re-weighting of the entire count matrix

Let’s focus on the last expression. As you can see, it’s the conditional probability of $w_b$ given $w_a$ times $\frac{1}{p(w_b)}$. If $w_b$ and $w_a$ are independent, there is no meaning to the multiplication (it’s going to be zero times something). But if the conditional probability is larger than zero, $p(w_b \vert w_a) > 0$, then there is a meaning to the multiplication. How `important` is the event $W_b = w_b$? if $P(W_b = w_b) = 1$ then the event $W_b = w_b$ is not really important is it? think a die which always rolls the same number; there is no point to consider it. But, If the event $W_b = w_b$ is fairly rare → $p(w_b)$ is relatively low → $\frac{1}{p(w_b)}$ is relatively high → the value of $p(w_b \vert w_a)$ becomes much more important in terms of information. So that is the first observation regarding the PMI formula. 

:paperclip: **Reference:**

- [PMI](https://medium.com/dataseries/understanding-pointwise-mutual-information-in-nlp-e4ef75ecb57a)
- [understanding-pointwise-mutual-information-in-statistics](https://eranraviv.com/understanding-pointwise-mutual-information-in-statistics/)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# GloVe: Global Vectors

Read this amazing paper by [Pennington et al. (2014)](https://www.aclweb.org/anthology/D14-1162.pdf)

**Main Idea:**

> The objective is to learn vectors for words such that their **dot product is proportional to their probability of co-occurrence**.

- Can use the `Mittens` package. [PyPi](https://pypi.org/project/mittens/), [Paper](https://www.aclweb.org/anthology/N18-2034/)

<center>
<img src="/assets/images/image_40_nlu_01.png" alt="image" width="500">
</center>

- $w_i$: row embedding
- $w_k$: column embedding
- $X_{ik}$: Co-occurrence count
- $\log(P_{ik})$: log of co-occurrence probability
- $\log(X_{ik})$: log of co-occurrence count
- $\log(X_i)$: log of row probability

Their dot products are the 2 primary terms + 2 bias terms.

And the idea is that should be equal to (at-least proportional to) the log of the co-occurrence probability. 

Equation 6 tells that the dot product is equal to the difference of  of two log terms and if re-arrange them they looks very similar to **PMI** !! Where PMI is the re-weighting of the entire count matrix. 

## The Weighted GloVe objective


<center>
<img src="/assets/images/image_40_nlu_02.png" alt="image" height="250">
</center>


<center>
<img src="/assets/images/image_40_nlu_03.png" alt="image" width="300">
</center>

Weighted by the function $f()$. Which is `flatten`ing out and `rescale`ing the co-occurrence count $X_{ik}$ values.

Say the co-occurrence count vector is like this `v = [100 99 75 10 1]`. Then $f(v)$ is `[1.00 0.99 0.81 0.18 0.03]`.

## What's happening behind the scene (BTS)?

<center>
<img src="/assets/images/image_40_nlu_04.png" alt="image" width="600">
</center>

**Example:**

Word `wicked` and `gnarly` (positive slang) never co-occur. If you look at the left plot in the above image, then you see, what GLoVe does is, it pushes both `wicked` and `gnarly` **away from negative word** `terrible` and moves them **towards positive word** `awsome`. Because even if `wicked` and `gnarly` don't occur together, they have co-occurrence with positive word `awsome`. GloVe thus achieves this latent connection.   

**Note:** Glove transforms the `raw count` distribution into a `normal distribution` which is essential when you train deep-learning model using word-embedding as your initial layer. It's essential because the embedding values have constant `mean` and `variance` and this is a crucial part for training any deep-learning model. The weight values while passing through different layers should maintain their distribution. That's why GloVe does so well as an input to another system. 

:paperclip: **Reference:**

- [CS224U Slide](https://web.stanford.edu/class/cs224u/materials/cs224u-2020-vsm-handout.pdf) :pushpin:
- [CS224U Youtube](https://www.youtube.com/watch?v=pip8h9vjTHY&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=4) :pushpin:



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----


# Scaling up with Noise-Contrastive Training

Neural probabilistic language models are traditionally trained using the maximum likelihood (ML) principle to maximize the probability of the next word $w_t$ (for `target`) given the previous words $h$ (for `history`) in terms of a softmax function,

$$
P(w_t \vert h) = \text{softmax}(\text{score}(w_t, h)) = \frac{\exp ({ \text{score}(w_t, h) }) } {\sum_\text{Word w' in Vocab} \exp ({ \text{score}(w', h) }) }
$$

where $\text{score}(w_t, h)$ computes the compatibility of word $w_t$ with the context $h$ (a dot product is commonly used). We train this model by maximizing its log-likelihood on the training set, i.e. by maximizing

$$ J_\text{ML} = \log P(w_t \vert h) = \text{score}(w_t, h) - \log \left( \sum_{w' \in V} \exp ({ \text{score}(w', h) }) \right) $$

- $w'$ is a word
- $V$ is the vocabulary set

This yields a properly normalized probabilistic model for language modeling. However this is very **expensive**, because we need to compute and normalize each probability using the score for all other $V$ words $w'$ in the current context $h$, at every training step.

On the other hand, for feature learning in `word2vec` we do not need a full probabilistic model. The CBOW and skip-gram models are **instead trained using a binary classification objective** (logistic regression) to discriminate the real target words $w_t$ from $k$ imaginary (`noise`) words $\tilde w$, in the same context. We illustrate this below for a CBOW model. For skip-gram the direction is simply inverted.

Mathematically, the objective (for each example) is to maximize

$$J_\text{NEG} = \log Q_\theta(D=1 \vert w_t, h) + k \mathop{\mathbb{E}}{\tilde w \sim P_\text{noise}} \left[ \log Q_\theta(D = 0 \vert \tilde w, h) \right]$$

- where $Q_\theta(D=1 \vert w, h)$ is the binary logistic regression probability under the model of seeing the word $w$ in the context $h$ in the dataset $D$, calculated in terms of the learned embedding vectors $\theta$. 
- In practice the author approximates the expectation by drawing $k$ contrastive words (contrasting/different words) from the noise distribution (i.e. we compute a [Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration)).

This objective is maximized when the model assigns high probabilities to the real words, and low probabilities to noise words. Technically, this is called [Negative Sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), and there is good mathematical motivation for using this loss function: 
- The updates it proposes **approximate the updates of the softmax function in the limit**. 
- But computationally it is especially appealing because computing the loss function now **scales only with the number of noise words that we select** ($k$), and not all words in the vocabulary ($V$). 
- This makes it much **faster to train**. The author uses the very similar [noise-contrastive estimation](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf) (NCE) loss.

:paperclip: **Reference:**

- [TensorFlow: Vector Representations of Words](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/refs/heads/0.6.0/tensorflow/g3doc/tutorials/word2vec/index.md) :bomb: :fire: :rocket:
- [Paper: Distributed Representations of Words and Phrases
and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [Learning word embeddings efficiently with
noise-contrastive estimation](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


-----

# Exercise:

- UMLFit
- U-Net

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>