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
- E**xample,** “cows flow supremely” is grammatically valid (subject — verb — adverb) but it doesn't make any sense.

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

# How to design a basic vector space model?

- [Youtube](https://www.youtube.com/watch?v=gtuhPq0Xyno&feature=youtu.be)

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


Let’s focus on the last expression. As you can see, it’s the conditional probability of $w_b$ given $w_a$ times $\frac{1}{p(w_b)}$. If $w_b$ and $w_a$ are independent, there is no meaning to the multiplication (it’s going to be zero times something). But if the conditional probability is larger than zero, $p(w_b \vert w_a) > 0$, then there is a meaning to the multiplication. How `important` is the event $W_b = w_b$? if $P(W_b = w_b) = 1$ then the event $W_b = w_b$ is not really important is it? think a die which always rolls the same number; there is no point to consider it. But, If the event $W_b = w_b$ is fairly rare → $p(w_b)$ is relatively low → $\frac{1}{p(w_b)}$ is relatively high → the value of $p(w_b \vert w_a)$ becomes much more important in terms of information. So that is the first observation regarding the PMI formula. 

:paperclip: **Reference:**

- [PMI](https://medium.com/dataseries/understanding-pointwise-mutual-information-in-nlp-e4ef75ecb57a)
- [understanding-pointwise-mutual-information-in-statistics](https://eranraviv.com/understanding-pointwise-mutual-information-in-statistics/)

-----

# What is Siamese Network

A **twin neural network** (sometimes called a Siamese Network, though this term is frowned upon) is an artificial neural network that **uses the same weights** while working in tandem (having two things arranged one in front of the other) on two different input vectors to compute comparable output vectors. Often one of the output vectors is precomputed, thus forming a baseline against which the other output vector is compared. This is similar to comparing fingerprints but can be described more technically as a distance function for `locality-sensitive hashing`.

It is possible to make a kind of structure that is functional similar to a siamese network, but implements a slightly different function. This is typically used for comparing similar instances in different type sets.

Uses of similarity measures where a twin network might be used are such things as 

- Recognizing handwritten checks
- Automatic detection of faces in camera images
- Matching queries with indexed documents.

## Learning

Learning in twin networks can be done with `triplet loss` or `contrastive loss`.

### Triplet Loss

Triplet loss is a loss function for artificial neural networks where a baseline (`anchor`) input is compared to a positive (`truthy`) input and a negative (`falsy`) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized. [wiki](https://en.wikipedia.org/wiki/Triplet_loss)

- minimize distance(baseline,truth)
- maximize distance(baseline,false) 

The negative (false) vector will force learning in the network, while the positive vector (truth) will act like a regularizer.

### Predefined metrics, Euclidean distance metric

The common learning goal is to minimize a distance metric for similar objects and maximize for distinct ones. This gives a loss function like 

$$
\delta(x^i, x^j)=\left\{
                \begin{array}{ll}
                  min \vert\vert f(x^i) - f(x^j) \vert\vert, i \ne j\\
                  max \vert\vert f(x^i) - f(x^j) \vert\vert, i = j
                \end{array}
              \right.
$$


### Twin Networks for Object Tracking

Twin networks have been used in object tracking because of its unique two tandem inputs and **similarity measurement**. In object tracking, one input of the twin network is user pre-selected exemplar image, the other input is a larger search image, which twin network's job is to locate exemplar inside of search image. By measuring the similarity between exemplar and each part of the search image, a map of similarity score can be given by the twin network. 

Furthermore, using a Fully Convolutional Network, the process of computing each sector's similarity score can be replaced with only one cross correlation layer.


**Reference:**

- [wiki: Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network)
- [wiki: triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Exercise:

- UMLFit
- U-Net

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>