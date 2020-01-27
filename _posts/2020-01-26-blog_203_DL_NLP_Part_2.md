---
layout: post
title:  "Blog 203: Deep Learning: Natural Language Processing (Part 2)"
date:   2020-01-26 00:06:31 +0530
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

Quick Refresher: Neural Network in NLP

# What is tf-idf?

**Term Frequency:** we assign to each term in a document a weight for that
term, that depends on the number of occurrences of the term in the document. We would like to compute a score between a `query term` $t$ and a `document` $d$, based on the weight of $t$ in $d$. The simplest approach is to assign the weight to be equal to the number of occurrences of term t in document d. This weighting scheme is referred to as `term frequency` and is denoted $tf_{t,d}$,
with the subscripts denoting the term and the document in order.

**Inverse Document Frequency:**

Raw term frequency as above suffers from a critical problem: all terms are considered equally important when it comes to assessing relevancy on a query.

> Rare term is more informative than the common term.

A mechanism is introduced for attenuating the effect of terms that occur too often. The idea would be to reduce the tf weight of a term by a factor that grows with its collection frequency. So it is more commonplace to use for this purpose, the `document frequency` $df_t$, defined to be the number of documents in the collection that contain a term $t$. We then define the inverse document frequency ($idf$) of a term $$ as follows

$$idf_t = \log \frac{N}{df_t}$$

Finally, The $tf\_idf$ weighting scheme assigns to term $t$ a weight in document $d$ given by:

$$tf\_idf_{t,d} = tf_{t,d} * idf_t$$

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Tf-Idf fails in document classification/clustering? How can you improve further?

 TF-IDF is supposed to be a `weighting scheme` that puts more weight on more relevant keywords. And by design, it chooses those keywords as relevant which are uncommon or rare in the corpus. 

The main disadvantages of using tf/idf is that it clusters documents which have `more similar uncommon keywords`. So it's only good to identify near identical documents. For example consider the following sentences:

**Example 1:**

1. The `website` `Stackoverflow` is a nice place. 
2. `Stackoverflow` is a `website`.


Two sentences from Example 1 will likely by clustered together with a reasonable threshold value since they share a lot of keywords. But now consider the following two sentences in example 2:

**Example 2:**

1. The website `Stackoverflow` is a nice place. 
2. I visit `Stackoverflow` regularly.

Now by using tf/idf the clustering algorithm will fail miserably because they only share one keyword even though they both talk about the same topic.

And also for very short documents, like tweet, it's difficult to rely on tf-idf score. 

So one can try: 

- LSI - Latent Semantic Indexing
- LDA - Latent Drichlet Allocation
- NMF - Non-negative Matrix Factorization

Currently the better solution is go with deep learning based approach. In simple from first convert the word into vector representation using any pre-trained model. And feed those vector representation in deep learning architecture.

Simple:

- Embedding Layer
- LSTM
- FC

This sort of architecture now outperforms all the traditional document clustering system.

**Reference:**

- [better-text-documents-clustering](https://stackoverflow.com/questions/17537722/better-text-documents-clustering-than-tf-idf-and-cosine-similarity)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How do you perform text classification? [ML Based Approach]

There are many ways to do it and those methods have evolved over time. 

- Naive Bayes (Count Based Approach)
- Language Modelling (ML technique)
  - N-gram approach
  - LDA
  - LSI
- Language Modelling (DL Technique)
  - LSTM
  - RNN
  - Encoder Decoder  

_TODO: Review the answer_

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Naive Bayes Classification:

We want to maximize the class ($c$) probability given the document $d$

$$
\hat c = argmax_{c \in C} P(c \vert d) = argmax_{c \in C} \frac{P(d \vert c)P(c)}{P(d)} 
$$

$$
\hat c = argmax_{c \in C} P(c \vert d) = argmax_{c \in C} P(d \vert c)P(c)
$$

Where $P(d \vert c)$, **likelihood** of the data and $P(c)$ is the **prior**.

Now the document $d$ can be seen as a collection of features $f_1, \dots ,f_n$. Therefore we can write

$$
P(d \vert c) = P(f_1, \dots ,f_n \vert c)
$$

Unfortunately, the above equation is still too hard to compute directly: without some simplifying assumption. ANd here comes the great Naive Bayes assumption which is the culprit of such naming **naive**.

- Assumption 1: bag of words - i.e features $f_1, \dots ,f_n$ only encode word identity bout not their position.
- **Naive-Bayes Assumption**: The conditional independence assumption that the probabilities $P(f_i \vert c)$ are independent given the class $c$. And thus:

$$
P(d \vert c) = P(f_1, \dots ,f_n \vert c) = \Pi_{i=1}^n P(f_i \vert c)
$$


$$
\hat c = argmax_{c \in C} P(c \vert d) = argmax_{c \in C} P(c) * \Pi_{i=1}^n P(f_i \vert c)
$$

Now again multiplication of probabilities. So again $\log()$ is at our rescue. 

$$
\hat c =  argmax_{c \in C} \log P(c) + \Sigma_{i=1}^n \log P(f_i \vert c)
$$

The beauty of the above equation is, in the log-space it's computing the predicted class as the liner combination of the input features $f_i \in d$. And also in the log-space the `prior` $P(c)$ somewhat `acts as regularizer`. 

**How to train the model?**

How can we learn the probabilities $P(c)$ and $P(f_i \vert c)$? Let’s first consider the maximum likelihood estimate. We’ll simply use the frequencies in the data.

To learn the probability $P( f_i \vert c)$, we’ll assume a feature is just the existence of a word in the document’s `bag of words`, and so we’ll want $P(w_i \vert c)$.

We will use smoothing technique to avoid any zero-count problem.

**Remember:** Naive-Bayes is an Generative Classifier where as Logistic-Regression is an Discriminative Classifier. 



**Reference:**

- Book: Speech and Language Processing, Jurafski, ch 4 



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Discuss more on Language Model

In language model we assign a probability to each possible next word. The same model will also serve to assign probability to an entire sentence. Assigning probabilities to sequence of words is very important.

- Speech recognition
- Spelling correction
- Grammatical error correction
- Machine translation

One way to estimate probability is: relative frequency count.

Say sentence $S = w_1 w_2 w_3$ where $w_i$ are words. Then 

$$P(w_3 \vert w_1 w_2) = \frac{C(w_1 w_2 w_3)}{C(w_1 w_2)}$$

i.e count number of times $w_3$ is used followed by $w_1 w_2$: $C(w_1 w_2 w_3)$. And divide this by how many times $w_1 w_2$ appear together $C(w_1 w_2)$.

However count based methods have flaws. New sentences are created very often and we won’t always be able to count entire sentences. For this reason, we’ll need to introduce cleverer ways of estimating the probability of a word w given a history h, or the probability of an entire word sequence W.

So we can compute the probabilities of the entire sentence $S=w_1 \dots w_n$ like this:


$$
P(x_1 \dots x_n) = P(x_1)P(x_2\vert x_1)P(x_3\vert x_1^2)\dots P(x_{n}\vert x_1^{n-1}) 
$$


$$
P(w_1^n)=\Pi_{k=1}^n P(x_k \vert x_1^{k-1})
$$

where $x_1^n$ means $x_1 x_2 \dots x_n$. The chain rule shows the link between computing the joint probability of a sequence and computing the conditional probability of a word given previous words.

The **intuition of the n-gram model** is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words. And applying **Markov Assumption** we can say that probability of $w_n$ depends on the previous word $w_{n-1}$ only i.e $P(w_n \vert w_{n-1})$ and thus we create the foundation of the **Bi-gram model**. 

Now product form of so many probabilities are going to be very small. So we can take $\log$ of $P(w_1^n)$ and get a nice summation form:

$$
\log(P(w_1^n) = \Sigma_{k=1}^{n} P(w_n \vert w_{n-1})
$$

_note: the above formulation is for Bi-gram model_

Now interestingly the above equation gives us the joint probability of those words together. So, it's showing the **likelihood** (i.e. the probability) of the data (i.e. the sentence). To estmiate these likelihood we can go for maximization -- enter **Maximum LIkelihood Estimation**. 

**Evaluation:** In practice we don’t use raw probability as our metric for evaluating language mod- els, but a variant called **perplexity**. However the concept of perplexity is related to entropy.  

> There is another way to think about perplexity: as the **weighted average branching factor** of a language. The branching factor of a language is the number of possible next words that can follow any word.

However with count based approach we may face the situation of **Zero Count** problem. And to mitigate that use **Smoothing**.

**Smoothing:**

- Laplace Smoothing (additive smoothing)
- Add-k Smoothing
- Kneser-Ney Smoothing (best one, most commonly used)
  - Kneser-Ney smoothing makes use of the probability of a word being a novel **continuation**. The interpolated Kneser-Ney smoothing algorithm mixes a discounted probability with a lower-order continuation probability.

**Reference:**

- Book: Speech and Language Processing, Jurafski, ch 3,  


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# What is Linear Discriminant Analysis

- Logistic regression involves directly modeling $P(Y = k \vert X = x)$ using the logistic function. But in LDA we take indirect approach. In this alternative approach, we model the distribution of the predictors $X$ separately in each of the response classes (i.e. given $$), and then use Bayes’ theorem to flip these around into estimates for $Pr(Y = k \vert X = x)$. 
- When these distributions are assumed to be normal, it turns out that the model is very similar in form to logistic regression.

## Why we need LDA when we have Logistic Regression?

- When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant analysis does not suffer from this problem.
- If $n$ is small and the distribution of the predictors $X$ is approximately normal in each of the classes, the linear discriminant model is again more stable than the logistic regression model.
- linear discriminant analysis is popular when we have more than two response classes.

## How LDA is related to PCA?

-  The goal of an LDA is to project a feature space (a dataset $n$-dimensional samples) onto a smaller subspace $k$ (where $k \leq n−1$) while maintaining the `class-discriminatory` information. 
- It's a classification algorithm.
- Linear Discriminant Analysis (LDA) is most commonly used as dimensionality reduction technique in the pre-processing step for pattern-classification and machine learning applications. The goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs.
- The general LDA approach is very similar to a Principal Component Analysis, but in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally interested in the axes that maximize the separation between multiple classes (LDA).

![image](https://sebastianraschka.com/images/blog/2014/linear-discriminant-analysis/lda_1.png)

- Both Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) are linear transformation techniques that are commonly used for dimensionality reduction.
- PCA is unsupervised where as LDA is supervised and LDA computes the `directions ("linear discriminants")` that will represent the axes that that maximize the separation between multiple classes


**Reference:**

- [Sebastian Raschka](https://sebastianraschka.com/Articles/2014_python_lda.html)
- Book: ISL, Chapter 4


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# What is Latent Dirichlet Allocation?

- Generative model for topic modelling. It's an unsupervised learning.
- LDA is one of the early versions of a ’topic model’ which was first presented by David Blei, Andrew Ng, and Michael I. Jordan in 2003.
- In essence, LDA is a `generative model` that allows observations about data to be explained by unobserved latent variables that explain why some parts of the data are similar, or potentially belong to groups of similar "topics"

![image](/assets/images/image_28_lda_3.png)
![image](/assets/images/image_28_lda_4.png)
![image](/assets/images/image_28_lda_5.png)

- Originally presented in a graphical model and using Variational Inference
- Builds on `latent semantic analysis`
- It is a mixed-­membership model.
- It relates to PCA and matrix factorization 

The `generative story` begins with begins with only Dirichlet Prior over the topics.

## Dirichlet Distribution

- In probability and statistics, the Dirichlet distribution (after Peter Gustav Lejeune Dirichlet), often denoted $Dir( \alpha )$, is a family of `continuous multivariate probability distributions` parameterized by a vector $\alpha$ of positive reals. 
- It is a `multivariate generalization of the beta distribution`, hence its alternative name of multivariate beta distribution (MBD).
- Dirichlet distributions are commonly used as prior distributions in Bayesian statistics, and in fact the Dirichlet distribution is the conjugate prior of the categorical distribution and multinomial distribution. 

![image](/assets/images/image_28_lda_1.png)
![image](/assets/images/image_28_lda_2.png)


**Resource:**

- [IMP: Topic Model: CMU](https://www.cs.cmu.edu/~mgormley/courses/10701-f16/slides/lecture20-topic-models.pdf)
- [Lecture 10: LDA](http://www.cs.columbia.edu/~verma/classes/uml/lec/uml_lec9_lda.pdf)
- [Dirichlet_distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Exercise:

1. What is LSI, NMF in nlp?
2. Using traditional ML, how to learn context?



----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
