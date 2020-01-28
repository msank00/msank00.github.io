---
layout: post
title:  "Statistical Analysis"
date:   2019-10-30 00:11:31 +0530
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

# When we should accept an Algorithm and when we shouldn't?

Each algorithm is based on some assumption which is applicable to some scenario. Now if the assumption fails in some scenario, then the ALgorithm will fail there. 

**Example:**

## K-Means:

**Assumptions:**

- k-means assume the variance of the distribution of each attribute (variable) is spherical;

![image](https://upload.wikimedia.org/wikipedia/commons/6/6e/Uniform_Spherical_Distribution_8.png)

- All variables have the same variance;
- The prior probability for all k clusters are the same, i.e. each cluster has roughly equal number of observations; 

**Drawbacks:**
- If any one of these 3 assumptions is violated, then k-means will fail.
- Number of cluster needs to know beforehand. 
- The clusters should be non-overlapping.

**Reference:**

- [kmeans-free-lunch](http://varianceexplained.org/r/kmeans-free-lunch/) 

## Hierarchical Clustering

> Hierarchical clustering is the hierarchical decomposition of the data based on group similarities

**Assumptions:**

1. There should be hierarchical relationship in the data, i.e a `smaller cluster is nested within a bigger cluster`.

> The term hierarchical refers to the fact that clusters obtained by `cutting the dendrogram at a given height` are **necessarily nested** within the clusters obtained by `cutting the dendrogram at any greater height`.


**Pros:**
- No need to know the number of cluster beforehand.

**Cons:**

- If the data doesn't satisfy Assumption 1, then it's not recommended to use Hierarchical Clustering. For **example:** In an arbitary dataset 120 people, 60 male and 60 female and they are from US, ITALY and INDIA with (40 people/Nationality). Here gender and nationality are not related so there is no nesting relation here and therefore, it's unwise to use Hierarchical Clustering here.

- Both K-Means and Hierarchical Clustering techniques are hard cluster. They force each datapoint to belong to any cluster even if it's an outlier. So the final clustering may be distorted. So these models are `sensitive to outlier`.
- In such scenario, soft cluster like Mixture Models are more appropriate.  

**Reference:**

- [Book: ISL Chapter 10, Page 394]()

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# Bayesian Analysis

$$P(Hyp \vert Data) = \frac{P(Data \vert Hyp) P(Hyp)}{P(Data)}$$

$$posterior \propto Likelihood \times prior$$

- $p(Hyp)$ is the `probability of the hypothesis` before we see the data, called the `prior probability`, or just **prior**.
- $p(Hyp\vert Data)$ is our goal, this is the `probability of the hypothesis` after we see the data, called the **posterior**.
- $p(Data \vert Hyp)$ is the `probability of the data under the hypothesis`, called the **likelihood**.

> There is an element which is key when we want to build a model under Bayesian approach: the Bayes factor. The Bayes factor is the ratio of the likelihood probability of two competing hypotheses (usually null and alternative hypothesis) and it helps us to quantify the support of a model over another one.

- however, the prior starts to lose weight when we add more data

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to interpret $R^2$ ?

> Specifically, this linear regression is used to determine how well a line fits’ to a data set of observations, especially when comparing models. Also, it is the fraction of the total variation in y that is captured by a model.

![image](https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Coefficient_of_Determination.svg/400px-Coefficient_of_Determination.svg.png)
![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/6b863cb70dd04b45984983cb6ed00801d5eddc94)

$$SS_{tot} = \Sigma_i (y_i - \bar y)^2$$
$$SS_{res} = \Sigma_i (y_i - \hat{y_i})^2$$ 

where $\hat{y_i} = f(x_i)$ 

- If the adjusted $R^2$ of our final model is only $0.3595$, so this means that $35.95$% of the variability is explained by the model.

**Reference:**

- [Wiki](https://en.wikipedia.org/wiki/Coefficient_of_determination)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Linear and Bayesian modeling in R: Predicting movie popularity


**Reference**

- [TDS Blog](https://towardsdatascience.com/linear-and-bayesian-modelling-in-r-predicting-movie-popularity-6c8ef0a44184)

----

# What is Sampling methods?

These are approximation methods to solve complex probabilistic model, which are difficult to solve in polinomial time with simpler methods. 

There exist two main families of approximate algorithms: 

- **Variational methods:** Formulate inference as an optimization problem
- **Sampling methods:**  Which produce answers by repeatedly generating random numbers from a distribution of interest.

## Sampling from a probability distribution

**Q:** How we might sample from a multinomial distribution with $k$ possible outcomes and associated probabilities $\theta_1, \dots , \theta_k$.

Reduce sampling from a multinomial variable to sampling a single uniform variable by subdividing a `unit interval` into $k$ regions with region $i$ having size $\theta_i$. We then sample uniformly from $[0,1]$ and return the value of the region in which our sample falls.

![image](https://ermongroup.github.io/cs228-notes/assets/img/multinomial-sampling.png)

_Reducing sampling from a multinomial distribution to sampling a uniform distribution in [0,1]._


- Sampling from a distribution lets us perform many useful tasks, including marginal and MAP inference, as well as computing integrals of the form

$$\E_{x \sim p}[f(x)] = \sum_x f(x) p(x)$$

>> Algorithms that construct solutions based on a large number of samples from a given distribution are referred to as Monte Carlo (MC) methods.

For different type of sampling technique check [here](https://ermongroup.github.io/cs228-notes/inference/sampling/).




**Reference**

- [Sampling Methods CS228](https://ermongroup.github.io/cs228-notes/inference/sampling/)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# What is Varietional Inference?

- Variational inference methods take their name from the `calculus of variations`, which deals with optimizing functions that take other functions as arguments.


> Inference in probabilistic models is often `intractable`, and we learned about algorithms that provide approximate solutions to the inference problem (e.g. marginal inference) by using subroutines that involve sampling random variables.

Unfortunately, these sampling-based methods have several important shortcomings.

So an alternative approach to approximate inference is by using the variational family of algorithms.

The **main idea** of variational methods is to cast `inference as an optimization problem`.

**Inference as Optimization:** Suppose we are given an `intractable probability distribution` $p$. Variational techniques will try to solve an optimization problem over a class of tractable distributions $Q$ in order to find a $q∈\in Q$ that is most similar to $p$. We will then query $q$ (rather than $p$) in order to get an approximate solution.


- Variational inference methods often scale better and are more amenable to techniques like stochastic gradient optimization, parallelization over multiple processors, and acceleration using GPUs.

To formulate inference as an optimization problem, we need to choose an approximating family $Q$ and an optimization objective $J(q)$. This objective needs to capture the similarity between $q$ and $p$; the field of information theory provides us with a tool for this called the _Kullback-Leibler (KL)_ divergence.

$$
KL(q\|p) = \sum_x q(x) \log \frac{q(x)}{p(x)} = E_{x \sim q}[f(x)]
$$

where $f(x)$ is $\log \frac{q(x)}{p(x)}$


**Reference**

- [High Level Understanding](https://www.cs.jhu.edu/~jason/tutorials/variational.html)
- [Important CS228](https://ermongroup.github.io/cs228-notes/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>