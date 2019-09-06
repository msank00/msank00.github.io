---
layout: post
title:  "Blog 101: ML QnA1"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---
# Q source: 

- [link_1](https://appliedmachinelearning.wordpress.com/2018/04/13/my-data-science-machine-learning-job-interview-experience-list-of-ds-ml-dl-questions/) 

## **What are the parameters in training a decision tree?**
   * [source](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

## **What is the philosophy behind Decision Tree?**

- A decision tree is a tree where each node represents a feature(attribute), each link(branch) represents a decision(rule) and each leaf represents an outcome(categorical or continues value)... [source](https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1)
- Find the feature that best splits the target class into the `purest possible` children nodes (ie: nodes that don't contain a mix of both classes, rather pure nodes with only one class).
- `Entropy` on the other hand is a measure of impurity. It is defined for a classification problem with N classes as:
- Entropy = $-\Sigma_i C_i * \log(C_i)$, where `i=1,...,N`
  - Say we have a dataset `D` and we are looking for a potential feature `f`, on which we will split the dataset w.r.t `f` into 2 parts `Dl` and `Dr` for left and right dataset respectively, such that those two datasets are at their purest form. Finally we use information gain to decide how good that feature is i.e how much pure the split is w.r.t `f` using `Information Gain`. 
    ```py
     Df
    / \
    Dl Dr
    ```
    + Information Gain: It is the difference of entropy before the split and after the split.
    `EntropyBefore_f = Entropy(Df)` and entropy after is 
    `EntropyAfter_f = Entropy(Dl)+Entropy(Dr)` and finaly 
    `InformationGain_f = EntropyBefore_f - EntropyAfter_f`.
      + [source](https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain)

## **How to build decision tree?**
   + There are couple of algorithms there to build a decision tree. Some of the important ones are
      + CART (Classification and Regression Trees) → uses Gini Index(Classification) as metric. Lower the Gini Index, higher the purity of the split.
      + ID3 (Iterative Dichotomiser 3) → uses Entropy function and Information gain as metrics. Higher the Information Gain, better the split is.
1. What are the criteria for splitting at a node in decision trees ?
   * Gini Index [[link](https://dni-institute.in/blogs/cart-decision-tree-gini-index-explained/)]
     + CART uses Gini index as a split metric. For `N` classes, the Gini Index is defined as: 
     $1-\Sigma_i P_i^2$, where `i=1,...,N` and $P_i=P(target=i)$ [[source](https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1)]
   * Information Gain
   * Cross Entropy
     + Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.
     + In binary classification, where the number of classes M equals 2, cross-entropy can be calculated as:
     $-{(y\log(p) + (1 - y)\log(1 - p))}$
     + If M>2 (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.
     $H(y, \hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} = -\sum_i y_i \log \hat{y}_i$
   * Entropy
   * **CHI Square:** It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node.
     + [link_1](https://www.mathsisfun.com/data/chi-square-test.html)
     + [link_2](https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb)
   * Reduction of Variance
   * [source](https://clearpredictions.com/Home/DecisionTree)

## **What is the formula of Gini index criteria?**
   * [link](http://dni-institute.in/blogs/cart-decision-tree-gini-index-explained/)

## **What is the formula for Entropy criteria?**
   + Entropy = $-\Sigma_i C_i * \log(C_i)$, where `i=1,...,N`

-----

## **What is KL Divergence?**
   + KL Divergence is the measure of **relative entropy**. It is a measure of the “distance” between two distributions. In statistics, it arises as an expected logarithm of the likelihood ratio. The relative entropy 
   ${KL}(p\sim||\sim q)$ 
   is a measure of the inefficiency of assuming that the distribution is q, when the true distribution is p. The KL divergence from p to q is simply the difference between cross entropy and entropy:
   $${KL}(y~||~\hat{y}) = \sum_i y_i \log \frac{1}{\hat{y}_i} - \sum_i y_i \log \frac{1}{y_i} = \sum_i y_i \log \frac{y_i}{\hat{y}_i}$$. 
   Where $y_i \sim p$ and $\hat{y}_i \sim q$, i.e. they come from two different probability distribution.

### **How is it decided that on which features it has to split?**
   + Based on for which feature the information gain is maximum.

### **How do you calculate information gain mathematically?** 
   + [clear explanation, slides](https://www3.nd.edu/~rjohns15/cse40647.sp14/www/content/lectures/23%20-%20Decision%20Trees%202.pdf)
   + If `H` is the entropy of the original data D and it has undergone `N` splits for feature `f`, then Information Gain: $IG(D,f) = H - \Sigma \frac{S_i}{S}H_i$ , where `i=1,...,N` and $S$ is the size of total datasets and $S_i$ is the size of the $i_{th}$ split data.  


## **What is the advantage with random forest ?**
   + [link](https://www.quora.com/What-are-some-advantages-of-using-a-random-forest-over-a-decision-tree-given-that-a-decision-tree-is-simpler)

## **Why ensemble is good?**

Suppose we have 10 independent classifiers, each with error rate of $0.3$ i.e $\epsilon=0.3$, 

In this setting, the error rate of the ensemble can be computed as below (we are taking a majority vote on the predictions. An ensemble makes a wrong prediction only when more than half of the base classifiers are wrong)

$$\epsilon_{ensemble}=\Sigma_{i=6}^{i=10} {10 \choose i}\epsilon^i(1-\epsilon)^{10-i} \approx 0.05 $$


+ [link](https://www.quora.com/What-are-some-advantages-of-using-a-random-forest-over-a-decision-tree-given-that-a-decision-tree-is-simpler)

## **Ensemble Learning algorithm**
   + [link1](https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/)
   + [link2](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/)


## **Boosting algorithms**

The term ‘Boosting’ refers to a family of algorithms which converts weak learner to strong learners.

1. AdaBoost (Adaptive Boosting)
2. Gradient Tree Boosting
3. XGBoost


   + [link1](https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/)


### **How does gradient boosting works ?**
   + [link](https://www.analyticsvidhya.com/blog/2015/09/complete-guide-boosting-methods/)
1.  **NOTE:** Bagging and Boosting both are ensemble learning algorithm, 
	+ where a collection of weak learner build the strong learner. While Bagging 
	works on resampling data with replacement and create different dataset and 
	the week learners are learnt on them, and final predictions are taken by 
	averaging or majority voting. E.g. Random Forest.
    + **Bagging** is a simple ensembling technique in which we build many independent predictors/models/learners on sampled data with replacement from the original data (**Bootstrap Aggregatoin**) and combine them using some model averaging techniques. (e.g. weighted average, majority vote or normal average). E.g: `Random Forest`
    + **Boosting** is also an ensemble learning method in which the predictors are not made independently, but sequentially. [source](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
      + **AdaBoost:** First a weak learner is applied and all the training 
	examples, which are misclassified, are given higher weight. 
	So while building the dataset for training the next learner, the 
	previously misclassified training examples will appear in the 
	dataset (as high weightage has been given to them). Now on this new 
	dataset another learner is trained. 
	Obviously this learner will correctly classify those previously 
	misclassified examples + some more misclassification in this step. 
	Now repeat.  
      + **Gradiant Boosting:** It 
	works differently. It learns a weak learner `F(x)+noise`. Then on the noise 
	it builds another weak learner `H(x)+error2` and so on... Thus it becomes 
	`F(x)+H(x)+G(x)+....+noise`.

### **Do you know about Adaboost algorithm ? How and why does it work ?**

- mathematical explanation [[link](http://math.mit.edu/~rothvoss/18.304.3PM/Presentations/1-Eric-Boosting304FinalRpdf.pdf)]

### **Difference of adaboost and gradiant boosting:**

- [link](https://www.quora.com/What-is-the-difference-between-gradient-boosting-and-adaboost) 
- Mathemetical Explanation [[Imp_link](http://www.ccs.neu.edu/home/vip/teach/MLcourse/4_boosting/slides/gradient_boosting.pdf)]


### **Bagging boosting difference:**

- [link](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)

----

## **SVM Summary:**

A Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The vectors (cases) that define the hyperplane are the support vectors.

### **Algorithm** 		

```py
1. Define an optimal hyperplane: maximize margin
2. Extend the above definition for non-linearly separable problems: have a penalty term for misclassifications.
3. Map data to high dimensional space where it is easier to classify with linear decision surfaces: reformulate problem so that data is mapped implicitly to this space.
```

<img src="https://saedsayad.com/images/SVM_optimize.png" alt="image" width="400"/>

<img src="https://saedsayad.com/images/SVM_optimize_1.png" alt="image" width="400" />

We find w and b by solving the following objective function using Quadratic Programming.

### **Hard Margin**

$$min \frac{1}{2}w^Tw$$ 

s.t $y_i(w.x_i+b)\ge 1, \forall x_i$ 

### **Soft Margin**

The beauty of SVM is that if the data is linearly separable, there is a unique global minimum value. An ideal SVM analysis should produce a hyperplane that completely separates the vectors (cases) into two non-overlapping classes. However, perfect separation may not be possible, or it may result in a model with so many cases that the model does not classify correctly. In this situation SVM finds the hyperplane that maximizes the margin and minimizes the misclassifications. 		

<img src="https://saedsayad.com/images/SVM_3.png" alt="image" width="400" />

<img src="https://saedsayad.com/images/SVM_optimize_3.png" alt="image" width="400"/>

The simplest way to separate two groups of data is with a straight line (1 dimension), flat plane (2 dimensions) or an N-dimensional hyperplane. However, there are situations where a nonlinear region can separate the groups more efficiently. SVM handles this by using a **kernel function** (nonlinear) to map the data into a different space where a hyperplane (linear) cannot be used to do the separation. 

- It means a non-linear function is learned by a linear learning machine in a high-dimensional **feature space** while the capacity of the system is controlled by a parameter that does not depend on the dimensionality of the space. This is called kernel trick which means the kernel function transform the data into a higher dimensional feature space to make it possible to perform the linear separation. 

- [Blog](https://saedsayad.com/support_vector_machine.htm)

### **How do you adjust the cost parameter for the SVM regularizer?**

Regularization problems are typically formulated as optimization problems involving the desired objective(classification lo  ss in our case)and a regularization penalty.The regularization penalty is used to help stabilize the minimization of the ob­jective or infuse prior knowledge we might have about desirable solutions.Many machine learning methods can be viewed as regularization methods in this manner.For later utility we will cast SVM optimization problem as a 
regularization problem.


Re write the soft margin problem using hinge loss $(z)$ defined as the positive part of $1-z$, written as $(1-z)^+$. The relaxed optimization problem (soft margin) can be reformulated as 

$$min \frac{1}{2}\vert \vert w \vert \vert^2 + C \Sigma_{t=1}^{n}(1 - y_t(w^T x_t + w_0))^+ $$

Here $\frac{1}{2}\vert \vert w \vert \vert^2$, the `inverse squared` **geometric margin** is viewed as a regularization penalty that helps stabilizes the objective $C \Sigma_{t=1}^{n}(1 - y_t(w^T x_t + w_0))^+$ . 

- [MOT OCW Notes](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/lecture-notes/lec4.pdf)

### **What sort of optimization problem would you be solving to train a support vector machine?**

  + `Maximize Margin` (best answer), quadratic program, quadratic with linear constraints, reference to solving the primal or dual form.

### **What are the kernels used in SVM ?**

Kernel $K(X_i, X_j)$ are:
- Linear Kernel: $X_i.X_j$
- Polynomial Kernel: $(\gamma X_i.X_j + C)^d$
- RBF Kernel: $\exp (-\gamma\vert X_i - X_j\vert ^2)$
- Sigmoid Kernel: $\tanh(\gamma X_i.X_j + C)$

where $K(X_i, X_j) = \phi(X_i).\phi(X_j)$

that is, the kernel function, represents a dot product of input data points mapped into the higher dimensional feature
space by transformation $\phi()$

$\gamma$ is an adjustable parameter of certain kernel functions.

The RBF is by far the most popular choice of kernel types used in Support Vector Machines. This is mainly because of their localized and finite responses across the entire range of the real x-axis.

- [Blog](http://www.statsoft.com/Textbook/Support-Vector-Machines)

### **What is the optimization technique of SVM?**

- [Imp_link](https://drona.csa.iisc.ac.in/~shivani/Teaching/E0370/Aug-2011/Lectures/2.pdf)

### **How does SVM learns the hyperplane ? Talk more about mathematical details?**

- [Imp_link](https://drona.csa.iisc.ac.in/~shivani/Teaching/E0370/Aug-2011/Lectures/2.pdf)


### **Why bring Lagrange Multiplier for solving the SVM problem?**

- Constrained Optimization Problem easier to solve with Lagrange Multiplier
- The existing constraints will be replaced by the constraints of the Lagrange Multiplier, which are easier to handle
- By this reformulation of the problem, the data will appear only as dot product, which will be very helpful while generalizing the SVM for non linearly separable class. [source page 129, last paragraph]

- [Youtube:Lagrange Multiplier Intuition](https://www.youtube.com/watch?v=yuqB-d5MjZA&list=PLg9_rXni6UXmjc7Cxw8HpYWRlFg72v15a&index=7)

### **KKT Condition for SVM?**

- [SB course](http://cse.iitkgp.ac.in/~sourangshu/coursefiles/ML15A/svm.pdf), [link2](http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/KKT.pdf)


### **Geometric analysis of Lagrangian, KKT, Dual**

- [link](http://anie.me/Lagrangian-And-Dual-Problem/)


### **How does SVM learns non-linear boundaries ? Explain.**
+ using `kernel trick`, it maps the examples from `input space` to `feature space`. 
In the higher dimension, they are separated linearly.

----

## Talking about unsupervised learning ? What are the algorithms ?

- Clustering
- [K Means](http://www.saedsayad.com/clustering_kmeans.htm)
  - K-Means clustering intends to partition n objects into k clusters in which each object belongs to the cluster with the nearest mean. This method produces exactly k different clusters of greatest possible distinction. The best number of clusters k leading to the greatest separation (distance) is not known as a priori and must be computed from the data. The objective of K-Means clustering is to minimize total intra-cluster variance, or, the squared error function:
  - $J = \Sigma_j \Sigma_i \vert\vert x_i - c_j \vert\vert^2$ where `j=1,...,K` and `i=1,...,N`. `N` total number of observations and `K` total number of classes

**Algorithm:**
```py    		
1. Clusters the data into k groups where k  is predefined.
2. Select k points at random as cluster centers.
3. Assign objects to their closest cluster center according to the Euclidean distance function.
4. Calculate the centroid or mean of all objects in each cluster.
5. Repeat steps b, c and d until the same points are assigned to each cluster in consecutive rounds. 
```
- **Seed K-Means:** For seeding, i.e, to decide the insitial set of `K` centroids, use **K-Means++** algorithm. 
- K Medoids
- Agglomerative Clustering
  - Hierarchichal Clustering
  - Dimensionality Reduction
- PCA
- ICA

## How do you decide K in K-Means clustering algorithm ?Tell me at least 

3 ways of deciding K in clustering ?

- Elbow Method
- Average Silhouette Method
- Gap Statistics 
- [link1](http://www.sthda.com/english/articles/29-cluster-validation-essentials/96-determining-the-optimal-number-of-clusters-3-must-know-methods/)
- [link2](https://uc-r.github.io/kmeans_clustering)


### How do you `seed` k-means algorithm,i.e. how to decide the first `k` clusters?

- k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm. It was proposed in 2007 by David Arthur and Sergei Vassilvitskii, as an approximation algorithm for the NP-hard k-means problem—a way of avoiding the sometimes poor clusterings found by the standard k-means algorithm.
- The intuition behind this approach is that spreading out the k initial cluster centers is a good thing: the first cluster center is chosen uniformly at random from the data points that are being clustered, after which each subsequent cluster center is chosen from the remaining data points with probability proportional to its squared distance from the point's closest existing cluster center.
- The exact algorithm is as follows:
```py
  **Algorithm**
      1. Choose one center uniformly at random from among the data points.
      2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
      3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
      4. Repeat Steps 2 and 3 until k centers have been chosen.
      5. Now that the initial centers have been chosen, proceed using standard k-means clustering.
```
- [resource_wiki](https://en.wikipedia.org/wiki/K-means%2B%2B), [link2](https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/)

### When K-means will fail?
- When data is **non linearly separable**. It works best when data clusters are discrete.  

## What other clustering algorithms do you know?

- ans: [link](https://sites.google.com/site/dataclusteringalgorithms/)
- Unsupervised **linear clustering algorithm**
- **k-means clustering algorithm** [link](https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html)
- Fuzzy c-means clustering algorithm
- **[Hierarchical clustering algorithm](https://sites.google.com/site/dataclusteringalgorithms/hierarchical-clustering-algorithm)**  
- Hierarchical Agglomerative Clustering (bottom up)
- Hierarchical DIvisive Clustering (top down)
- Gaussian(EM) clustering algorithm
- Quality threshold clustering algorithm      
- Unsupervised **non-linear clustering algorithm**
- **MST based clustering algorithm**
  - **Basic Idea:** Apply MST on the data points. Use the _Euclidean_ distance as the weight between two data points. After building the MST removes the longest edge, then the 2nd longest and so on. And thus clusters will be formed. [source](http://shodhganga.inflibnet.ac.in/bitstream/10603/9728/10/10_chapter%203.pdf)
- kernel k-means clustering algorithm
- Density based clustering algorithm
- **[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)**
- [code](https://plot.ly/scikit-learn/plot-dbscan/)

## What is DB-SCAN algorithm ?

- It is a density-based clustering algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions
- A point p is a core point if at least `minPts` points are within distance `ε` (`ε` is the maximum radius of the neighborhood from p) of it (including p). Those points are said to be directly reachable from p.
- A point q is `directly reachable` from p if point q is within distance `ε` from point p and p must be a core point.
- A point q is `reachable` from p if there is a path `p1, ..., pn` with `p1 = p` and `pn = q`, where each `pi+1` is directly reachable from `pi` (all the points on the path must be core points, with the possible exception of q).
- All points not reachable from any other point are outliers. 

## How does HAC (Hierarchical Agglomerative clustering) work ?
 
- [link1](https://newonlinecourses.science.psu.edu/stat505/node/143/)
- [link2](https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html)
- [code](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)


## **Explain PCA? Tell me the mathematical steps to implement PCA?**
+ In PCA, we are interested to find the directions (components) that maximize the variance in our dataset)
  + maximizes the variance
  + minimizes the reconstruction error
+ Let’s assume that our goal is to reduce the dimensions of a d-dimensional 
dataset by projecting it onto a `k`-dimensional subspace (where `k<d`). So, 
how do we know what size we should choose for `k`, and how do we know if we 
have a feature space that represents our data **well**?
  + We will compute `eigenvectors` (the components) from our data set and 
  collect them in a so-called scatter-matrix (or alternatively calculate 
  them from the **covariance matrix**). 
  + Each of those eigenvectors is associated with an eigenvalue, which tell us about the `length` or `magnitude` of the eigenvectors. If we observe that all the eigenvalues are of very similar 
  magnitude, this is a good indicator that our data is already in a "good" 
  subspace. Or if some of the eigenvalues are much much higher than others, 
  we might be interested in keeping only those eigenvectors with the much 
  larger eigenvalues, since they contain more information about our data 
  distribution. Vice versa, eigenvalues that are close to 0 are less 
  informative and we might consider in dropping those when we construct 
  the new feature subspace.
+ `kth` Eigenvector determines the `kth` direction that maximizes the variance in that direction.
+ the corresponding `kth` Eigenvalue determines the variance along `kth` Eigenvector

### Tl;DR: Steps 
  + Let $X$ is the original dataset (`n x d`) [$n$: data points, $d$: dimension]
  + Centered the data w.r.t mean and get the centered data: $M$ 
  + Compute covariance matrix: $C=(MM^T)/(n-1)$
  + Diagonalize it: $C = VDV^T$. $V$ is an `orthogonal` matrix so $V^T = V^{-1}$ [[proof](https://math.stackexchange.com/questions/1936020/why-is-the-inverse-of-an-orthogonal-matrix-equal-to-its-transpose)]
  + Compute `principal component`: $P = V\sqrt{D}$.
  + Combining all:
  $$C=(MM^T)/(n-1)=VDV^T= = V\sqrt{D}\sqrt{D}V^T =  PP^T$$
  + Apply principle component matrix $P$ on the centered data $M$ to get the tranformed data projected on the principle component and thus doing `dimensionality reduction`: $M* = M^T P$, [M*: new dataset after the PCA]

**Resource:**
+ [pdf](http://www.math.ucsd.edu/~gptesler/283/slides/pca_15-handout.pdf)
+ [Link1](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)
+ [Link2](https://rstudio-pubs-static.s3.amazonaws.com/249839_48d65d85396a465986f2d7df6db73c3d.html)
+ [PPT: Prof. Piyush Rai IIT Kanpur](https://cse.iitk.ac.in/users/piyush/courses/ml_autumn16/771A_lec11_slides.pdf)

### **What is disadvantage of using PCA?**

+ one disadvantage of PCA lies in interpreting the results of dimension reduction analysis. This challenge will become particularly telling when the data needs to be normalized.
+ PCA assumaes approximate normality of the input space distribution. [link](http://www.stat.columbia.edu/~fwood/Teaching/w4315/Fall2009/pca.pdf)
+ for more reading [link](https://www.quora.com/What-are-some-of-the-limitations-of-principal-component-analysis)

### **Why PCA needs normalization?**
+ A reason why we need to normalize before applying PCA is to mitigate the effects of scale. For example, if one of the attributes is orders of magnitude higher than others, PCA tends to ascribe the highest amount of variance to this attribute and thus skews the results of the analysis. By normalizing, we can get rid of this effect. However normalizing results in spreading the influence across many more principal components. In others words, more PCs are required to explain the same amount of variance in data. The interpretation of analysis gets muddied. 
[source](http://www.simafore.com/blog/bid/105347/Feature-selection-with-mutual-information-Part-2-PCA-disadvantages)


## **How do you deploy Machine Learning models ?**

- Microservice
- Docker
- Kubernetes

### **Lot of times, we may have to write ML models from scratch in C++ ? Will you be able to do that?**
+ [quora](https://www.quora.com/I-want-to-use-C++-to-learn-Machine-Learning-instead-of-Python-or-R-is-it-fine)

## **How Stochastic Gradient Decent with momentum works?**
+ An SGD can be thought of as a ball rolling down the hill where the velocity of the ball is influenced by the gradient of the 
curve. However, in this approach, the ball has a chance to get stuck in any ravine. So if the ball can have enough momentum to get past
get past the ravine would have been better. based on this idea, SGD with Momentum works. Where the ball has been given 
some added momentum which is based on the previous velocity and gradient. 
   + `velocity = momentum*past_velocity + learning_rate*gradient`
   +  `w=w-velocity`
   +  `past_velocity = velocity`
+ [easy explanation](http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/)
+ book: deep learning in Python - by Cholet, page 51, but the equation looks suspicious 
+ [through explanation, distill pub](https://distill.pub/2017/momentum/)

## **How the model varies in KNN for K=1 and K=N?**

- When K equals 1 or other small number the model is prone to overfitting (high variance), while when K equals number of data points or other large number the model is prone to underfitting (high bias)

## **Generative model vs Discriminative model.**

- Discriminative algorithms model `P(y|x; w)`, that is, given the dataset and learned parameter, what is the probability of y belonging to a specific class. A discriminative algorithm doesn't care about how the data was generated, it simply categorizes a given example.
- Generative algorithms try to model `P(x|y)`, that is, the distribution of features given that it belongs to a certain class. A generative algorithm models how the data was generated. [source](https://github.com/ShuaiW/data-science-question-answer#knn)

------

## **Scenario based Question**

Let’s say, you are given a scenario where you have terabytes of data files consisting of pdfs, text files, images, scanned pdfs etc. What approach will you take in understanding or classifying them ?

###  **How will you read the content of scanned pdfs or written documents in image formats?**

### **Why is naive bayes called “naive”? Tell me about naive bayes classifier?**

- Because it's assumed that all the features are independent of each other. This is a very _naive_ assumption.

----

## **Logistic Regression loss function?**
+ [ml-cheatsheet quick summary](http://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)
+ [CMU ML slides](https://www.cs.cmu.edu/~mgormley/courses/10701-f16/slides/lecture5.pdf)

## **What do you mean by mutable and immutable objects in python ?**
+ [Everything in Python is an object](https://medium.com/@meghamohan/mutable-and-immutable-side-of-python-c2145cf72747). 
Since everything in Python is an Object, every variable holds 
an object instance. When an object is initiated, it is assigned a unique object id. Its type is 
defined at runtime and once set can never change, however its state can be changed if it is 
mutable. Simple put, a mutable object can be changed after it is created, and an immutable object can’t.
+ Objects of built-in types like (`int, float, complex, bool, str, tuple, unicode, frozen set`) are immutable. Objects of 
built-in types like (`list, set, dict,byte array`) are mutable. Custom classes are generally mutable.

## What are the data structures you have used in python ?

+ set,list,tuple,dictionary, string, frozen set. [link1](https://docs.python.org/3/tutorial/datastructures.html),
[link2](http://thomas-cokelaer.info/tutorials/python/data_structures.html) 
------

## How do you handle multi-class classification with unbalanced dataset ?
+ [link1](https://www.linkedin.com/pulse/multi-class-classification-imbalanced-data-using-random-burak-ozen/)
+ handling imbalanced data by resampling original data to provide balanced classes. 
    + Random under sampling
    + Random over sampling
    + Cluster based over sampling
    + Informed Over Sampling: Synthetic Minority Over-sampling Technique (**SMOTE**)
+ Modifying existing classification algorithms to make them appropriate for imbalanced data sets.
   + Bagging based
   + Boosting based: AdaBoost, Gradient Boost, XGBoost
+ [imp source](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)

----


## **How do you select between 2 models (Model Selection techniques)?**

To choose between 2 model generally `AIC` or `BIC` are used.
Generally, the most commonly used metrics, for measuring regression model quality and for comparing models, are: `Adjusted R2`, `AIC`, `BIC` and `Cp`.

- **AIC** stands for (Akaike’s Information Criteria), a metric developped by the Japanese Statistician, Hirotugu Akaike, 1970. The basic idea of AIC is to `penalize the inclusion of additional variables` to a model. It adds a penalty that increases the error when including additional terms. `The lower the AIC, the better the model.`
`AICc` is a version of AIC corrected for small sample sizes.
- **BIC** (or Bayesian information criteria) is a variant of AIC with a stronger penalty for including additional variables to the model.
- **Mallows Cp:** A variant of AIC developed by Colin Mallows
- $R^2$ not a good criterion.Always increase with model size –> `optimum` is to take the biggest model.
- `Adjusted` $R^2$: better. It `penalized` bigger models.



**Source**
- [Blog](http://www.sthda.com/english/articles/38-regression-model-validation/158-regression-model-accuracy-metrics-r-square-aic-bic-cp-and-more/)
- [Blog](https://www.sciencedirect.com/topics/medicine-and-dentistry/akaike-information-criterion)
- [ppt-Stanford](https://statweb.stanford.edu/~jtaylo/courses/stats203/notes/selection.pdf)


### **Okay great BIC & AIC !! How does it work mathematically?Explain the intuition behind BIC or AIC ?**

In general, it might be best to use AIC and BIC together in model selection. 
- For example, in selecting the number of latent classes in a model, if BIC points to a three-class model and AIC points to a five-class model, it makes sense to select from models with 3, 4 and 5 latent classes. 
- `AIC is better in situations when a false negative finding would be considered more misleading than a false positive`, 
- `BIC is better in situations where a false positive is as misleading as, or more misleading than, a false negative`.


$$AIC=-2\log L(\hat \theta) + 2k$$

where, 

- $\theta$= the set (vector) of model parameters
- $L(\theta)$ =  the  likelihood  of  the  candidate  model  given  the  data  when  evaluated at the maximum likelihood estimate of $\theta$
- $k$ = the number of estimated parameters in the candidate model

The first compo-nent, $−2\log L(\hat \theta)$, is the value of the likelihood function, $\log L(\theta)$, which is the probability of obtaining the data given the candidate model.
The  more  parameters,  the  greater the amount added to the first component, increasing the value for the AIC and penalizing the model. 

**BIC** is  another  model  selection  criterion  based  on  infor-mation theory but set within a Bayesian context. The difference between the BIC and the AIC is the greater penalty imposed for the number of param-eters  by  the BIC  than the AIC.

$$BIC=-2\log L(\hat \theta) + k \log n$$

**Source:**
- [Blog](https://www.methodology.psu.edu/resources/aic-vs-bic/)
- [Paper](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118856406.app5)


-----
## **What is precision and recall ? Which one of this do you think is important in medical diagnosis?**

## **Type I and Type II Errors**

>> One fine morning, Jack got a phone call. It was a stranger on the line. Jack, still sipping his freshly brewed morning coffee, was barely in a position to understand what was coming for him. The stranger said, “Congratulations Jack! You have won a lottery of $10 Million! I just need you to provide me your bank account details, and the money will be deposited in your bank account right way…”

What are the odds of that happening? What should Jack do? What would you have done?

<img src="https://miro.medium.com/max/454/1*t_t7cMq3FGqDk6gbwfA4EA.png" alt="image" width="400"/>

- **Type I: False Positive**
- **Type II: False Negative**

<img src="https://miro.medium.com/max/700/1*pOtBHai4jFd-ujaNXPilRg.png" alt="image" width="600"/>

- [blog](https://towardsdatascience.com/precision-vs-recall-386cf9f89488)

----

## **What does AUC-ROC curve signify ?**

AUC - ROC curve is a performance measurement for classification problem `at various thresholds settings`. ROC is a **probability curve** and AUC represents **degree or measure of separability**. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.

### **How do you draw AUC-ROC curve ?**

<img src="https://miro.medium.com/max/700/1*k65OKy7TOhBWRIfx0u6JqA.png" alt="image" width="600"/>

<img src="https://miro.medium.com/max/700/1*hf2fRUKfD-hCSw1ifUOCpg.png" alt="image" width="600"/>

- `True positive` is the area designated as “bad” on the right side of the threshold (mild sky blue region). `False positive` denotes the area designated as “good” on the right of the threshold. 
- `Total positive` is the total area under the “bad” curve while total negative is the total area under the “good” curve.
- We divide the value as shown in the diagram to derive TPR and FPR. 
- We derive the TPR and FPR at different threshold values (by sliding the black vertical bar in the above image) to get the ROC curve. Using this knowledge, we create the ROC plot function.

**Bonus Question:** Write pseudo-code to generate the data for such a curve. [Check the below blog]

- [Imp Blog](https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0)

The ROC curve is plotted with `TPR` against the `FPR` where TPR is on y-axis and FPR is on the x-axis.

<img src="https://miro.medium.com/max/361/1*pk05QGzoWhCgRiiFbz-oKQ.png" alt="image" width="250"/>

- `TPR == RECALL`

$$\frac{TP}{TP+FN}$$

- Specificity:

$$\frac{TN}{TN+FP}$$

- `FPR == 1-Specificity`

$$\frac{FP}{TN+FP}$$

### **How will you draw ROC for multi class classification problem**

In multi-class model, we can plot N number of AUC ROC Curves for N number classes using One vs ALL methodology. So for Example, If you have three classes named X, Y and Z, you will have one ROC for X classified against Y and Z, another ROC for Y classified against X and Z, and a third one of Z classified against Y and X.


- [Blog](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

----

## **What is random about Random Forest ?**
   + for different `tree`, a different datasets (build from original using _random resampling with replacement_) are given as input.

----

1. Tell me any other metric to measure multi-class classification result ?
What is sensitivity and specificity ?
2. **Name the package of scikit-learn that implements logistic regression** ?
3. What is mean and variance of standard normal distribution ?
4. **What is central limit theoram**?
5. **Law of Large Number**?
4. What are the data structures you have used in python ?
 
11. What is naive bayes classifier ?
12. What is the probability of heads coming 4 times when a coin is tossed 10 number of times ?
13. **How do you get an index of an element of a list in python** ?
    + [link1](https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python)
14. How do you merge two data-set with pandas?
    + ```frames = [df1, df2, df3];result = pd.concat(frames)```
    + [link1](https://pandas.pydata.org/pandas-docs/stable/merging.html)
15. From user behavior, you need to model fraudulent activity. How are you going to solve this ? May be anomaly detection problem or a classification problem !!
16. What will you prefer a decision tree or a random forest ?
17. How is using a logistic regression different from using a random forest ?
    + If your data is linearly separable, go with logistic regression. However, in real world, data is rarely linearly separable. Most of the time data would be a jumbled mess.
    In such scenarioes, Decision trees would be a better fit as DT essentially is a non-linear classifier. As DT is prone to over fitting, Random Forests are used in practice to better generalize the fitment. RF provide a good balance between precision and overfitting.
      + If your problem/data is linearly separable, then first try logistic regression. If you don’t know, then still start with logistic regression because that will be your baseline, followed by non-linear classifier such as random forest. Do not forget to tune the parameters of logistic regression / random forest for maximizing their performance on your data.
      + If your data is categorical, then random forest should be your first choice; however, logistic regression can be dealt with categorical data[1].
      + If you want easy to understand results, logistic regression is a better choice because it leads to simple interpretation of the explanatory variables.
      + If speed is your criteria, then logistic regression should be your choice[2].
      + If your data is unbalanced, then random forest may be a better choice[3].
      + If number of data objects are less than the number of features, logistic regression should not be used[4].
      + Lastly, as noted in this paper, either of the random forest or logistic regression “models appear to perform similarly across the datasets with performance more influenced by choice of dataset rather than model selection”
    + [link1](https://www.quora.com/When-should-random-forest-be-used-over-logistic-regression-for-classification-and-vice-versa)
18. Will you use decision tree or random forest for a classification problem ? 
    What is advantage of using  random forest ?
------
1. Which model would you use in case of unbalanced dataset: Random Forest or Boosting ? Why ?
   + Gradient boosting is also a good choice here. You can use the gradient boosting classifier in sci-kit learn for example. Gradient boosting is a principled method of dealing with class imbalance by constructing successive training sets based on incorrectly classified examples.
   + [link1](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/)
   + An alternative could be a cost-sensitive algorithm like C5.0 that doesn't need balanced data. You could also think about applying Markov chains to your problem.
2. What are the boosting techniques you know ?
3. **Which model would you like to choose if you have many classes in a supervised learning problem ? Say 40-50 classes !!**
4. How do you perform ensemble technique?
5. How does SVM work ?
6. What is Kernel ? Explain a few.
7. **How do you perform non-linear regression**?
   + [link1](http://www.statisticshowto.com/nonlinear-regression/)
   + `\frac{a}{b}`
8. What are Lasso and Ridge regression ?
------

1. What is Gaussian Mixture model ? How does it perform clustering ?
2. How is Expectation Maximization performed ? Explain both the steps ?
3. How is likelihood calculated in GMM ?
