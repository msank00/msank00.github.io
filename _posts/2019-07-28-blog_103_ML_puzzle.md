---
layout: post
title:  "Blog 103: ML Puzzle"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

# Interesting ML Puzzle (curated over internet)

1. **[ML Questions] Overfitting**: I am training a Random Forest on 1 million ($10^6$) points having 10000 ($10^4$) dimensions. I have already trained 5000 trees and want to train another 10000. Should I go ahead and train 15000 trees or do I have danger of overfitting? [[Qlink](https://www.linkedin.com/feed/update/urn:li:activity:6498386172857933824/)]

    - **[Ans]**: In his 1999 seminal paper (Theorem 1.2), Breiman already mathematically proved that Random Forest does not overfit on the number of trees. This is due to the `law of large numbers`. The error will always converge with more number of trees. [[paper link](https://www.stat.berkeley.edu/~breiman/random-forests.pdf)]
    - **[Q]**:  How many features (apprx) are used at a time to train single tree? 
      - $\log_e(K)$, where K is the dimension of the input data, i.e number of features.
    - **[Observation]**: With more trees the training and prediction time will be higher. Why?
      - Prediction time is usually a big factor. Just increasing the number of trees will  soon become prohibitive. In my case, assume every tree was balanced, then I will need ($\log_2(10^6)$) 20 iteration on average to reach to leaf. With 10000 trees, I am doing 20 X 10000 threshold evaluations at prediction time. Very expensive.

2. **[ML Questions] Bias and Variance**: Assume that I am training a random forest, where each tree is grown fully. The training data consists of N samples. To train a tree I create a subset of size N by sampling with replacement from training data.
The original training data is composed of F features, and for determining the split at any node in a tree, I am using $\log_e(F)$ features as candidates.
Assume that the trees are grown fully.
If we consider the individual trees, will they have `high variance, high bias`, or nothing can be said about individual trees?
When we combine the trees, are we trying to correct the variance or the bias or both? [[Qlink](https://www.linkedin.com/feed/update/urn:li:activity:6497416642216198144/)]
    - [[effect of different hyper-parameter](https://towardsdatascience.com/random-forests-and-the-bias-variance-tradeoff-3b77fee339b4)]
    -  A single decision tree if fully grown that means you are increasing the complexity of the tree. This implies the bias is reduced but variance increased.
    - **[My Ans]**: here even if each tree is grown fully, they are grown over a dataset of same size of original data, but sampled with replacement. So they haven't seen the full data, so they shouldn't overfit i.e variance will be low. But w.r.t each sampled dataset each tree is fully grown, so they have high variance and low bias (fully grown). By combining the trees we are correcting the variance. 

