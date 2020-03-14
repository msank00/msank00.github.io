---
layout: post
title:  "5th Elephant Conference Summary"
date:   2019-08-20 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

5th Elephant Conference

- [Website](https://fifthelephant.in/2019/)

# Day 1

## Talk 1: Vehicle Routing Problem for Solving the Shipment Delivery

- **Speaker:** Venkateshan (Flipkart)

- Variation of TSP 
- NP Complete
- Formulate as a cost problem: assign a cost `W` to every routing configuration. Find routing with minimum cost.

**Algo:**

- Integer Linear Algo
- Dynamic Programming

Exact algorithm is costly, use approximation algorithm.

- Inserting an `un-routed` customer `c` between routing customer `u` and `v` in a route.
- Neighboring customer shrinkage in time window.
- Find minimum insertion cost. 

### Different Cost Associated with route

- Total travel time cost
- Fairness cost
- Outlier cost
- Compactness cost

**Reference:**

- [Paper:Vehicle Routing with Time Windows: Two Optimization Algorithms ](https://www.jstor.org/stable/172024?seq=1#page_scan_tab_contents)
- [Paper:Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints](https://dl.acm.org/citation.cfm?id=2778358)
- [Blog:Matchmaking in Lyft Line — Part 1](https://eng.lyft.com/matchmaking-in-lyft-line-9c2635fe62c4)
- [Blog: ETA Phone Home: How Uber Engineers an Efficient Route](https://eng.uber.com/engineering-an-efficient-route/)
- [Paper: A Decomposition Algorithm to Solve the Multi-Hop Peer-to-PeerRide-Matching Problem](https://arxiv.org/pdf/1704.06838.pdf)
- [Paper:A Matching Algorithm for Dynamic Ridesharing](https://www.sciencedirect.com/science/article/pii/S2352146516308730)
- [Paper: Algorithms for Trip-Vehicle Assignment in Ride-Sharing](https://www.ntu.edu.sg/home/xhbei/papers/ridesharing.pdf)
- [Youtube: Dawn Woodard: How Uber matches riders and drivers to reduce waiting time](https://www.youtube.com/watch?v=GyPq2joHZv4)
- [Youtube: Matching and Dynamic Pricing in Ride-Hailing Platforms](https://www.youtube.com/watch?v=cddFAgRyxQ0)
- [Youtube: Map Matching @ Uber: Markov Process](https://www.youtube.com/watch?v=ChtumoDfZXI)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Talk 2: Optimizing Debt Collection with Survival Model

- **Speaker:** Fasih Khativ (Simpl)

Broader Probelem Statement:  Skip paying again & again Pay one bill instead get all your online purchases added up.
and pay the total in one go.

Algorithm

- Mainly using Survival Model

**Reference::**

- [Blog: Survival Analysis: Intuition & Implementation in Python](https://towardsdatascience.com/survival-analysis-intuition-implementation-in-python-504fde4fcf8e)
- [Blog: Survival Analysis — Part A](https://towardsdatascience.com/survival-analysis-part-a-70213df21c2e)
- [Blog: Survival Analysis](http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Survival/BS704_Survival_print.html)
- [Notes](https://data.princeton.edu/wws509/notes/c7.pdf)
- [AV: Survival Analysis, Comprehensive Guide](https://www.analyticsvidhya.com/blog/2015/05/comprehensive-guide-parametric-survival-analysis/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Talk 3: Tutorial: Deeplearning in Production using RedisAI

- **Speaker:** Sherin Thomas (Tensorwerk)

- `Hanger` - Git for tensor [Github](https://github.com/tensorwerk/hangar-py)
- [Tutorial Material: Deeplearning in Production using RedisAI](https://github.com/konferenz/fifthel19)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Talk 4: The Anaconda Journey

- **Speaker:** Peter Wang (Anaconda, Inc)

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>