---
layout: post
title:  "Scalable Machine Learning"
date:   2020-12-01 00:00:10 -0030
categories: jekyll update
mathjax: true
---



# Content

1. TOC
{:toc}
---


# Using the Right Processors CPUs, GPUs, ASICs, and TPUs

CPUs are not ideal for large scale machine learning (ML), and they can quickly turn into a bottleneck because of the sequential processing nature. An upgrade on CPUs for ML is GPUs (graphics processing units). Unlike CPUs, GPUs contain hundreds of embedded ALUs, which make them a very good choice for any process that can benefit by leveraging parallelized computations. GPUs are much faster than CPUs for computations like vector multiplications. However, both CPUs and GPUs are designed for general purpose usage and suffer from **von Neumann bottleneck** () and higher power consumption.

> :bulb: **von Neumann bottleneck:** The shared bus between the program memory and data memory leads to the von Neumann bottleneck, the limited throughput (data transfer rate) between the central processing unit (CPU) and memory compared to the amount of memory. 

A step beyond CPUs and GPUs is **ASICs (Application Specific Integrated Chips)**, where we trade general flexibility for an increase in performance. There have been a lot of exciting research on for designing ASICs for deep learning, and Google has already come up with three generations of **ASICs called Tensor Processing Units (TPUs)**.

TPUs exploit the fact that neural network computations are operations of matrix multiplication and addition, and have the specialized architecture to perform just that. TPUs consist of **MAC units (multipliers and accumulators)** arranged in a `systolic array` fashion, which enables **matrix multiplications without memory access**, thus consuming less power and reducing costs.

> :bulb: **systolic array:** In parallel computer architectures, a systolic array is a homogeneous network of tightly coupled data processing units (DPUs) called cells or nodes. Each node or DPU independently computes a partial result as a function of the data received from its upstream neighbors, stores the result within itself and passes it downstream.

This way of performing matrix multiplications also reduces the computational complexity from the order of $n^3$ to order of $3n - 2$. for more details check [here](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu).

**Reference:**

- [Machine Learning: How to Build Scalable Machine Learning Models](https://www.codementor.io/blog/scalable-ml-models-6rvtbf8dsd) :fire:
- [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu) :rocket: 



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


-----

# Distributed model training

<center>
<img src="https://cdn.filestackcontent.com/jERYMQCTuWmEdsb3Lrfs" width="600">
</center>

A typical, supervised learning experiment consists of feeding the data via the input pipeline, doing a forward pass, computing loss, and then correcting the parameters with an objective to minimize the loss. Performances of various hyperparameters and architectures are evaluated before selecting the best one.

Let's explore how we can apply the `divide and conquer` approach to decompose the computations performed in these steps into granular ones that can be run independently of each other, and aggregated later on to get the desired result. After decomposition, we can leverage horizontal scaling of our systems to improve time, cost, and performance.

There are two dimensions to decomposition: functional decomposition and data decomposition.

:atom_symbol: **Functional decomposition:** Functional decomposition generally implies breaking the logic down to distinct and independent functional units, which can later be recomposed to get the results. "Model parallelism" is one kind of functional decomposition in the context of machine learning. The idea is to split different parts of the model computations to different devices so that they can execute in parallel and speed up the training.

:atom_symbol: **Data decomposition:** Data decomposition is a more obvious form of decomposition. Data is divided into chunks, and multiple machines perform the same computations on different data.

<center>
<img src="https://cdn.filestackcontent.com/a5diV7cSQ9iM1m8OvMRu" width="600">
</center>

**Example:** One instance where you can see both the functional and data decomposition in action is the training of an ensemble learning model like random forest, which is conceptually a collection of decision trees. Decomposing the model into individual decision trees in functional decomposition, and then further training the individual tree in parallel is known as data parallelism. It is also an example of what's called embarrassingly parallel tasks.

## Distributed Machine Learning

**MapReduce paradigm**

<center>
<img src="https://cdn.filestackcontent.com/yKx5ZzTsqz7Q5JainYIw" width="600">
</center>


:dart: **Distributed machine learning architecture**

The data is partitioned, and the driver node assigns tasks to the nodes in the cluster. The nodes might have to communicate among each other to propagate information, like the gradients. There are various arrangements possible for the nodes, and a couple of extreme ones include Async parameter server and Sync AllReduce.

<center>
<img src="https://cdn.filestackcontent.com/vr1fW04CRuCvFGZgzP8A" width="600">
</center>

**Async parameter server architecture**

<center>
<img src="https://process.filestackapi.com/cache=expiry:max/XaF4LtslScS1hGgOwyS2" width="600">
</center>

**Sync AllReduce architecture**

<center>
<img src="https://cdn.filestackcontent.com/WmjdHLxYQjqBlLiWQwSV" width="600">
</center>


**Popular frameworks for distributed machine learning**

- Apache Hadoop
- Apache Spark
- Apache Mahout

**Reference:**

- [Machine Learning: How to Build Scalable Machine Learning Models](https://www.codementor.io/blog/scalable-ml-models-6rvtbf8dsd)



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


-----
# Scaling common machine learning algorithms

## Logistic regression with 1 billion examples

**Reference:**

- [ML impossible: Train 1 billion samples in 5 minutes on your laptop using Vaex and Scikit-Learn](https://towardsdatascience.com/ml-impossible-train-a-1-billion-sample-model-in-20-minutes-with-vaex-and-scikit-learn-on-your-9e2968e6f385)



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Seminal Papers on Distributed Learning

- [Large Scale Distributed Deep Networks by Jeff Dean Google - NeurIPS 2012](https://papers.nips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html) :fire:
- [HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](https://arxiv.org/abs/1106.5730)
- [Dogwild! — Distributed Hogwild for CPU & GPU](http://stanford.edu/~rezab/nips2014workshop/submits/dogwild.pdf)



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Scalable Machine Learning

- [Scalable Machine Learning and Deep Learning](https://id2223kth.github.io/schedule/) :fire:
- [berkeley COMPSCI 294 - LEC 123 Scalable Machine Learning](https://bcourses.berkeley.edu/courses/1413454/)
- [berkeley SML: Scalable Machine Learning Alex Smola](http://alex.smola.org/teaching/berkeley2012/)
- [O'Reilly Podcast - How to train and deploy deep learning at scale](https://www.oreilly.com/radar/podcast/how-to-train-and-deploy-deep-learning-at-scale/)
- [O'Reilly Podcast - Scaling machine learning](https://www.oreilly.com/radar/podcast/scaling-machine-learning/)
- [Lessons Learned from Deploying Deep Learning at Scale ](https://algorithmia.com/blog/deploying-deep-learning-at-scale) :fire:
- [Deploying AI to production: 12 tips from the trenches](https://www.linkedin.com/pulse/deploying-ai-production-12-tips-from-trenches-max-pagels/) :rocket:



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Tools to build large scale ML System


- [GraphLab](https://www.analyticsvidhya.com/blog/2015/12/started-graphlab-python/)
- [Vowpal Wabbit](https://www.youtube.com/watch?v=gyCjancgR9U)
- [Sibyl: A System for Large Scale Machine Learning at Google](https://www.youtube.com/watch?v=3SaZ5UAQrQM) :fire:


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# How to structure PySpark application 

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/Bp0XvA3wIXw" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/Bp0XvA3wIXw)_ :fire:


## Debugging PySPark Application

<center>
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/A0jYQlxc2FU" frameborder="0" allowfullscreen="true" width="100%" height="300"> </iframe>
</figure>
</center>

_*In case the above link is broken, click [here](https://www.youtube.com/embed/A0jYQlxc2FU)_ :fire:



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Distributed Deep Learning


- [Distributed Deep Learning Pipelines with PySpark and Keras](https://towardsdatascience.com/distributed-deep-learning-pipelines-with-pyspark-and-keras-a3a1c22b9239)

## PyTorch Dataparallel

PyTorch has relatively simple interface for distributed training. To do distributed training, the model would just have to be wrapped using `DistributedDataParallel` and the training script would just have to be launched using `torch.distributed.launch`.

- Please look at [this](https://leimao.github.io/blog/PyTorch-Distributed-Training/) tutorial for docker based distributed deep learning training :fire:

**Resource:**

- [Official PyTorch example](https://pytorch.org/tutorials/#parallel-and-distributed-training) :rocket:
- [Writing Distributed Applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html#)
- [PyTorch Distributed communication package](https://pytorch.org/docs/stable/distributed.html)
- [Single node PyTorch to distributed deep learning using HorovodRunner](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/mnist-pytorch.html)
- [Databricks HorovodRunner: distributed deep learning with Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) :fire:
- [HorovodRunner for Distributed Deep Learning](https://databricks.com/blog/2018/11/19/introducing-horovodrunner-for-distributed-deep-learning-training.html)




<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>


----

# Guide to Scale Machine Learning Models in Production

Hands-on example [here](https://hackernoon.com/a-guide-to-scaling-machine-learning-models-in-production-aa8831163846). :fire: :fire:

----

# Scalable API - Deep Learning in Production

![image](https://pyimagesearch.com/wp-content/uploads/2018/01/deep_learning_cloud_animation.gif)

_*image [source](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/)_

> Shipping deep learning models to production is a non-trivial task 

If you don’t believe me, take a second and look at the “tech giants” such as Amazon, Google, Microsoft, etc. — nearly all of them provide some method to ship your machine learning/deep learning models to production in the cloud.

This type of situation is more common than you may think. Consider:

- An in-house project where you cannot move sensitive data outside your network
- A project that specifies that the entire infrastructure must reside within the company
- A government organization that needs a private cloud
- A startup that is in “stealth mode” and needs to stress test their service/application in-house

_*Please follow this amazing [blog](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/) from Adrian Rosebrock from Pyimagesearch_ :fire:

## First Approach: 

- [A scalable Keras + deep learning REST API](https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/)

<center>

<img src="https://www.pyimagesearch.com/wp-content/uploads/2018/01/keras_api_header.png" width="500">

</center>

:atom_symbol: **A short introduction to Redis as a REST API message broker/message queue**

<center>

<img src="https://www.pyimagesearch.com/wp-content/uploads/2018/01/keras_api_message_broker.png" width="500">

</center>

[Redis](https://redis.io/) is an in-memory data store. It is different than a simple `key/value` store (such as [memcached](https://memcached.org/)) as it can can store actual data structures.

> :bulb: **Memcache:** Free & open source, high-performance, distributed memory object caching system, generic in nature, but intended for use in speeding up dynamic web applications by alleviating database load.

Steps:


- Running Redis on our machine
- Queuing up data (images) to our Redis store to be processed by our REST API
- Polling Redis for new batches of input images
- Classifying the images and returning the results to the client

Read the full blog and have practical experience.

:atom_symbol: **Considerations when scaling your deep learning REST API**

- If you anticipate heavy load for extended periods of time on your deep learning REST API you may want to consider a load balancing algorithm such as round-robin scheduling to help evenly distribute requests across multiple GPU machines and Redis servers.
- Keep in mind that Redis is an **in-memory data store** so we can only store as many images in the queue we have available memory.
- A single `224 x 224 x 3` image with a float32 data type will consume $602112$ bytes of memory.
- Assuming a server with a modest $16$ GB of RAM, this implies that we can hold approximately $26500$ images in our queue, but at that point we likely would want to add more GPU servers to burn through the queue faster.

:warning: **Subtle issue: Multiple Model Syndrome:**

:dart: Depending on how you deploy your deep learning REST API, there is a subtle problem with keeping the `classify_process`
function in the same file as the rest of our web API code.

Most web servers, including Apache and nginx, allow for multiple client threads.

If you keep `classify_process` in the same file as your `predict`
view, then you may load multiple models if your server software deems it necessary to create a new thread to serve the incoming client requests — **for every new thread, a new view will be created, and therefore a new model will be loaded**.

**Solution:** The solution is to move classify_process
to an entirely separate process and then start it along with your Flask web server and Redis server. Follow second approach


## Second Approach: Deep learning in production with Keras, Redis, Flask, and Apache

_*Follow this amazing [blog](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/) by Adrian Rosebrock from pyimagesearch and try yourself_ :fire:

- The main idea is almost similar to first approach, but with code refactorring to avoid subtle memory issues.
- Also focus on the concept of `stress test`. 

:dart: There is a FastAPI + Docker vesion available for the second approach. Please follow this  [blog](https://medium.com/analytics-vidhya/deploy-machine-learning-models-with-keras-fastapi-redis-and-docker-4940df614ece). Try this [code](https://github.com/shanesoh/deploy-ml-fastapi-redis-docker).



<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>

----

# Exploring PUB-SUB Architecture with Redis & Python

## What is the PUB/SUB Pattern?

`Pub/Sub` aka Publish-Subscribe pattern is a pattern in which there are three main components, `sender`, `receiver` & a `broker`. The communication is processed by the broker, it helps the sender or publisher to publish information and deliver that information to the receiver or subscriber. 

## Test Environment

![image](https://miro.medium.com/max/1000/1*zC4JrtL2QfYDxGqzNbkzCg.png)


Start `redis-server`

`subscriber.py`

```py
import redis
# connect with redis server as Bob
bob_r = redis.Redis(host='localhost', port=6379, db=0)
bob_p = bob_r.pubsub()

# subscribe to channel: classical music
bob_p.subscribe('classical_music')
```

`publisher.py`

```py
# connect with redis server as Alice
alice_r = redis.Redis(host='localhost', port=6379, db=0)
# publish new music in the channel epic_music
alice_r.publish('classical_music', 'Raga Vairabi - Pt. Ajoy Chakrabarty')
```


By using the `publish` method Alice can now publish music on the `classical_music` Channel. Bob, on the other hand, can easily fetch publish music using the `get_message()` method:

```py
# ignore Bob subscribed message
bob_p.get_message()
# now bob can find alice’s music by simply using get_message()
new_music = bob_p.get_message()['data']
print(new_music)
```

**Reference:**

- [Basic Redis Usage Example - Part 1: Exploring PUB-SUB with Redis & Python](https://kb.objectrocket.com/redis/basic-redis-usage-example-part-1-exploring-pub-sub-with-redis-python-583)
- [Event Data Pipelines with Redis Pub/Sub, Async Python and Dash](https://itnext.io/event-data-pipelines-with-redis-pub-sub-async-python-and-dash-ab0a7bac63b0)


----


<a href="#Top"><img align="right" width="28" height="28" src="/assets/images/icon/arrow_circle_up-24px.svg" alt="Top"></a>
