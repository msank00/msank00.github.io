---
layout: post
title:  "Survey - GAN"
date:   2020-03-01 00:00:10 -0030
categories: jekyll update
mathjax: true
---


# Content

1. TOC
{:toc}

----

# Understanding Generative Adversarial Networks

A GAN is comprised of two neural networks — 
1. **Generator** that synthesizes new samples from scratch
2. **Discriminator** that compares training samples with these generated samples from the generator. 

The discriminator’s goal is to distinguish between `real` and `fake` inputs (ie. classify if the samples came from the model distribution or the real distribution). As we described, these samples can be images, videos, audio snippets, and text.

<center>
<img src="https://miro.medium.com/proxy/1*KF-XzsW2F44sCxlgdDy_9w.png" height="200">
</center>

<center>
<img src="https://guimperarnau.com/files/blog/Fantastic-GANs-and-where-to-find-them/GAN_training_overview.jpg" height="200">
</center>

- At first, the generator generates images. It does this by sampling a vector noise $Z$ from a simple distribution (e.g. normal) and then upsampling this vector up to an image. In the first iterations, these images will look very noisy. 
- Then, the discriminator is given fake and real images and learns to distinguish them. 
- The generator later receives the “feedback” of the discriminator through a backpropagation step, becoming better at generating images. 
- At the end, we want that the distribution of fake images is as close as possible to the distribution of real images. Or, in simple words, we want fake images to look as plausible as possible.

It is worth mentioning that due to the **minimax optimization** used in GANs, the training might be quite unstable. There are some hacks, though, that you can use for a more robust training.

<center>
<img src="/assets/images/image_34_gen_model_1.png" alt="image" height="300">
</center>

<center>
<img src="/assets/images/image_34_gen_model_2.png" alt="image" height="300">
</center>


<center>
<img src="/assets/images/image_34_gen_model_3.png" alt="image" height="300">
</center>

<center>
<img src="/assets/images/image_34_gen_model_4.png" alt="image" height="300">
</center>

<center>
<img src="/assets/images/image_34_gen_model_5.png" alt="image" height="300">
</center>

<center>
<img src="/assets/images/image_34_gen_model_6.png" alt="image" height="300">
</center>

<center>
<img src="/assets/images/image_34_gen_model_7.png" alt="image" height="300">
</center>



**Reference:**

- [Blog](https://towardsdatascience.com/graduating-in-gans-going-from-understanding-generative-adversarial-networks-to-running-your-own-39804c283399)
- [GAN implementation](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9)
- [Fantastic-GANs-and-where-to-find-them](https://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them)
- [Introduction to Generative Adversarial Networks (GANs):](https://heartbeat.fritz.ai/introduction-to-generative-adversarial-networks-gans-35ef44f21193)
- [Stanford Slide](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Implementation

There are really only 5 components to think about:

- **R:** The original, genuine data set

In our case, we’ll start with the simplest possible R — a bell curve. This function takes a mean and a standard deviation and returns a function which provides the right shape of sample data from a Gaussian with those parameters.

<center>
<img src="https://miro.medium.com/max/716/1*xsuE-nhsJOzk9lfI3rayuw.png" alt="image" width="500">
</center>


- **I:** The random noise that goes into the generator as a source of entropy

The input into the generator is also random, but to make our job a little bit harder, let’s use a uniform distribution rather than a normal one. This means that our model G can’t simply shift/scale the input to copy R, but has to reshape the data in a non-linear way.


<center>
<img src="https://miro.medium.com/max/427/1*wuhEVnK25V3zXQzuCwFDAg.png" alt="image" width="300">
</center>



- **G:** The generator which tries to copy/mimic the original data set

The generator is a standard feedforward graph — two hidden layers, three linear maps. We’re using a hyperbolic tangent activation function ‘cuz we’re old-school like that. G is going to get the uniformly distributed data samples from I and somehow mimic the normally distributed samples from R — without ever seeing R.

<center>
<img src="https://miro.medium.com/max/928/1*ZWdLJE92goGCO2IckGz3tA.png" alt="image" width="500">
</center>


- **D:** The discriminator which tries to tell apart G’s output from R

The discriminator code is very similar to G’s generator code; a feedforward graph with two hidden layers and three linear maps. The activation function here is a sigmoid — nothing fancy, people. It’s going to get samples from either R or G and will output a single scalar between 0 and 1, interpreted as ‘fake’ vs. ‘real’. In other words, this is about as milquetoast as a neural net can get.



<center>
<img src="https://miro.medium.com/max/932/1*k92BAYSiIn49Q2sTUWnVtw.png" alt="image" width="500">
</center>


- The actual ‘training’ loop where we teach G to trick D and D to beware G.

Finally, the training loop alternates between two modes: first training D on real data vs. fake data, with accurate labels (think of this as Police Academy); and then training G to fool D, with inaccurate labels (this is more like those preparation montages from Ocean’s Eleven). It’s a fight between good and evil, people.


<center>
<img src="https://miro.medium.com/max/2104/1*gNhL1T1dr4YXCTI1B5U03A.png" alt="image" width="700">
</center>

**Reference:**

- [GAN Implementation in 50 line PyTorch](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>