---
layout: post
title:  "Survey - Model Explainability"
date:   2020-03-05 00:00:10 -0030
categories: jekyll update
mathjax: true
---


# Content

1. TOC
{:toc}

----
# Introduction

According to this [blog](https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726):

Autoencoder is an **unsupervised artificial neural network** that learns how to efficiently **compress and encode** data then **learns how to reconstruct** the data back from the reduced encoded representation to a representation that is as close to the original input as possible.


> Autoencoder, by design, reduces data dimensions by learning how to ignore the noise in the data.


<center>
<img src="https://miro.medium.com/max/700/1*P7aFcjaMGLwzTvjW3sD-5Q.jpeg" width="500">
</center>

From architecture point of view, it looks like this:

<center>
<img src="https://miro.medium.com/max/1096/1*ZEvDcg1LP7xvrTSHt0B5-Q@2x.png" width="300">
</center>

# Implementation

According to this wonderful [notebook](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb) we can build the autoencoder as follows 

```py
##########################
### MODEL
##########################

class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):
        super(Autoencoder, self).__init__()
        
        ### ENCODER
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        # The following to lones are not necessary, 
        # but used here to demonstrate how to access the weights
        # and use a different weight initialization.
        # By default, PyTorch uses Xavier/Glorot initialization, which
        # should usually be preferred.
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        
        ### DECODER
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        

    def forward(self, x):
        
        ### ENCODER
        encoded = self.linear_1(x)
        encoded = F.leaky_relu(encoded)
        
        ### DECODER
        logits = self.linear_2(encoded)
        decoded = torch.sigmoid(logits)
        
        return decoded

    
torch.manual_seed(random_seed)
model = Autoencoder(num_features=num_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

```

# Theory

Listen to the [lecture 7](https://www.cse.iitm.ac.in/~miteshk/CS7015.html) by Prof. Mitesh 

**Reference:**

- [Very Important Lecture 7, both slide and video](https://www.cse.iitm.ac.in/~miteshk/CS7015.html)

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>
