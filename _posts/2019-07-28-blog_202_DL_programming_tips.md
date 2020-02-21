---
layout: post
title:  "Deep Learning Programming Tips, Best Practices"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---


# Deep Learning Best Practices - Mistakes and Tips:

The purpose of this repo is to consolidate all the **Best Practices** for building neural network model curated over the internet

- Try to overfit a single batch first
  - It's a very quick sanity test of your wiring; i.e. if you can't overfit a small amount of data you've got a simple bug somewhere 
  - it's by far the most "bang for the buck" trick that noone uses that exists.
5 replies 7 retweets 219 likes
- Forgot to toggle train/eval mode for the net
- Forgot to `.zero_grad()` (in pytorch) before `.backward()`.
- Passed `softmaxed outputs` to a loss that expects `raw logits`.
- You didn't use `bias=False` for your `Linear/Conv2d` layer when using `BatchNorm`, or conversely forget to include it for the output layer .This one won't make you silently fail, but they are spurious parameters
- Thinking `view()` and `permute()` are the same thing (& incorrectly using view)
- starting with `small model` + `small amount of data` & growing both together; I always find it really insightful
  - I like to start with the simplest possible sanity checks - e.g. also training on all zero data first to see what loss I get with the base output distribution, then gradually include more inputs and scale up the net, making sure I beat the previous thing each time.
- ...

 **Reference**

These are pure gold.

- [Tweet_andrej_karpathy](https://twitter.com/karpathy/status/1013244313327681536)
- [Recipe for training neural network](https://karpathy.github.io/2019/04/25/recipe/)
- [What should I do when my neural network doesn't learn?](https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn)
- [Practical Advice for Building Deep Neural Networks](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/) 


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# Technical Mistakes while Model Building

- Create a `non-reproducible` data preparation steps
- Evaluate a model based on performance of training set
- Didn't notice `large outlier`
- Dropped missing values when it made sense to flag them
- Flagged missing values when it made sense to drop them
- Set missing values to Zero
- Not comparing a complex model to a **simple baseline**
- Failed to understand nuances of data collection
- Build model for **wrong point in time**
- Deleted records with missing values
- Predicted the wrong outcome
- Made **faulty assumptions** about `time zones`
- Made **faulty assumptions** about `data format`
- Made **faulty assumptions** about `data source`
- Included `anachronistic` (belonging to a period other than that being portrayed) variables
- Treated categorical variables as continuous
- Treated continuous variables as categorical
- Filtered training set to **incorrect population**
- Forgot to include `y-variable` in the training set
- Didn't look at **number of missing** values in column
- Not filtering for **duplicates** in the dataset
- Accidently included `ID` field as predictors
- Failing to bin or account for **rare categories**
- Using proxies of outcomes as predictors
- Incorrect handling of `missing values`
- Capped outliers in a way that didn't make sense with data
- **Misunderstanding of variable** relationships due to incomplete EDA
- Failed to create calculated variables from raw data
- Building model on the wrong population

**Reference:**

- [Tweet_Caitlin_Hudon](https://twitter.com/beeonaposy/status/1122964504910938121)
- [ICLR2019_Workshop_on_Debug_ML](https://github.com/debug-ml-iclr2019/debug-ml-iclr2019.github.io)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


---

# Software Engineering Skills for Data Science

> Because our day-to-day involves writing code, I am convinced that we data scientists need to be equipped with basic software engineering skills. Being equipped with these skills will help us write code that is, in the long-run, easy to recap, remember, reference, review, and rewrite. In this collection of short essays, I will highlight the basic software skills that, if we master, will increase our efficiency and effectiveness in the long-run.

**Reference:**

- [Essays on Data Science](https://ericmjl.github.io/essays-on-data-science/software-skills/)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


---

# Pytorch Learning

- [JovianML: Pytorch Basics](https://jovian.ml/aakashns/01-pytorch-basics)
- [IMP Pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
- [JovianML: Linear Regression in Pytorch](https://jovian.ml/aakashns/02-linear-regression)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Difference of `Conv1D` and `Conv2D` in deep learning.

**Conv2D**

In simple terms, images have shape `(height, width)`. So a filter can move in 2 direction, so `conv2D` is used.

![image](https://miro.medium.com/max/700/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif)

In the above image, the green matrix is the `kernel` convolving on `2 direction` over the image and creating the red `feature map` in the right

**Conv1D**
But in case of text, initially the text are converted to some fixed dimension vectors `one-hot-encoded` or `dense embedding` of fixed dimension. Where the filters can move in one direction only, i.e in the direction of the words or characters, but not in the corresponding embedding dimension because it's fixed.   

![image](https://debajyotidatta.github.io/assets/images/conv.001.png)

- So the green boxes represent the words or the characters depending on your approach. 
- And the corresponding blue rows shows the vector representation (one-hot-encoding or embedding) of the words or the characters.

Here is a corresponding kernel whose `height=kernel size` but it's `widht=embedding_dim` which is fixed. 

![kernel](https://debajyotidatta.github.io/assets/images/conv.002.png)

So the above kernel can move along the direction of the words or characters, i.e, along the green boxes in the previous image.

> **Convolve is a fancy term for multiplication with corresponding cells and adding up the sum.** 

It varies based on things like 1. `stride` (How much the filter moves every stage?) and the 2. `length` of the filter. The output of the convolution operation is directly dependent on these two aspects. 

_first convolution_

![image](https://debajyotidatta.github.io/assets/images/conv.003.png)

_last convolution_

![image](https://debajyotidatta.github.io/assets/images/conv.006.png)

See after each `stride` a single cell is generated at the right and after the full pass, a 1D vector is generated.

Now if multiple convolution filters are used, then multiple such `1D vectors` will be generated. Then you do `maxpooling` to get the `max element` from each such `1D` vector and then soncatenate and finally apply `softmax`.

_multiple feature maps due to multiple kernels_

![image](https://debajyotidatta.github.io/assets/images/conv2.006.png)

_max pooling and concatenation_

![image](https://debajyotidatta.github.io/assets/images/conv2.007.png)

The entire process was very nicely illustrated by Zhang et al, in the paper “A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification”, for words.


![image](https://debajyotidatta.github.io/assets/images/Zhang.png)


**Reference:**

- [Understanding Convolutions in Text](https://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/11/27/Understanding-Convolutions-In-Text/)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# 9 Tips For Training Lightning-Fast Neural Networks In Pytorch

- [Blog](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


---

# How to write training loop in PyTorch?

- [notebook](https://nbviewer.jupyter.org/github/msank00/deeplearning_4m_scratch/blob/master/03_minibatch_training.ipynb)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# How to use `checkpoint` in your code?

## What is checkpoint?


- The architecture of the model, allowing you to re-create the model
- The weights of the model
- The training configuration (loss, optimizer, epochs, and other meta-information)
- The state of the optimizer, allowing to resume training exactly where you left off.

> Again, a checkpoint contains the information you need to save your current experiment state so that you can resume training from this point.

## How to save and load checkpoint in Pytorch?

```py
#Saving a checkpoint
torch.save(checkpoint, ‘checkpoint.pth’)#Loading a checkpoint
checkpoint = torch.load( ‘checkpoint.pth’)
```

> A checkpoint is a python dictionary that typically includes the following:

1. **Network structure:** input and output sizes and Hidden layers to be able to reconstruct the model at loading time.
2. **Model state dict:** includes parameters of the network layers that is learned during training, you get it by calling this method on your model instance.
`model.state_dict()`
3. **Optimizer state dict:** In case you are saving the latest checkpoint to continue training later, you need to save the optimizer’s state as well.
you get it by calling this method on an optimizer’s instance `optimizer.state_dict()`
4. Additional info: You may need to store additional info, like number of epochs and your class to index mapping in your checkpoint.

```py
#Example for saving a checkpoint assuming the network class named #Classifier
checkpoint = {'model': Classifier(),
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
```

```py
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint('checkpoint.pth')
```


**Reference:**
- [saving-loading-your-model-in-pytorch](https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c)
- [checkpointing-tutorial-for-tensorflow-keras-and-pytorch](https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/)
- [Notebook-Github](https://github.com/msank00/nlproc/blob/master/text_classification_pytorch_v1.ipynb)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# Text Classification in Pytorch

PyTorchNLPBook by Delip Rao, Chapter 3

- [Classifying_Yelp_Review_Sentiment](https://nbviewer.jupyter.org/github/msank00/nlproc/blob/master/Classifying_Yelp_Reviews.ipynb)



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

---

# Dive into Deep Learning with PyTroch

- [Original Book using mxnet](https://www.d2l.ai/)
- [PyTorch Version](https://github.com/dsgiitr/d2l-pytorch)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


---

# Programming Tips for RNN, LSTM, GRU in Pytorch


## LSTM


![image](/assets/images/image_24_lstm_1.png)

**Problem definition:** Given family name, identify nationality

```py
class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input_, hidden):
        """
        input_.view(ni, nb, -1): 
            ni: Number of inputs given simultaneously
                (this helps in vectorization)
            nb: number of batches
            -1: rest, here number of characters 
        """
        
        out, hidden = self.lstm_cell(input_.view(1, 1, -1), hidden)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output.view(1, -1), hidden
    
    def init_hidden(self):
        """
        Return:
            Tuple of size 2
            - initializing hidden state
            - initializing cell state
        """
        # dim: n_layers x n_batches x hid_dim
        init_hidden_state = torch.zeros(1, 1, self.hidden_size)

        # dim: n_layers x n_batches x hid_dim
        init_cell_state = torch.zeros(1, 1, self.hidden_size) 
        return (init_hidden_state, init_cell_state)

n_hidden = 128
net = LSTM_net(n_letters, n_hidden, n_languages)
```

- one parameter update per batch

**code source**: Deep Learning course from PadhAI, IIT M, Module: Sequence Model, Lecture: Sequence model in Pytorch

From PyTorch `nn.LSTM` [documentation](https://pytorch.org/docs/stable/nn.html)

```py
torch.nn.LSTM(*args, **kwargs)
```

Applies a **multi-layer** `long short-term memory` (LSTM) RNN to an input sequence.



- $i_t​=\sigma(W_{ii}​x_t​+b_{ii}​+W_{hi}​h_{(t−1)}​+b_{hi}​)$
- $f_t​=\sigma(W_{if}​x_t​+b_{if}​+W_{hf}​h_{(t−1)}​+b_{hf}​)$
- $g_t​=\tanh(W_{ig}​x_t​+b_{ig}​+W_{hg}​h_{(t−1)}​+b_{hg}​)$
- $o_t​=\sigma(W_{io}​x_t​+b_{io}​+W_{ho}​h_{(t−1)}​+b_{ho}​)$
- $c_t​=f_t​∗c_{(t−1)​}+i_t​∗g_t$
- $h_t = o_t*\tanh(c_t)$

**Parameters:**

- `input_size` – The number of expected features in the input `x`
- `hidden_size` – The number of `features/neurones` in the hidden state `h`
- `num_layers` – Number of recurrent layers. E.g., setting `num_layers=2` would mean **stacking two LSTMs** together to form a `stacked LSTM`, with the second LSTM taking in outputs of the first LSTM and computing the final results. `Default: 1`


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# What's the difference between `hidden` and `output` in PyTorch LSTM?

According to Pytorch documentation 

```py
"""
Outputs: output, (h_n, c_n)
"""
```

- `output (seq_len, batch, hidden_size * num_directions)`: Tensor containing the output features (h_t) from the last layer of the RNN, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
- `h_n (num_layers * num_directions, batch, hidden_size)`: tensor containing the hidden state for t=seq_len
- `c_n (num_layers * num_directions, batch, hidden_size)`: tensor containing the cell state for t=seq_len


## How to interpret it?

output comprises all the hidden states in the last layer ("last" depth-wise, not time-wise). $(h_n, c_n)$ comprises the hidden states after the last time step, $t = n$, so you could potentially feed them into another LSTM.

![image](https://i.stack.imgur.com/SjnTl.png)

The batch dimension is not included.

- [source_stackOverflow](https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm)


**Remember**

For each element in the input sequence, each layer computes the following function:



- The `RNN_Net` and the `LSTM_Net` should be equivalent from outside, i.e their `function signature` should be equivalent meaning their input and output signature are equivalent even if their internal mechanism is different

```py
"""
input_dim:int = size of the input vectors depending on problem definition. It can be number of words or number of characters etc.
hid_dim:int = size of the hidden dimension, i.e number of neurons in the hidden layer, you SHOULD NOT interpret this as number of hidden layer
output_dim:int = size of the output, it's mostly size of the multi-class vector, e.g: number of language, number of sentiments etc.
"""
net_rnn = RNN_net(input_dim, hidden_dim, output_dim)
net_lstm = RNN_net(input_dim, hidden_dim, output_dim)
```

- Both should return `output` and `hidden` state


**Reference:**

- Deep Learning course from PadhAI, IIT M, Module: Sequence Model, Lecture: Sequence model in Pytorch


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# How to design and debug deep learning models?


**[1/4]** Learning ML engineering is a long slog even for legendary hackers like @gdb 

IMO, the two hardest parts of ML eng are:

1. Feedback loops are measured in minutes or days in ML (compared to seconds in normal eng)
2. Errors are often silent in ML



**[2/4]** Most ML people deal with silent errors and slow feedback loops via the `ratchet` approach:

1. Start with known working model
2. Record learning curves on small task (~1min to train)
3. Make a tiny code change
4. Inspect curves
5. Run full training after ~5 tiny changes



**[3/4]** Downside of ratchet approach is some designs cant be reached via small incremental changes. Also hard to know **which** tiny code changes to make.



**[4/4]** Within the ratchet approach, I want more tools and best practices for making feedback loops shorter and for making errors louder.

Below is a short list of development speed hacks that I have found useful.


## ML dev speed hack #0 - Overfit a single batch

- Before doing anything else, verify that your model can memorize the labels for a single batch and quickly bring the loss to zero
- This is fast to run, and if the model can't do this, then you know it is broken

## ML dev speed hack #1 - PyTorch over TF

- Time to first step is faster b/c no static graph compilation
- Easier to get loud errors via assertions within the code
- Easier to drop into debugger and inspect tensors
- (TF2.0 may solve some of these problems but is still raw)

## ML dev speed hack #2 - Assert tensor shapes

- Wrong shapes due to silent broadcasting or reduction is an extreme hot spot for silent errors, asserting on shapes (in torch or TF) makes them loud
- If you're ever tempted to write shapes in a comment, make an assert instead

## ML dev speed hack #3 - Add ML test to CI

- If more than one entry point or more than one person working on the codebase, then add a test that runs for N steps and then checks loss
- If you only have one person and entry point then an ML test in CI is probably overkill


##  ML dev speed hack #4 - Use `ipdb.set_trace()`

- It's hard to make an ML job take less than 10 seconds to start, which is too slow to maintain flow
- Using the ipdb workflow lets you zero in on a bug and play with tensors with a fast feedback loop

## ML dev speed hack #5 - Use `nvvp` to debug throughput

- ML throughput (step time) is one place where we have the tools to make errors loud and feedback fast
- You can use `torch.cuda.nvtx.range_push` to annotate the nvvp timeline to be more readable



**Reference:**

- [Twitter Thread](https://twitter.com/nottombrown/status/1156350020351713281)
- [how-i-became-a-machine-learning-practitioner](https://blog.gregbrockman.com/how-i-became-a-machine-learning-practitioner)
- [Youtube: Troubleshooting Deep Neural Networks - Full Stack Deep Learning](https://www.youtube.com/watch?time_continue=8&v=GwGTwPcG0YM)
- [Youtube: Full Stack Deep Learning](https://www.youtube.com/channel/UCVchfoB65aVtQiDITbGq2LQ/videos)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Pytorch RNN tips, set `batch_first=True`:

Always set `batch_first=True` while implementing RNN using PyTorch RNN module. 

**Reference:**

- [Beginners guide on RNN](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Some useful blogs:

- [Understanding binary cross-entropy / log loss: a visual explanation](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>