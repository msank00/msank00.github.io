---
layout: post
title:  "Blog 202: Deep Learning Programming Tips"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

## **Deep Learning Best Practices - Mistakes and Tips**

The purpose of this repo is to consolidate all the **Best Practicies** for building neural network model curated over the internet

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

 Source:

 - [Github](https://github.com/msank00/datascience_bestpractices/blob/master/dl_mistakes_tips.md)

------

**Read the below references**

These are pure gold.

#### Reference

- [tweet_andrej_karpathy](https://twitter.com/karpathy/status/1013244313327681536)
- [Recipe for training neural network](https://karpathy.github.io/2019/04/25/recipe/)
- [What should I do when my neural network doesn't learn?](https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn)
- [Practical Advice for Building Deep Neural Networks](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/) 

---

## **Techinical Mistakes while Model Building**

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

#### Reference
- [tweet_Caitlin_Hudon](https://twitter.com/beeonaposy/status/1122964504910938121)
- [ICLR2019_Workshop_on_Debug_ML](https://github.com/debug-ml-iclr2019/debug-ml-iclr2019.github.io)

---



## **9 Tips For Training Lightning-Fast Neural Networks In Pytorch**

- [Blog](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565)
---

## How to write training loop in PyTorch?

- [notebook](https://nbviewer.jupyter.org/github/msank00/deeplearning_4m_scratch/blob/master/03_minibatch_training.ipynb)

## How to use `checkpoint` in your code?

### What is checkpoint?


- The architecture of the model, allowing you to re-create the model
- The weights of the model
- The training configuration (loss, optimizer, epochs, and other meta-information)
- The state of the optimizer, allowing to resume training exactly where you left off.

> Again, a checkpoint contains the information you need to save your current experiment state so that you can resume training from this point.

### How to save and load checkpoint in Pytorch?

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


**Resource:**
- [saving-loading-your-model-in-pytorch](https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c)
- [checkpointing-tutorial-for-tensorflow-keras-and-pytorch](https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/)
- [Notebook-Github](https://github.com/msank00/nlproc/blob/master/text_classification_pytorch_v1.ipynb)

---
## Text Classification in Pytorch

PyTorchNLPBook by Delip Rao, Chapter 3

- [Classifying_Yelp_Review_Sentiment](https://nbviewer.jupyter.org/github/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_3/3_5_Classifying_Yelp_Review_Sentiment.ipynb)


---

## **Dive into Deep Learning with PyTroch**

- [Original Book using mxnet](https://www.d2l.ai/)
- [PyTorch Version](https://github.com/dsgiitr/d2l-pytorch)

---

## **How to design and debug deep learning models?**

```py
(1/4) Learning ML engineering is a long slog even for legendary hackers like @gdb
. 

IMO, the two hardest parts of ML eng are:

1) Feedback loops are measured in minutes or days in ML (compared to seconds in normal eng)
2) Errors are often silent in ML
```

```py
(2/4) Most ML people deal with silent errors and slow feedback loops via the "ratchet" approach:

1) Start with known working model
2) Record learning curves on small task (~1min to train)
3) Make a tiny code change
4) Inspect curves
5) Run full training after ~5 tiny changes
```

```py
(3/4) Downside of ratchet approach is some designs cant be reached via small incremental changes. Also hard to know *which* tiny code changes to make.
```

```py
(4/4) Within the ratchet approach, I want more tools and best practices for making feedback loops shorter and for making errors louder.

Below is a short list of development speed hacks that I have found useful.
```

### **ML dev speed hack #0 - Overfit a single batch**
- Before doing anything else, verify that your model can memorize the labels for a single batch and quickly bring the loss to zero
- This is fast to run, and if the model can't do this, then you know it is broken

### **ML dev speed hack #1 - PyTorch over TF**
- Time to first step is faster b/c no static graph compilation
- Easier to get loud errors via assertions within the code
- Easier to drop into debugger and inspect tensors
- (TF2.0 may solve some of these problems but is still raw)

### **ML dev speed hack #2 - Assert tensor shapes**
- Wrong shapes due to silent broadcasting or reduction is an extreme hot spot for silent errors, asserting on shapes (in torch or TF) makes them loud
- If you're ever tempted to write shapes in a comment, make an assert instead

### **ML dev speed hack #3 - Add ML test to CI**
- If more than one entrypoint or more than one person working on the codebase, then add a test that runs for N steps and then checks loss
- If you only have one person and entrypoint then an ML test in CI is probably overkill


### **ML dev speed hack #4 - Use `ipdb.set_trace()`**
- It's hard to make an ML job take less than 10 seconds to start, which is too slow to maintain flow
- Using the ipdb workflow lets you zero in on a bug and play with tensors with a fast feedback loop

### **ML dev speed hack #5 - Use `nvvp` to debug throughput**
- ML throughput (step time) is one place where we have the tools to make errors loud and feedback fast
- You can use `torch.cuda.nvtx.range_push` to annotate the nvvp timeline to be more readable



Source:

- [Twitter Thread](https://twitter.com/nottombrown/status/1156350020351713281)
- [how-i-became-a-machine-learning-practitioner](https://blog.gregbrockman.com/how-i-became-a-machine-learning-practitioner)
- [Youtube: Troubleshooting Deep Neural Networks - Full Stack Deep Learning](https://www.youtube.com/watch?time_continue=8&v=GwGTwPcG0YM)
- [Youtube: Full Stack Deep Learning](https://www.youtube.com/channel/UCVchfoB65aVtQiDITbGq2LQ/videos)

----

## Some useful blogs:

- [Understanding binary cross-entropy / log loss: a visual explanation](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
