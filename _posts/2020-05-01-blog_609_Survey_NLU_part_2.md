---
layout: post
title:  "Survey - Natural Language Understanding (NLU - Part 2)"
date:   2020-05-01 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

---

Quick Refresher on Natural Language Understanding

# NLU Task: Relation Extraction using distant supervision

- Output of the task is a `discrete object` rather than a numeric value.

## Core Reading

- Section 18.2.3 and 18.2.4  **Relation extraction by Bootstrapping and Distant Supervision** -  Book by Jurafsky, 3rd ed :fire: :fire:
- [Distant supervision for relation extraction without labeled data - Mintz et al. ACL2009](https://www.aclweb.org/anthology/P09-1113.pdf) :fire:
- [Learning Syntactic Patterns for Automatic Hypernym Discovery - Snow et al. NIPS2005](https://papers.nips.cc/paper/2659-learning-syntactic-patterns-for-automatic-hypernym-discovery)
- [Snorkel](https://www.snorkel.org/)


## Overview

Overview

This notebook illustrates an approach to [relation extraction](http://deepdive.stanford.edu/relation_extraction) using [distant supervision](http://deepdive.stanford.edu/distant_supervision). It uses a simplified version of the approach taken by Mintz et al. in their 2009 paper, `Distant supervision for relation extraction without labeled data`. Read the paper. Must.

## The task of relation extraction

Relation extraction is the task of extracting from natural language text relational triples such as:

- `(founders, SpaceX, Elon_Musk)`
- `(has_spouse, Elon_Musk, Talulah_Riley)`
- `(worked_at, Elon_Musk, Tesla_Motors)`

If we can accumulate a large knowledge base (KB) of **relational triples**, we can use it to **power question answering** and other applications. 

Building a KB manually is slow and expensive, but much of the knowledge we'd like to capture is already expressed in abundant text on the web. The aim of relation extraction, therefore, is to **accelerate the construction of new KBs** — and facilitate the ongoing curation of existing KBs — by extracting relational triples from natural language text.

- Huge number of human knowledge can be expressed in this form.
- Microsoft's KB `satori` powers Bing Search.


**What is WordNet?**

- It's a knowledge base of `lexical semantic relation`. Where role of entities are played by `words` or `synsets`. And the relation between them are `hypernym`, `synonym` or `antonym` relation. 


**Supervised learning:**


Effective relation extraction will require applying machine learning methods. The natural place to start is with supervised learning. This means training an extraction model from a dataset of examples which have been labeled with the target output.
The difficulty with the fully-supervised approach is the cost of generating training data. Because of the great diversity of linguistic expression, our model will need lots and lots of training data: at least tens of thousands of examples, although hundreds of thousands or millions would be much better. But labeling the examples is just as slow and expensive as building the KB by hand would be.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

## Distant supervision

![image](https://www.snorkel.org/doks-theme//assets/images/2017-07-16-weak-supervision/WS_mapping.png)

_- ***Distant Supervision** is a type of **Weak Supervision**_
- [image source: stanford blog](http://ai.stanford.edu/blog/weak-supervision/)

The goal of distant supervision is to capture the 

> benefits of supervised learning without paying the cost of labeling training data. 

Instead of labeling extraction examples by hand, we use existing relational triples (SME: subject matter expert) to automatically identify extraction examples in a large corpus. For example, if we already have in our KB the relational triple `(founders, SpaceX, Elon_Musk)`, we can search a large corpus for sentences in which "SpaceX" and "Elon Musk" co-occur, make the (unreliable!) **assumption:** that all the sentences express the founder relation, and then use them as training data for a learned model to identify new instances of the founder relation — all without doing any manual labeling.

This is a powerful idea, but it has two limitations. 
1. Some of the sentences in which "SpaceX" and "Elon Musk" co-occur will not express the founder relation — like the BFR example: 

> "Billionaire entrepreneur Elon Musk announced the latest addition to the SpaceX arsenal: the 'Big F---ing Rocket' (BFR)"

By making the blind assumption that all such sentences do express the founder relation, we are essentially **injecting noise into our training data**, and making it harder for our learning algorithms to learn good models. Distant supervision is effective in spite of this problem because it makes it possible to leverage **vastly greater quantities of training data**, and the benefit of more data outweighs the harm of noisier data.

2. We **need an existing KB** to start from. We can only train a model to extract new instances of the founders relation if we already have many instances of the founders relation. Thus, while distant supervision is a great way to extend an existing KB, it's not useful for creating a KB containing new relations from scratch.


:paperclip: **Reference:**

- [Blog on Data Programming](https://msank00.github.io/blog/2020/03/03/blog_602_Survey_data_programming#snorkel-rapid-training-data-creation-with-weak-supervision)
- [Section 18.2.3 and 18.2.4  Relation extraction by Bootstrapping -  Book by Jurafsky, 3rd ed] :fire: :fire:



----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>