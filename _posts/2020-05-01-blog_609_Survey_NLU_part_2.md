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
- [Stanford: Notebook Relation Extraction Part 1](https://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/rel_ext_01_task.ipynb) :fire: :fire:


## Overview


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

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Entity Extraction

- While extracting entities, direction matters.

```py
corpus.show_examples_for_pair('Elon_Musk', 'Tesla_Motors')
corpus.show_examples_for_pair('Tesla_Motors', 'Elon_Musk')
``` 

Check **both directions** when we're looking for examples contains a specific pair of entities.

- Connect the `corpus` with the `knowledge base`. The corpus tells nothing except the english words.

> "...Elon Musk is the founder of Tesla Motors..."

To get the information from the free text, connect with the knowledge base. 

## Knowledge Base

For any entity extraction work, you need to have a Knowledge Base (KB). Earlier there was the knowledge base [Freebase](https://en.wikipedia.org/wiki/Freebase). Unfortunately, Freebase was shut down in $2016$, but the Freebase data is still available from various sources and in various forms. Check the [Freebase Easy data dump](http://freebase-easy.cs.uni-freiburg.de/dump/).

**How the knowledge base looks like?**

The KB is a collection of relational triples, each consisting of a relation, a subject, and an object. For example, here are three triples from the KB:


- The relation is one of a handful of predefined constants, such as place_of_birth or has_spouse.
- The subject and object are entities represented by Wiki IDs (that is, suffixes of Wikipedia URLs).

The freebase kowledge base stats:

- 45,884 KB triples
- 16 types of relations

**Example:**

```py
for rel in kb.all_relations:
    print(tuple(kb.get_triples_for_relation(rel)[0]))

('adjoins', 'France', 'Spain')
('author', 'Uncle_Silas', 'Sheridan_Le_Fanu')
('capital', 'Panama', 'Panama_City')
('contains', 'Brickfields', 'Kuala_Lumpur_Sentral_railway_station')
('film_performance', 'Colin_Hanks', 'The_Great_Buck_Howard')
('founders', 'Lashkar-e-Taiba', 'Hafiz_Muhammad_Saeed')
('genre', '8_Simple_Rules', 'Sitcom')
('has_sibling', 'Ari_Emanuel', 'Rahm_Emanuel')
('has_spouse', 'Percy_Bysshe_Shelley', 'Mary_Shelley')
('is_a', 'Bhanu_Athaiya', 'Costume_designer')
('nationality', 'Ruben_Rausing', 'Sweden')
('parents', 'Rosanna_Davison', 'Chris_de_Burgh')
('place_of_birth', 'William_Penny_Brookes', 'Much_Wenlock')
('place_of_death', 'Jean_Drapeau', 'Montreal')
('profession', 'Rufus_Wainwright', 'Actor')
('worked_at', 'Brian_Greene', 'Columbia_University')
```

### Limitation

> Note that there is no promise or expectation that this KB is complete. 

Not only does the KB contain no mention of many entities from the corpus — even for the entities it does include, there may be possible triples which are true in the world but are missing from the KB. As an example, these triples are in the KB:

```py
# (founders, SpaceX, Elon_Musk)
# (founders, Tesla_Motors, Elon_Musk)
# (worked_at, Elon_Musk, Tesla_Motors)
```

but this one is not:

```py
# (worked_at, Elon_Musk, SpaceX)
```

In fact, the whole point of developing methods for automatic relation extraction is to **extend existing KBs** (and build new ones) by identifying new relational triples from natural language text. If our KBs were complete, we wouldn't have anything to do.


### Joining the corpus and the KB

In order to leverage the distant supervision paradigm, we'll need to connect information in the corpus with information in the KB. There are two possibilities, depending on how we formulate our prediction problem:

- **Use the KB to generate labels for the corpus**. If our problem is to classify a pair of entity mentions in a specific example in the corpus, then we can use the KB to provide labels for training examples. Labeling specific examples is how the fully supervised paradigm works, so it's the obvious way to think about leveraging distant supervision as well. Although it can be made to work, it's not actually the preferred approach.
- **Use the corpus to generate features for entity pairs**. If instead our problem is to classify a pair of entities, then we can use all the examples from the corpus where those two entities co-occur to generate a feature representation describing the entity pair. This is the approach taken by Mintz et al. 2009, and it's the approach we'll pursue here.

So we'll formulate our prediction problem such that the 
- **Input** is a pair of entities
- The goal is to **predict what relation(s) the pair** belongs to. 
- The `KB` will provide the labels
- The `corpus` will provide the features.


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# Problem formulation

We need to specify:

- What is the **input** to the prediction?
  - Is it a specific pair of entity mentions in a specific context?
  - Or is it a pair of entities, apart from any specific mentions?
- What is the **output** of the prediction?
  - Do we need to predict at most one relation label? (This is `multi-class classification`.)
  - Or can we predict multiple relation labels? (This is `multi-label classification`.)


**Multi-label classification**

A given pair of entities can belong to more than one relation. In fact, this is quite common in any KB.

```py
dataset.count_relation_combinations()

The most common relation combinations are:
      1216 ('is_a', 'profession')
       403 ('capital', 'contains')
       143 ('place_of_birth', 'place_of_death')
        61 ('nationality', 'place_of_birth')
        11 ('adjoins', 'contains')
         9 ('nationality', 'place_of_death')
         7 ('has_sibling', 'has_spouse')
         3 ('nationality', 'place_of_birth', 'place_of_death')
         2 ('parents', 'worked_at')
         1 ('nationality', 'worked_at')
         1 ('has_spouse', 'parents')
         1 ('author', 'founders')
```

Multiple relations per entity pair is a commonplace phenomenon.

**Solution:**

- [Binary relevance method](https://en.wikipedia.org/wiki/Multi-label_classification#Problem_transformation_methods) : which just factors multi-label classification over $n$ labels into $n$ **independent binary classification** problems, one for each label. A **disadvantage** of this approach is that, by treating the binary classification problems as independent, it **fails to exploit correlations between labels**. But it has the great virtue of simplicity, and it will suffice for our purposes.


## Building datasets

We're now in a position to write a function to build datasets suitable for training and evaluating predictive models. These datasets will have the following characteristics:

Because we've formulated our problem as multi-label classification, and we'll be training separate models for each relation, we won't build a single dataset. Instead, we'll build a dataset for each relation, and our return value will be a map from relation names to datasets.
The dataset for each relation will consist of **two parallel lists**:
- A list of `candidate KBTriples` which combine the given relation with a pair of entities.
- A corresponding `list of boolean labels` indicating whether the given KBTriple belongs to the KB.

The dataset for each relation will include KBTriples derived from two sources:
- Positive instances will be drawn from the KB.
- Negative instances will be sampled from unrelated entity pairs, as described above.




:paperclip: **Reference:**

- [Blog on Data Programming](https://msank00.github.io/blog/2020/03/03/blog_602_Survey_data_programming#snorkel-rapid-training-data-creation-with-weak-supervision)
- [Section 18.2.3 and 18.2.4  Relation extraction by Bootstrapping -  Book by Jurafsky, 3rd ed] :fire: :fire:
- [Stanford: Notebook Relation Extraction Part 1](https://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/rel_ext_01_task.ipynb) :fire: :fire:
- :movie_camera: [Stanford Lecture CS224U](https://www.youtube.com/watch?v=pO3Jsr31s_Q&list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20&index=7) :rocket:



----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>