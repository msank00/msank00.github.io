---
layout: post
title:  "Blog 106: ML System Design"
date:   2019-11-30 00:11:31 +0530
categories: jekyll update
mathjax: true
---

## System design interview question strategy?

- Define the problem
- High level design only
- Tackle from different angle
  - Scale 
  - Latency
  - Response time
  - Database management
  - ...


Formal Way:

- Step 1 — Understand the Goals
  - What is the goal of the system?
  - Who are the users of the system? What do they need it for? How are they going to use it?
  - What are the inputs and outputs of the system?
- Step 2 — Establish the Scope [Ask clarifying questions, such as:]
  - Do we want to discuss the end-to-end experience or just the API?
  - What clients do we want to support (mobile, web, etc)?
  - Do we require authentication? Analytics? Integrating with existing systems?
- Step 3 — Design for the Right Scale
  - What is the expected read-to-write ratio?
  - How many concurrent requests should we expect?
  - What’s the average expected response time?
  - What’s the limit of the data we allow users to provide?
- Step 4 — Start High-Level, then Drill-Down
  - User interaction
  - External API calls
  - Offline processes
- Step 5 — Data Structures and Algorithms (DS&A)
  - URL shortener? Makes me think of a hashing function. 
  - Oh, you need it to scale? Sharding might help
  - Concurrency? 
  - Redundancy? 
  - Generating keys becomes even more complicated.
- Step 6 — Tradeoffs
  - What type of database would you use and why?
  - What caching solutions are out there? Which would you choose and why?
  - What frameworks can we use as infrastructure in your ecosystem of choice?
  - 



### Technology to focus

- Horizontal Scaling
- Vertical Scaling
- Intelligent Caching
- Datadase:
  - [NoSQL](https://www.guru99.com/nosql-tutorial.html)
    - **Key Value Pair Based**: Key value stores help the developer to store schema-less data. They work best for `shopping cart contents`. Redis, `Dynamo`, `Riak` are some examples of key-value store DataBases. They are all based on Amazon's Dynamo paper. 
    - **Column-based**: Column-based NoSQL databases are widely used to manage `data warehouses`, business intelligence, CRM, Library card catalogs. HBase, `Cassandra`, HBase, Hypertable
    - **Document-Oriented**: The document type is mostly used for CMS systems, `blogging platforms`, real-time analytics & `e-commerce applications`. Amazon SimpleDB, CouchDB, `MongoDB`, Riak, Lotus Notes, MongoDB
    - **Graph-Based**: Graph base database mostly used for `social networks`, logistics, `spatial data`. Neo4J, Infinite Graph, `OrientDB`, FlockDB.


### Algorithm to Focus:

- Ranking Algorithm
- Searching Algorithm
- Similarity Score
- Recommendation Algo

**Resource:**

- [how-to-succeed-in-a-system-design-interview](https://blog.pramp.com/how-to-succeed-in-a-system-design-interview-27b35de0df26)

----    

## Design a movie recommendation system like Netflix?

- [system-design-interview-questions](http://blog.gainlo.co/index.php/category/system-design-interview-questions/)

----

## How to design Twitter?


### Define Problem:

1. Data modeling. 
   1. Data modeling – If we want to use a **relational database** like `MySQL`, we can define `user object` and `feed object`. Two relations are also necessary. One is user can follow each other, the other is each feed has a user owner.
2. How to serve feeds.
   1. The most straightforward way is to fetch feeds from all the people you follow and render them by time.


### Follow Up question

1. When users followed a lot of people, fetching and rendering all their feeds can be costly. How to improve this?

- There are many approaches. Since Twitter has the **infinite scroll** feature especially on mobile, each time we only need to `fetch the most recent N` feeds instead of all of them. Then there will many details about how the `pagination` should be implemented.
- Use `cache` to store most recent stuff to reduce fetching time


2. How to detect fake users?

- This can be related to machine learning. One way to do it is to identify several related features like `registration date`, the `number of followers`, the `number of feeds` etc. and build a machine learning system to detect if a user is fake.
- Check for pattern like how for a reglar user their number of followers and number of feeds grow over time. For a regular user the growth is monotonic but generally Fake user gains lots of followers and contents in a short span [excluding true celibrity, who if joined, on day 1 will get million followers]

### Can we order feed by other algorithms? Relevency and Recency algorithm


There are a lot of debate about this topic over the past few weeks. If we want to order based on users interests, how to design the algorithm?

>> Facebook, is ranked by relevance

**Relevance Ranking:** Relevancy ranking is the method that is used to order the results list in such a way that `the records most likely to be of interest to a user will be at the top`. This makes searching easier for users as they won't have to spend as much time looking through records for the information that interests them. A good ranking algorithm will put information most relevant to a user's query at the beginning of the returned results.

**How does `relevancy ranking` algorithms work?**

Some factors/features:


- The number of times the search term occurs within a given record.
- The number of times the search term occurs across the collection of records.
- The number of words within a record.
- The frequencies of words within a record.
- The number of records in the index. 

**How does `recency ranking` algorithms work?**

>> According to Instagram, back when feeds were organized in `reverse-chronological order`,i.e using recency, Instagram estimates people missed 50 percent of those important posts, and 70 percent of their feed overall.

**Resource**

- [system-design-interview-question-how-to-design-twitter-part-1](http://blog.gainlo.co/index.php/2016/02/17/system-design-interview-question-how-to-design-twitter-part-1/)
- [Infinite Scrolling](https://eviltrout.com/2013/02/16/infinite-scrolling-that-works.html.html)
- [Relevance Ranking](https://www.lextek.com/manuals/onix/ranking.html)
----

## How to design a Search Engine ?



----


## How the Instagram algorithm works in 2019?

>> Instagram’s primary goal is to maximize the time users spend on the platform. Because the longer users linger, the more ads they see. So directly or indirectly, accounts that help Instagram achieve that goal are rewarded.


### How the algorithm uses `ranking signals` to decide how to arrange each individual user’s feed.

- Relationship
  - Instagram’s algorithm prioritizes content from accounts that users interact with a lot, (commenting each other, DM each other, tag each others post)
- Interest:
  - Algorithm also predicts which posts are important to users based on their past behaviour. Potentially includes the use of machine vision (a.k.a. image recognition) technology to assess the content of a photo.
- Timeliness (Recency)
  - For brands, the timeliness (or “recency”) ranking signal means that paying attention to your audience’s behaviour, and posting when they’re online, is key.

**Resource:**

- [instagram-algorithm](https://blog.hootsuite.com/instagram-algorithm/)


----

## How Does the YouTube Algorithm Work? A Guide to Getting More Views


Features for the algorithm:

- what people watch or don’t watch (a.k.a. impressions vs plays)
- how much time people spend watching your video (watch time, or retention)
- how quickly a video’s popularity snowballs, or doesn’t (view velocity, rate of growth)
- how new a video is (new videos may get extra attention in order to give them a chance to snowball)
- how often a channel uploads new video
- how much time people spend on the platform (session time)
- likes, dislikes, shares (engagement)
- ‘not interested’ feedback (ouch)

**Resource:**

- [How-the-youtube-algorithm-works](https://blog.hootsuite.com/how-the-youtube-algorithm-works/)
- [Paper: Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)



---- 

## Important resources to follow:

- [How to succeed in a system design interview?](https://blog.pramp.com/how-to-succeed-in-a-system-design-interview-27b35de0df26)


----

## Recommendation system for Duolingo [@chiphuyen]

**Question:** Duolingo is a platform for language learning. When a student is learning a new language, Duolingo wants to recommend increasingly difficult stories to read. 

- How would you measure the difficulty level of a story? 
- Given a story, how would you edit it to make it easier or more difficult?


**Answer:**




**Prologue:** This problem can be mapped to predict `Text Readability`.

The RAND Reading Study Group (2002:25), a 14-member panel funded by the United States Department of Education’s Office of Educational Research and Improvement, propose the _following categories and dimensions_ that vary among texts and create varying challenges for readers:


- discourse genre, such as narration, description, exposition, and persuasion;
- discourse structure, including rhetorical composition and coherence;
- media forms, such as textbooks, multimedia, advertisements, and the Internet;
- Sentence difficulty, including vocabulary, syntax, and the propositional text base;
- content, such as age-appropriate selection of subject matter;
- texts with varying degrees of engagement for particular classes of readers.


A text can introduce different level of complexity

### Lexical and syntactic complexity

The best estimate of a text’s difficulty involved the use of eight elements:

- Number of different hard words 
- Number of easy words 
- Percentage of monosyllables
- Number of personal pronouns 
- Average sentence length in words 
- Percentage of different words 
- Number of prepositional phrases
- Percentage of simple sentences

These are all structural elements in the style group, as they “lend themselves most readily to quantitative enumeration and statistical treatment

### Content and subject matter

In terms of content and subject matter, it is commonly believed that abstract texts (e.g., philosophical texts) will be harder to understand than concrete texts describing real objects, events or activities (e.g., stories), and texts on everyday topics are likely to be easier to process than those that are not (Alderson 2000:62).


### How to measure Text Readability?

To measure text difficulty, reading researchers have tended to focus on developing `readability formulas` since the early 1920s. A readability formula is an equation which combines the statistically measurable text features that best predict text difficulty, such as: 

- average sentence length in words or in syllables,
- average word length in characters
- percentage of difficult words (i.e., words with more than two syllables, or words not on a particular wordlist)

Until the 1980s, more than 200 readability formulas had been published (Klare 1984). Among them, the `Flesch Reading Ease Formula`, the `Dale–Chall Formula`, `Gunning Fog Index`, the `SMOG Formula`, the `Flesch–Kincaid Readability test`, and the `Fry Readability Formula` are the most popular and influential (DuBay 2004). These formulas use one to three factors with a view to easy manual application.


Among these factors, vocabulary difficulty (or semantic factors) and sentence length (or syntactic factors) are the strongest indexes of readability (Chall and Dale 1995). The following is the `Flesch Reading Ease` Formula.

$$206.835-1.015\frac{N_{words}}{N_{sents}} - 84.6 \frac{N_{syllables}}{N_{words}}$$


The resulting score ranges from 0 to 100; the lower the score, the more difficult to read the material.


**Final Solution:**

Q1. How would you measure the difficulty level of a story?

- Given a story, process it to get all the features related to Syntactic or Structural complexities as mentioned above.
- Then for all the sotries $S_i$ we have such feature vector $f_i$.
- Apply clustering technique on all the data points over the feature space.
- Now for each data point inside the cluster, measure their `readability score` as per the formula mentioned above and rank the stories inside each cluster by sorting. 
-  Also calculate the `mean readability score` for each cluster and rank the clusters (via sorting) as well.
-  now pick the cluster with minimum mean readability score and randomly pick $k$ stories and recommend the user one after another. After the user finishes the $k$ stories from the cluster $C_i$, pick the next tougher cluster $C_{i+1}$ and pick $K$ stories again. 
-  Also keep a liked/disliked or easy/medium/hard check box for each story to get user feedback after he finished the story. These feedbacks can be passed as feedback loop and can be combined with the existing recommendation system.

Q2. Given a story, how would you edit it to make it easier or more difficult?

- Modify the structural complexities.
- Refactor the content to introduce more simple sentence, Easy synonyms, inject more words with mono syllables etc.
- 

### Resource

- [Link 1](http://www.sanjun.org/html/publication01.html)


----