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
- 