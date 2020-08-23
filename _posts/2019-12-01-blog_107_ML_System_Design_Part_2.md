---
layout: post
title:  "Machine Learning System Design (Part - 2)"
date:   2019-12-01 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

----

# How to approach System Design Interview Question

## Be familiar with basic knowledge

First of all, there’s no doubt you should be very good at `data structure` and `algorithm`. Take the URL shortening service as an example, you won’t be able to come up with a good solution if you are not clear about `hash`, `time/space` complexity analysis.

Quite often, there’s a trade-off between `time` and `memory` efficiency and you must be very proficient in the big-O analysis in order to figure everything out,

There are also several other things you’d better be familiar although it’s possible that they may not be covered in your interview.

- **Abstraction:** It’s a very important topic for system design interview. You should be clear about how to abstract a system, what is visible and invisible from other components, and what is the logic behind it. Object oriented programming is also important to know.
- **Database:** You should be clear about those basic concepts like relational database. Knowing about No-SQL might be a plus depends on your level (new grads or experienced engineers).
- **Network:** You should be able to explain clearly what happened when you type “gainlo.co” in your browser, things like DNS lookup, HTTP request should be clear.
- **Concurrency:** It will be great if you can recognize concurrency issue in a system and tell the interviewer how to solve it. Sometimes this topic can be very hard, but knowing about basic concepts like race condition, dead lock is the bottom line.
- **Operating system:** Sometimes your discussion with the interviewer can go very deeply and at this point it’s better to know how OS works in the low level.
- **Machine learning:** (optional). You don’t need to be an expert, but again some basic concepts like feature selection, how ML algorithm works in general are better to be familiar with.

## :+1:  Top-down + modularization

This is the general strategy for solving a system design problem and ways to explain to the interviewer. 

:warning: The worst case is always jumping into details immediately, which can only make things in a mess.

It’s always good to start with **high-level ideas** and then figure out details step by step, so this should be a `top-down approach`. Why? Because many system design questions are very general and there’s no way to solve it without a big picture. 

> You should always have a big picture.

Let’s use Youtube recommendation system as an example. I might first divide this into front-end and backend (the interviewer may only ask for backend or a specific part, but I’ll cover the whole system to give you an idea). For backend, the flow can be 3 steps: collect user data (like videos he watched, location, preferences etc.), offline pipeline that generating the recommendation, and store and serve the data to front-end. And then, we can jump into each detailed components.


**Reference:**

- [8 Things You Need to Know Before a System Design Interview](http://blog.gainlo.co/index.php/2015/10/22/8-things-you-need-to-know-before-system-design-interviews/) :fire:


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

-----

# Design Youtube Recommendation system?

## Big Picture:

- Front End
- Back End
  - Collect user data (like videos he watched, location, preferences etc.)
  - Offline pipeline that generates the recommendation [Hybrid Approach: Heuristic + ML Based]
  - Store and Serve the data to front-end.


Basically, we can simplify the system into a couple of major components as follows:

- **Storage:** How do you design the database schema? What database to use? Videos and images can be a subtopic as they are quite special to store.
- **Scalability:** When you get millions or even billions of users, how do you scale the storage and the whole system? This can be an extremely complicated problem, but we can at least discuss some high-level ideas.
- **Web server:** The most common structure is that front ends (both mobile and web) talk to the web server, which handles logics like user authentication, sessions, fetching and updating users’ data, etc.. And then the server connects to multiple backends like video storage, recommendation server and so forth.
- **Cache:** is another important components. We’ve discussed in details about cache before, but there are still some differences here, e.g. we need cache in multiple layers like web server, video serving, etc..
- There are a couple of other important components like **recommendation system**, security system and so on. As you can see, just a single feature can be used as a stand-alone interview question.

## Storage and data model

If you are using a **relational database** like `MySQL`, designing the data schema can be straightforward. And in reality, **Youtube does use MySQL** as its main database from the beginning and it works pretty well.

First and foremost, we need to define the 
- `User model`: which can be stored in a single table including email, name, registration data, profile information and so on. 
  - Another common approach is to keep user data in two tables – 
    - `Authentication Table`: For authentication related information like email, password, name, registration date, etc.
    - `Profile Info Table`: Additional profile information like address, age and so forth.

- `Video Model`: A video contains a lot of information including meta data (title, description, size, etc.), video file, comments, view counts, like counts and so on. Apparently, basic video information should be kept in separate tables so that we can first have a video table.

- `Author-Video Table`: Another table to map `user id` to `video id`. And `user-like-video` relation can also be a separate table. The idea here is **database normalization** – organizing the columns and tables to reduce data redundancy and improve data integrity.

## Video and image storage

It’s recommended to store large static files like videos and images separately as it has better performance and is much easier to organize and scale. It’s quite counterintuitive that Youtube has more images than videos to serve. Imagine that each video has thumbnails of different sizes for different screens and the result is having 4X more images than videos. Therefore we should never ignore the image storage.

One of the most common approaches is to use **CDN (Content delivery network)**. In short, CDN is a globally distributed network of proxy servers deployed in multiple data centers. The goal of a CDN is to serve content to end-users with high availability and high performance. It’s a kind of 3rd party network and many companies are storing static files on CDN today.

The biggest benefit using CDN is that CDN replicates content in multiple places so that there’s a better chance of content being closer to the user, with fewer hops, and content will run over a more friendly network. In addition, CND takes care of issues like scalability and you just need to pay for the service.


## Popular VS long-tailed videos

If you thought that CDN is the ultimate solution, then you are completely wrong. Given the number of videos Youtube has today ($819,417,600$ hours of video), it’ll be extremely costly to host all of them on CDN especially majority of the videos are **long-tailed**, which are videos have only $1$-$20$ views a day.

However, one of the most interesting things about Internet is that usually, it’s those long-tailed content that attracts the majority of users. The reason is simple – those popular content can be found everywhere and only long-tailed things make the product special.

Coming back to the storage problem. One straightforward approach is to 
- Host popular videos in CDN
- Less popular videos are stored in our own servers by location. 

This has a couple of advantages:

- Popular videos are viewed by a huge number of audiences in different locations, which is what CND is good at. It replicates the content in multiple places so that it’s more likely to serve the video from a close and friendly network.
- Long-tailed videos are usually consumed by a particular group of people and if you can predict in advance, it’s possible to store those content efficiently.


## Recommendation Architecture:

- Apparently, the system contains multiple steps/components. Which can be divided into `online` and `offline` part.
  - e.g. comparing similar users/videos can be time-consuming on Youtube, this part should be done in offline pipelines. For the offline part, all the user models and videos need to store in `distributed systems`. 
  - In fact, for most machine learning systems, it’s common to use offline pipeline to process big data as you won’t expect it to finish with few seconds.
- Feedback loop
- Periodic model training to capture new behavior


### ML Algorithm:

- `Colleborative Filtering`: In a nutshell, to recommend videos for a user, I can provide videos liked by similar users. For instance, if user A and B have watched a bunch of same videos, it’s highly likely that user A will like videos liked by B. Of course, there are many ways to define what “similar” means here. It could be two users have liked same videos, it could also mean that they share the same location.
  - The above algorithm is called `user-based` collaborative filtering. Another version is called `item-based` collaborative filtering, which means to recommend videos (items) that are similar to videos a user has watched.

- `Locally Sensitive Hashing`

### Cold Start:

For a new user, based on his/her age, sex, location (available from login information) recommends from a `general pool`. The genearl pool may contain:

- Most seen/liked videos in the country X
- Most seen/liked videos for the given age, sex

Slowly the user starts to engage with the platform, click some videos, like/dislike some, comment on sime, search some videos. All these activities will help to improve the recommendation system.

The final solution can be a hybrid solution, that is a mixture of Rule Based (Heuristic) and AI based approach.

### Heuristic Solution

- Rule based approach
  - Based on videos a user has watched, we can simply suggest videos from same authors
  - Suggest videos with similar titles or labels.
  - If use Popularity (number of comments, shares) as another signal, the recommendation system can work pretty well as a baseline 
  - Suggest videos whose `title` is similar to the `search queries`


### Feature engineering:

Usually, there are two types of features – `explicit` and `implicit features`. 


- Demographic:
  - Age [From Login information]
  - Sex [From Login Information]
  - Country 

- **Explicit features** can be `ratings`, `favorites` etc.. In Youtube, it can be the 
  - `like`/`share`/`subscribe actions`.
  - If `do comment` for any video
  - Video title, label, category
  - Time of the day [morning ritual/religious/gym video, evening music video, dance video, party] 
  - Add videos to `Watch Later` or to explicit `User Playlist`

- **Implicit features** are less obvious. 
  - `Watch Time`: If a user has watched a video for only a couple of seconds, probably it’s a negative sign. 
  - `Personal Preference`: Given a list of recommended videos, if a user clicks one over another, it can mean that he prefer to the one clicked. Usually, we need to explore a lot about implicit features.
  - Freshness [just launched]


**Recommend from heavy tail**

- Under this category the recommendation system will show some diversified content. 

## Potential Scale Issues:

- **Response time:** Offline pipelines to precompute some signals that can speed up the ranking
  - Model inference time
- **Scale Architecture:** With millions of users, a single server is far from enough due to storage, memory, CPU bound issues etc.. That’s why it’s pretty common to see server crashes when there are a large number of requests. To scale architecture, the rule of thumb is that **service-oriented architecture beats monolithic application**. 
  - Instead of having everything together, it’s better to **divide the whole system into small components by service** and `separate each component`. To communicate between different components use `load balancer`
  - Cloud Based solution: AWS/Google Cloud/ Azure
  - Horizontal Scaling
  - Kubernetes based solution
- **Scale database:** Even if we put the database in a separate server, it will not be able to store an infinite number of data. At a certain point, we need to scale the database. For this specific problem, we can either do the `vertical splitting` (partitioning) by splitting the database into sub-databases like `user database`, `comment database` etc. or `horizontal splitting` (**sharding**) by splitting based on attributes like US users, European users.

You can check [this](http://highscalability.com/blog/2014/5/12/4-architecture-issues-when-scaling-web-applications-bottlene.html) post for deeper analysis of scalability issues.



**Reference:**

- [Design a Recommendtion System](http://blog.gainlo.co/index.php/2016/05/24/design-a-recommendation-system/)
- [Leetcode Discussion](https://leetcode.com/discuss/interview-question/124565/Design-Netflix-recommendation-engine)
- [IJCAI 2013 Tutorial PPT](http://ijcai13.org/files/tutorial_slides/td3.pdf)
- [4 Architecture Issues When Scaling Web Applications: Bottlenecks, Database, CPU, IO](http://highscalability.com/blog/2014/5/12/4-architecture-issues-when-scaling-web-applications-bottlene.html) :fire:

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Design a movie recommendation system like Netflix?

- [system-design-interview-questions](http://blog.gainlo.co/index.php/category/system-design-interview-questions/)


-----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>