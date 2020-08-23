
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

# Design Youtube Recommendation system?

**Feature engineering:**

Usually, there are two types of features – `explicit` and `implicit features`. 


- Demographic:
  - Age [From Login information]
  - Sex [From Login Information]
  - Country 

- Explicit features can be `ratings`, `favorites` etc.. In Youtube, it can be the 
  - `like`/`share`/`subscribe actions`.
  - If `do comment` for any video
  - Video title, label, category
  - Time of the day [morning ritual/religious/gym video, evening music video, dance video, party] 
  - Add videos to `Watch Later` or to explicit `User Playlist`

- Implicit features are less obvious. 
  - `Watch Time`: If a user has watched a video for only a couple of seconds, probably it’s a negative sign. 
  - `Personal Preference`: Given a list of recommended videos, if a user clicks one over another, it can mean that he prefer to the one clicked. Usually, we need to explore a lot about implicit features.
  - Freshness [just launched]


**Recommend from heavy tail**


**Reference:**

- [Design a Recommendtion System](http://blog.gainlo.co/index.php/2016/05/24/design-a-recommendation-system/)
- [Leetcode Discussion](https://leetcode.com/discuss/interview-question/124565/Design-Netflix-recommendation-engine)
- [IJCAI 2013 Tutorial PPT](http://ijcai13.org/files/tutorial_slides/td3.pdf)


----

# Design a movie recommendation system like Netflix?

- [system-design-interview-questions](http://blog.gainlo.co/index.php/category/system-design-interview-questions/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


-----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>