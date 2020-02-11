---
layout: post
title:  "Crack the Code"
date:   2019-10-23 00:11:31 +0530
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

# Practice followup questions for each coding questions

1. How do you test your solution? Testing is also an important area in interview and you can treat the interview question as a real life project and how can you guarantee the system works as intended?
2. What if the input size is too large to put in memory?
3. How do you refine your solution and analyze the efficiency?
4. What if the input size is too large to put in disk? How do you solve the scalability issue?
5. What if you have infinite amount of memory? Any way to make the solution faster?
6. You can also try to change some restriction of the question, like what if it’s a normal tree instead of binary tree? What if you can store the dictionary in any data structure you like?

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Understand Heap

- A heap is one common implementation of a priority queue.
- A heap is one of the tree structures and represented as a binary tree.

## Representation:

![image](https://miro.medium.com/max/835/1*ds0JXOw3lLqNo6hw__NtZw.png)

- The heap above is called a min heap, and each value of nodes is less than or equal to the value of child nodes. We call this condition the heap property.

- In a min heap, when you look at the parent node and its child nodes, the parent node always has the smallest value. When a heap has an opposite definition, we call it a max heap. For the following discussions, we call a min heap a heap.

```py
    A root node｜i = 1, the first item of the array
    A parent node｜parent(i) = i / 2
    A left child node｜left(i) = 2i
    A right child node｜right(i)=2i+1
```

- When you look at the node of index 4, the relation of nodes in the tree corresponds to the indices of the array below.

![image](https://miro.medium.com/max/818/1*ysSV1xV0OMm-1amWBpFb0A.png)

## How to build heap

- `min_heapify`: make some node and its `descendant nodes meet the heap property`.

```py
def min_heapify(array, i):
    left = 2 * i + 1
    right = 2 * i + 2
    length = len(array) - 1
    smallest = i    
    if left <= length and array[i] > array[left]:
        smallest = left
    if right <= length and array[smallest] > array[right]:
        smallest = right
    if smallest != i:
        array[i], array[smallest] = array[smallest], array[i]
        min_heapify(array, smallest)
```

- First, this method computes the node of the smallest value among the node of index i and its child nodes and then exchange the node of the smallest value with the node of index i.
- When the exchange happens, this method applies min_heapify to the node exchanged.


Index of a list (an array) in Python starts from 0, the way to access the nodes will change as follow.

```py
    The root node｜i = 0
    The parent node｜parent(i) = (i-1) / 2
    The left child node｜left(i) = 2i + 1
    The right child node｜right(i)=2i+2
```

- `build_min_heap`: produce a heap from an arbitrary array.

```py
def build_min_heap(array):
    #for i=n/2 downto 1
    for i in reversed(range(len(array)//2)):
        min_heapify(array, i)
```

This function iterates the nodes except the leaf nodes with the for-loop and applies min_heapify to each node. We don’t need to apply min_heapify to the items of indices after n/2+1, which are all the leaf nodes. We apply min_heapify in the orange nodes below.

![image](https://miro.medium.com/max/940/1*Qa4zV-Ys8iXRbPCt2Xt3Zw.png)



## Application of Heap (Priority Queue)

Applications of Heaps:
1. `Heap Sort:` Heap Sort uses Binary Heap to sort an array in `O(nLogn)` time.

2. `Priority Queue:` Priority queues can be efficiently implemented using Binary Heap because it supports `insert()`, `delete()` and `extractmax()`, `decreaseKey()` operations in `O(logn)` time. Binomoial Heap and Fibonacci Heap are variations of Binary Heap. These variations perform union also efficiently.

3. Graph Algorithms: The priority queues are especially used in Graph Algorithms like `Dijkstra’s Shortest Path` and `Prim’s Minimum Spanning Tree`.

4. Many problems can be efficiently solved using Heaps. See following for example.
   1. K’th Largest Element in an array.
   2. Sort an almost sorted array/
   3. Merge K Sorted Arrays.


**Reference:**

- [TDS: heap data Structure](https://towardsdatascience.com/data-structure-heap-23d4c78a6962)
- [G4G: Binary Heap](https://www.geeksforgeeks.org/binary-heap/)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Difference between `bounded` and `unbounded` 0/1 knapsack

## In Recursive Approach

The only difference between the 0/1 Knapsack `bounded` and `unbounded` problem is that, after including the item

- `unbounded`: We recursively call to process `all the items` (including the current item)
 - `currentIndex` is not increased while calling the recursive function
  
```py
profit1 = 0
if weights[currentIndex] <= capacity:
    profit1 = profits[currentIndex] + solve_knapsack_recursive(
      profits, weights, capacity - weights[currentIndex], currentIndex)
```

 

- `bounded`: We recursively call to process all the `remaining items` (excluding the current item)
  - `currentIndex+1` is introduced while calling the recursive function
  
```py
profit1 = 0
if weights[currentIndex] <= capacity:
    profit1 = profits[currentIndex] + solve_knapsack_recursive(
      profits, weights, capacity - weights[currentIndex], currentIndex+1)
```

## In Bottom-Up Approach

**0/1 Bounded:** for each item at index `i` ($0 \leq i \lt items.length$) and capacity `c` ($0 \leq c \leq capacity$), we have two options:

1. Exclude the item at index `i`. In this case, we will take whatever profit we get from the sub-array excluding this item => `profit_exclude = dp[i-1][c]`
2. Include the item at index `i` if its weight is not more than the capacity. In this case, we include its profit 
`profit_include = profit[i] + dp[i-1][c-weight[i]]`
   - plus the state `c-weight[i]` for rest of the item (excluding) 
3. `dp[index][c] = max(profit_include, profit_exclude)`


**Un-Bounded:** for each item at index `i` ($0 \leq i \lt items.length$) and capacity `c` ($0 \leq c \leq capacity$), we have two options:

1. Exclude the item at index `i`. In this case, we will take whatever profit we get from the sub-array excluding this item => `profit_exclude = dp[i-1][c]`
2. Include the item at index `i` if its weight is not more than the capacity. In this case, we include its profit 
`profit_include = profit[i] + dp[i][c-weight[i]]`
   - plus the state `c-weight[i]` for the item (including)
3. `dp[index][c] = max(profit_include, profit_exclude)`

for both the approach, the subtle difference is at step 2, after adding the profit.

- **Bounded:** `dp[i-1][c-weight[i]]`
- **Unbounded:** `dp[i][c-weight[i]]`

Rest same. 

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>


----

# How to Visualize DP Bototm Up for Longest Palindromic Substring or Subsequence 

The below example is to find `Longest Palindromic Substring` for string `cddpd`

![image](/assets/images/image_26_DP_sol_1.jpg)
![image](/assets/images/image_26_DP_sol_2.jpg)



- `si = 2` and `ei = 4`: means we are checking substring `dpd`. 
- Now `st[si] == st[ei]` as both are `d`
- So check if `(si+1:ei-1)` is sub string or not ?
- `si+1 = 3`, `ei-1 = 3` i.e from `st[3:3]`, which is the char `p` and from the `dp[][]` we see `dp[3][3] = T` , so overall `dp[2][4] = T`


This is the overall thought processing, how the 2 things are related in High Dimension i.e DP 2D matrix with Low Dimension i.e 1D string.

----

# Difference between Longest Palindromic `Substring` and Longest Palindromic `Subsequence`

A `subsequence` is a sequence that can be derived from another sequence by `deleting some` or `no elements` without changing the order of the remaining elements.

A `substring` is a continuous sequence.

## Longest Palindromic `Subsequence`

**Example 1:**

```py
Input: "abdbca"
Output: 5
Explanation: LPS is "abdba".
```

**Example 2:**

```py
Input: = "cddpd"
Output: 3
Explanation: LPS is "ddd".
```

The `abdba` from example 1, is not continuous (not a substring) as the problem was to find Longest Palindromic `Subsequence`. So while implementing, then it's `NOT NECESSARY`, if `str[start] == str[end]` then remaining elements in between `str[strat+1:end-1]` is also a palindrome.  


```py
if st[endIndex] == st[startIndex]: 
  dp[startIndex][endIndex] = 2 + dp[startIndex + 1][endIndex - 1]
else: 
  dp[startIndex][endIndex] = Math.max(dp[startIndex + 1][endIndex], dp[startIndex][endIndex - 1])
```

## Longest Palindromic `Substring`

**Example 1:**

```py
Input: "abdbca"
Output: 3
Explanation: LPS is "bdb".
```

**Example 2:**

```py
Input: = "cddpd"
Output: 3
Explanation: LPS is "dpd".
```

```py
if st[startIndex] == st[endIndex]:
    remaining_length = endIndex - startIndex - 2
    if remaining_length <=1 or dp[startIndex+1][endIndex-1] == True:
        dp[startIndex][endIndex] = True
```

- Here the problem is to find Longest Palindromic `Substring`, then it's `NECESSARY` if `str[start] == str[end]` then remaining elements in between `str[strat+1:end-1]` is a palindrome.

- `dp[startIndex+1][endIndex-1] == True` denotes if remaining in between elements are also palindrome.

----

# Comprehensive Data Structure and Algorithm Study Guide

- [Leetcode: Gold Mine](https://leetcode.com/discuss/general-discussion/494279/comprehensive-data-structure-and-algorithm-study-guide)

----

# Exercise:

1. [DP: bellman-equations-reinforcement-learning](https://int8.io/bellman-equations-reinforcement-learning/)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>