---
layout: post
title:  "Blog 800: Crack the Coding Topics"
date:   2019-10-23 00:11:31 +0530
categories: jekyll update
mathjax: true
---

## Understand Heap

- A heap is one common implementation of a priority queue.
- A heap is one of the tree structures and represented as a binary tree.

### Representation:

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

### How to ubild heap

- `min_heapify`: make some node and its `descendant nodes meet the heap property`.

```py
def min_heapify(array, i):
    left = 2 * i + 1
    right = 2 * i + 2
    length = len(array) - 1
    smallest = i    if left <= length and array[i] > array[left]:
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
1) `Heap Sort:` Heap Sort uses Binary Heap to sort an array in `O(nLogn)` time.

2) `Priority Queue:` Priority queues can be efficiently implemented using Binary Heap because it supports `insert()`, `delete()` and `extractmax()`, `decreaseKey()` operations in `O(logn)` time. Binomoial Heap and Fibonacci Heap are variations of Binary Heap. These variations perform union also efficiently.

3) Graph Algorithms: The priority queues are especially used in Graph Algorithms like `Dijkstra’s Shortest Path` and `Prim’s Minimum Spanning Tree`.

4) Many problems can be efficiently solved using Heaps. See following for example.
a) K’th Largest Element in an array.
b) Sort an almost sorted array/
c) Merge K Sorted Arrays.


**Reference:**

- [TDS: heap data Structure](https://towardsdatascience.com/data-structure-heap-23d4c78a6962)
- [G4G: Binary Heap](https://www.geeksforgeeks.org/binary-heap/)


----