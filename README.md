# MAPDmod.B
Managment analysis of physical dataset project

Implementation of clustering machine learning techniques in PySpark environment
# Introduction
The k-means optimization problem is to find the set C of cluster centers c ∈ R m, with |C| = k, to minimize over a set

X of examples x ∈ R m the following objective function:

min X
x∈X

$$||f(C, x) − x||^2$$

Here, f(C, x) returns the nearest cluster center c ∈ C to x using Euclidean distance.

In our project we harnessed mini-batch optimization for K-means clustering. The reason is that mini-batches have smaller stochastic noise than examples in SGD. The Algorithm for Mini batch K-means is:
Algorithm 1 Mini-batch k-Means.


$$c ← (1 − η)c + ηx $$
<br>
<ul>
    <li> c - the cluster center
    <li> η - learning rate
    <li> x - sample
</ul>
