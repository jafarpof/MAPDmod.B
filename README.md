
#Implementation of Mini-Batch Kmeans clustering in PySpark environment
# Introduction
The k-means optimization problem is to find the set C of cluster centers c ∈ R m, with |C| = k, to minimize over a set X of examples x ∈ R m the following objective function:

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

In addition, we implemented a handy method called data parallelization.
 Data parallelism is a popular technique used to speed up training on large mini-batches when each mini-batch is too large to fit on a GPU. Under data parallelism, a mini-batch is split up into smaller sized batches that are small enough to fit on the memory available on different GPUs on the network.

We use "Spark" as a cluster processing engine that allows data to be processed in parallel. Apache Spark's parallelism will enable developers to run tasks parallelly and independently on hundreds of computers in a cluster. All thanks to Apache Spark's fundamental idea, RDD.
