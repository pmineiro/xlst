aloi
==========
[eXtreme Learning Spectral Trees](http://arxiv.org/abs/1511.03260) applied to [aloi](http://aloi.science.uva.nl/).

Aloi is small enough that results can be obtained in a few minutes using a 
pure matlab implementation (no mex required).

Notes
----------

  * The routine xhat() corresponds to the eigenvalue problem associated with learning a tree node.  For multiclass problems Y<sup>&#x22A4;</sup>Y is diagonal which simplifies things a bit.
  * The tree is a label filter which can be coupled with an arbitrary classifier.  In this case we share a 50-dimensional embedding of the original features across all nodes, and then learn a per-node classifier (over the node candidates).  This is a compromise between distinct classifiers per node and the same classifier at each node.

Thanks
----------
aloi.mat is derived from data provided by John Langford.