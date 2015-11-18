matlab
==========
Some mex to speed things up.

Build Instructions
------------------

Type `make` at the command line.  You can adjust the number of threads used by the routines via `make NUM_THREADS=n`.  On Linux I attempt to guess the right number of threads, YMMV.

Things specific to the tree:
 * [treemakeimpweights.cpp](treemakeimpweights.cpp): mex variant of the `makeimpweights` routine from [runaloi](../aloi/runaloi.m), which does three things.  The first one is critical and the other two are just useful things to know.
   * counts how many of each example's labels are preserved by the tree when filtering.  This is the really important one: we don't want to update the underlying classifier on labels that are filtered out.
   * computes the average number of candidate labels over the examples.
   * computes the average depth in the tree over the examples.
 * [treepredict.cpp](treepredict.cpp): mex variant of the `predict` routine from [runaloi](../aloi/runaloi.m), which computes the most likely class for each example.
 * [treeroute.cpp](treeroute.cpp): mex variant of the `route` routine from [runaloi](../aloi/runaloi.m), which computes the leaf node for each example, optionally using randomized routing.
 * [treeupdate.cpp](treeupdate.cpp): hogwild multicore SGD training of the underlying classifier.

Stuff to improve Matlab's performance with sparse matrices:

 * [dmsm.cpp](dmsm.cpp): multicore dense matrix times sparse matrix. 
 * [sparsequad.cpp](sparsequad.cpp): multicore sparse quadratic form.
 * [sparseweightedsum.cpp](sparseweightedsum.cpp): multicore sparse weighted column sum.
