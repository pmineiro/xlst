odp
==========
[eXtreme Learning Spectral Trees](http://arxiv.org/abs/1511.03260) applied to the [odp](https://www.dmoz.org/) dataset.  This is a multiclass document classification dataset derived from the open directory project.

Notes
----------

  * You have to download [odpmunge.mat](http://1drv.ms/1MTF7A1) from my public onedrive.
  * The tree code is (intended to be) semantically identical to the [aloi](../aloi) demo, but with several routines replaced by faster mex equivalents.  In particular this is the multclass version of the eigenvalue problem in `xhat()`.  For a multilabel version of the `xhat()` routine see [lshtc](../lshtc/runlshtc.m).
  * The underlying classifier only uses the node identifier to adjust the bias of the logistic regression.  This is a less aggressive use of the node identifier than in the [aloi](../aloi) solution.
  * The underlying classifier is trained using hogwild SGD.  Matlab will be completely unresponsive during a training pass as the interpreter is stuck waiting for the threads.
  * I store the model in an mmap in case there is not enough RAM.  On a machine with at least 64Gb of RAM, an SSD, and an appropriately tuned page cache this should not generate any appreciable I/O load.  Otherwise the script will run, but more slowly.
  * On my desktop it takes about 10 hours to build a depth 14 tree, which has a test recall of 50.4%.  Then it takes about 5 hours to do a training pass.  25 training passes leads to 19.5% test accuracy.

Thanks
----------
odpmunge.mat is derived from data provided by John Langford from [Logarithmic Time Online Multiclass prediction](http://arxiv.org/abs/1406.1822) which in turn is derived from data provided by Paul Bennet from [Refined experts: improving classification in large taxonomies](http://research.microsoft.com/en-us/um/people/pauben/papers/sigir-2009-refined-experts-bennett-nguyen.pdf).
