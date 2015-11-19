lshtc
==========
[eXtreme Learning Spectral Trees](http://arxiv.org/abs/1511.03260) applied to the [lshtc](https://www.kaggle.com/c/lshtc) dataset.  This is a multilabel document classification dataset derived from Wikipedia.

Notes
----------

  * You have to download [lshtcmanik.mat](http://1drv.ms/1YiHLmv) from my public onedrive to get the train-test split (of the Kaggle training set) used by [Manik Varma](http://research.microsoft.com/en-us/um/people/manik/).  This is the default train-test split used, if you invoke `runlshtc()` with no arguments.
    * I also have my own train-test split [lshtcmunge.mat](http://1drv.ms/1MprMOn) that I used in a couple of papers.  If you invoke `runlshtc(false)` then the script will use my train-test split.
  * The tree code is similar to [odp](../odp/runodp.m), modified for multilabel.  The `xhat()` routine contains the most significant difference: Y<sup>&#x22A4;</sup>Y is not diagonal for multilabel, but it is sparse, so we use conjugate gradient to invert.
  * The underlying classifier only uses the node identifier to adjust the bias of the (independent per-class) logistic regression, similar to the [odp](../odp/runodp.m) script.
  * The underlying classifier is trained using hogwild SGD.  Matlab will be completely unresponsive during a training pass as the interpreter is stuck waiting for the threads.
  * I store the model in an mmap in case there is not enough RAM.  On a machine with at least 96Gb of RAM, an SSD, and an appropriately tuned page cache this should not generate any appreciable I/O load.  Otherwise, the script will run, but more slowly.
  * On my desktop it takes about 10 hours to build a depth 14 tree, which has a test recall of 50.4%.  Then it takes about 5 hours to do a training pass.  25 training passes leads to 19.5% test accuracy.

Thanks
----------
lshtcmanik.mat is derived from data provided by Manik Varma from [Locally Non-linear Embeddings for Extreme Multi-label Learning](http://arxiv.org/abs/1507.02743) and [FastXML: A Fast, Accurate and Stable Tree-classifier for eXtreme Multi-label Learning](http://research.microsoft.com/apps/pubs/default.aspx?id=245233).
