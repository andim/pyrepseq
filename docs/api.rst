.. api reference

API Reference
=============

.. toctree::
   :maxdepth: 2

Main module
-----------

.. automodule:: pyrepseq
   :members:

IO
~~

.. automodule:: pyrepseq.io
   :members:

Stats
~~~~~

.. automodule:: pyrepseq.stats
   :members:

Distance
~~~~~~~~

.. automodule:: pyrepseq.distance
   :members:


Nearest Neighbor
~~~~~~~~~~~~~~~~

.. automodule:: pyrepseq.nn
   :members:



Plotting submodule
------------------

.. automodule:: pyrepseq.plotting
   :members: rankfrequency, density_scatter, similarity_clustermap, seqlogos, seqlogos_vj, align_seqs, label_axes, labels_to_colors_hls, labels_to_colors_tableau

Metrics
-------

General Metrics
~~~~~~~~~~~~~~~

.. autoclass:: pyrepseq.metric.Metric
   :members:

.. autoclass:: pyrepseq.metric.Levenshtein
.. autoclass:: pyrepseq.metric.WeightedLevenshtein

TCR Metrics
~~~~~~~~~~~

.. autoclass:: pyrepseq.metric.tcr_metric.TcrMetric
   :members:

.. autoclass:: pyrepseq.metric.tcr_metric.AlphaCdr3Levenshtein
.. autoclass:: pyrepseq.metric.tcr_metric.BetaCdr3Levenshtein
.. autoclass:: pyrepseq.metric.tcr_metric.Cdr3Levenshtein
.. autoclass:: pyrepseq.metric.tcr_metric.AlphaCdrLevenshtein
.. autoclass:: pyrepseq.metric.tcr_metric.BetaCdrLevenshtein
.. autoclass:: pyrepseq.metric.tcr_metric.CdrLevenshtein

.. autoclass:: pyrepseq.metric.tcr_metric.AlphaCdr3Tcrdist
.. autoclass:: pyrepseq.metric.tcr_metric.BetaCdr3Tcrdist
.. autoclass:: pyrepseq.metric.tcr_metric.Cdr3Tcrdist
.. autoclass:: pyrepseq.metric.tcr_metric.AlphaTcrdist
.. autoclass:: pyrepseq.metric.tcr_metric.BetaTcrdist
.. autoclass:: pyrepseq.metric.tcr_metric.Tcrdist
