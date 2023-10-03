* 2023.8.22 - Changed `dense_to_condensed` to `redundant_to_condensed` to agree with updates in `ensemble_networkx`
* 2021.05.18 - Added a `node_label_mapping` argument to `.plot`, an `axes_theta_degrees` argument to .compile, automatically disable labels for quadrant 0/180 when no theta split, and making `pad_axis_label` capable of accepting an iterable of pads.
* 2021.03.08 - This version has migrated many functions and classes to EnsembleNetworkX and now uses this as a dependency.  The rationale was to maintain HiveNetworkX's core objective.  EnsembleNetworkX will now be the more generalizable NetworkX extension.  For a complete list, please review the PR associated with this update.  Also, fixed minor issue with np.all on pd.Index of boolean values generated from .map method (https://github.com/pandas-dev/pandas/issues/40259).  This doesn't happen in pandas v1.0.2 but it does in at least v1.2.2,3 but this is only needed temporarily as this will be resolved in a later version of pandas.  This does not significantly affect performance as this usage was very minimal.
* 2021.01.19 - Added an option to `fill_diagonal` from the `to_dense` option on `Symmetric`
* 2020.09.15 - Import `dict_filter` from soothsayer_utils as it wasn't loaded
* 2020.08.03 - `show_node_labels` can now take an iterable of nodes to show
* Fixed error in `convert_network` where graph objects were being incorrectly labeled reinstantiated as an empty dictionary.

Pending: 
* Need to fix when axis only has one node.
* Add mplcyberpunk for glow on lines (mplcyberpunk.make_lines_glow(axes[0], n_glow_lines=10, alpha_line=0.1618, diff_linewidth=1))