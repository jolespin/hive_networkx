
### Hive NetworkX
High-level [Hive plot](https://doi.org/10.1093/bib/bbr069) (Martin Krzywinski et al. 2012) implementations using [Matplotlib](https://matplotlib.org/) in Python.  Built on top of [NetworkX](https://github.com/networkx/networkx) and [Pandas](https://pandas.pydata.org/).  

#### Dependencies:
Compatible for Python 3.

    pandas >= 1
    numpy
    scipy >= 1
    networkx >= 2
    matplotlib >= 3
    soothsayer_utils >= 2021.03.08

#### Install:
```
# "Stable" release (still developmental)
pip install hive_networkx
# Current release
pip install git+https://github.com/jolespin/hive_networkx
```

#### Source:
* Migrated from [`soothsayer`](https://github.com/jolespin/soothsayer)

#### Usage:

```python
import hive_networkx as hx
```

Hive plots can be produced from the following objects:

 * `enx.Symmetric`
 * `pd.DataFrame`
 * `nx.Graph`, `nx.OrderedGraph` # Note: `DiGraph` functionality has not been tested but should work as undirected 

Simple case of plotting a small subset of the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)

```python
import soothsayer_utils as syu
import ensemble_networkx as enx
import hive_networkx as hx
import numpy as np
import pandas as pd
from collections import OrderedDict

# Load data
X, y, colors = syu.get_iris_data(["X", "y", "colors"])
n, m = X.shape

# Get association matrix (n,n)
method = "pearson"
df_sim = X.T.corr(method=method)
ratio = 0.382
number_of_edges = int((n**2 - n)/2)
number_of_edges_negative = int(ratio*number_of_edges)

# Make half of the edges negative to showcase edge coloring (not statistically meaningful at all)
for a, b in zip(np.random.RandomState(0).randint(low=0, high=149, size=number_of_edges_negative), np.random.RandomState(1).randint(low=0, high=149, size=number_of_edges_negative)):
    if a != b:
        df_sim.values[a,b] = df_sim.values[b,a] = df_sim.values[a,b]*-1

# Create a Symmetric object from the association matrix
sym_iris = enx.Symmetric(data=df_sim, node_type="iris sample", edge_type=method, name="iris", association="network")
# ====================================
# Symmetric(Name:iris, dtype: float64)
# ====================================
#     * Number of nodes (iris sample): 150
#     * Number of edges (correlation): 11175
#     * Association: network
#     * Memory: 174.609 KB
#     --------------------------------
#     | Weights
#     --------------------------------
#     (iris_1, iris_0)        0.995999
#     (iris_0, iris_2)        0.999974
#     (iris_3, iris_0)        0.998168
#     (iris_0, iris_4)        0.999347
#     (iris_0, iris_5)        0.999586
#                               ...   
#     (iris_148, iris_146)    0.988469
#     (iris_149, iris_146)    0.986481
#     (iris_147, iris_148)    0.995708
#     (iris_149, iris_147)    0.994460
#     (iris_149, iris_148)    0.999916

# Create NetworkX graph from the Symmetric object
graph_iris = sym_iris.to_networkx()

# # Create Hive
hive = hx.Hive(graph_iris, axis_type="species")

# Organize nodes by species for each axis
number_of_query_nodes = 3
axis_nodes = OrderedDict()
for species, _y in y.groupby(y):
    axis_nodes[species] = _y.index[:number_of_query_nodes]
    
# Make sure there each node is specific to an axis (not fastest way, easiest to understand)
nodelist = list()
for name_axis, nodes in axis_nodes.items():
    nodelist += nodes.tolist()
assert pd.Index(nodelist).value_counts().max() == 1, "Each node must be on only one axis"

# Add axis for each species
node_styles = dict(zip(['setosa', 'versicolor', 'virginica'], ["o", "p", "D"]))
for name_axis, nodes in axis_nodes.items():
    hive.add_axis(name_axis, nodes, sizes=150, colors=colors[nodes], split_axis=True, node_style=node_styles[name_axis])
hive.compile()
# ===============================
# Hive(Name:iris, dtype: float64)
# ===============================
#     * Number of nodes (iris sample): 150
#     * Number of edges (pearson): 11175
#     * Axes (species): ['setosa', 'versicolor', 'virginica']
#     * Memory: 174.609 KB
#     * Compiled: True
#     ---------------------------
#     | Axes
#     ---------------------------
#     0. setosa (3)              [iris_0, iris_1, iris_2]
#     1. versicolor (3)       [iris_50, iris_51, iris_52]
#     2. virginica (3)     [iris_100, iris_101, iris_102]

# Plot Hive
color_negative, color_positive = ('#278198', '#dc3a23')
edge_colors = hive.weights.map(lambda w: {True:color_negative, False:color_positive}[w < 0])
legend = dict(zip(["Positive", "Negative"], [color_positive, color_negative]))
fig, axes = hive.plot(func_edgeweight=lambda w: (w**10), edge_colors=edge_colors, style="light", show_node_labels=True, title="Iris", legend=legend)
```
![simple](https://i.imgur.com/arNASnul.png)

Hive plot of full dataset without labels to reduce clutter

```python
# # Create Hive
hive_complex = hx.Hive(df_sim, axis_type="species") # Creating Hive plot from a pd.DataFrame

# Organize nodes by species for each axis
axis_nodes = OrderedDict()
for species, _y in y.groupby(y):
    axis_nodes[species] = _y.index
    
# Make sure there each node is specific to an axis (not fastest way, easiest to understand)
nodelist = list()
for name_axis, nodes in axis_nodes.items():
    nodelist += nodes.tolist()
assert pd.Index(nodelist).value_counts().max() == 1, "Each node must be on only one axis"

# Add axis for each species
node_styles = dict(zip(['setosa', 'versicolor', 'virginica'], ["o", "p", "D"]))
for name_axis, nodes in axis_nodes.items():
    hive_complex.add_axis(name_axis, nodes, sizes=150, colors=colors[nodes], split_axis=False, node_style=node_styles[name_axis])
hive_complex.compile()
# ===============================
# Hive(Name:iris, dtype: float64)
# ===============================
#     * Number of nodes (iris sample): 150
#     * Number of edges (pearson): 11175
#     * Axes (species): ['setosa', 'versicolor', 'virginica']
#     * Memory: 174.609 KB
#     * Compiled: True
#     ---------------------------
#     | Axes
#     ---------------------------
#     0. setosa (50)        [iris_0, iris_1, iris_2, iris_3, iris_4, iris_...
#     1. versicolor (50)    [iris_50, iris_51, iris_52, iris_53, iris_54, ...
#     2. virginica (50)     [iris_100, iris_101, iris_102, iris_103, iris_...

# Plotting
color_negative, color_positive = ('#278198', '#dc3a23')
edge_colors = hive_complex.weights.map(lambda w: {True:color_negative, False:color_positive}[w < 0])
legend = dict(zip(["Positive", "Negative"], [color_positive, color_negative]))
fig, axes = hive_complex.plot(func_edgeweight=lambda w: (w**10), edge_colors=edge_colors, style="dark", title="Iris", legend=legend, show_nodes=False)
```
![complex](https://i.imgur.com/3P7l5Bsl.png)