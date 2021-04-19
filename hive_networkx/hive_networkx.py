# -*- coding: utf-8 -*-
from __future__ import print_function, division

# Built-ins
from collections import OrderedDict, defaultdict
import sys, datetime, copy, warnings

# External
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import entropy, mannwhitneyu
from scipy.spatial.distance import squareform, pdist
from itertools import combinations

# soothsayer_utils
from soothsayer_utils import assert_acceptable_arguments, is_symmetrical, is_graph, is_nonstring_iterable, dict_build, dict_filter, is_dict, is_dict_like, is_color, is_number, write_object, format_memory, format_header, check_packages
try:
    from . import __version__
except ImportError:
    __version__ = "ImportError: attempted relative import with no known parent package"

# ensemble_networkx
from ensemble_networkx import Symmetric, condensed_to_dense

# ==========
# Conversion
# ==========
# Polar to cartesian coordinates
def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return(x, y)

# Cartesian to polar coordinates
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(r, theta)

# =============
# Normalization
# =============
# Normalize MinMax
def normalize_minmax(x, feature_range=(0,1)):
    """
    Adapted from the following source:
    * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    x_std = (x - x.min())/(x.max() - x.min())
    return x_std * (feature_range[1] - feature_range[0]) + feature_range[0]

# =======================================================
# Hive
# =======================================================
class Hive(object):
    def __init__(self, data,  name=None,  node_type=None, edge_type=None, axis_type=None, description=None, tol=1e-10):
        """
        Hive plots for undirected networks
        
        Hive plots:
        Should only be used with 2-3 axis unless intelligently ordered b/c the arcs will overlap.
        
        Notes:
        * Does not store networkx graph to overuse memory just use .to_networkx as generate them in real time. 
        
        Usage:
        import soothsayer_utils as syu
        import ensemble_networkx ax enx
        import hive_networkx as hx

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
        """ 
                
        # Placeholders
        self.nodes_in_hive = None 
        self.edges_in_hive = None
        self.weights = None
#         self.graph = None
        self.name = name
        self.node_type = node_type
        self.edge_type = edge_type
        
        # Propogate
        if isinstance(data, pd.DataFrame):
            data = self._from_pandas_adjacency(data, name, node_type, edge_type, tol) # -> Symmetric
        if isinstance(data, Symmetric):
            self._from_symmetric(data, name, node_type, edge_type) 
        if all([
            (self.nodes_in_hive is None),
            (self.edges_in_hive is None),
            (self.weights is None),
            ]):
            assert is_graph(data), "`data` must be either a pd.DataFrame adjacency, a Symmetric, or a networkx graph object" # Last resort, use this if Symmetric isn't provided
            self._from_networkx(data)

        # Initialize
        self.axes = OrderedDict()
        self.node_mapping_ = OrderedDict()
        self.compiled = False
        self.axis_type = axis_type
        self.description = description
        self.version = __version__
        self.number_of_nodes_ = None
        self.memory = self.weights.memory_usage()
        self.__synthesized__ = datetime.datetime.utcnow()

    def _from_pandas_adjacency(self, data, name, node_type, edge_type, tol):
        # Convert pd.DataFrame into a Symmetric object
        assert isinstance(data, pd.DataFrame), "Must be a 2-dimensional pandas DataFrame object"
        assert is_symmetrical(data, tol=tol), "DataFrame must be symmetrical.  Please force symmetry with (X + X.T)/2"
        return Symmetric(data=data, name=name, node_type=node_type, edge_type=edge_type, association="network", nans_ok=False, tol=tol)

    def _from_symmetric(self, data, name, node_type, edge_type):
        # Propogate information from Symmetric
        if name is None:
            self.name = data.name
        if node_type is None:
            self.node_type = data.node_type
        if edge_type is None:
            self.edge_type = data.edge_type
        self.nodes_in_hive = data.nodes
        self.edges_in_hive = data.edges
        self.weights = data.weights
#             return data.to_networkx()

    def _from_networkx(self, graph):
        # Propogate information from graph
        for attr in ["name", "node_type", "edge_type"]:
            if getattr(self, attr) is None:
                if attr in graph.graph:
                    value =graph.graph[attr]
                    if bool(value):
                        setattr(self, attr, value)
                
#             if self.graph is None:
#                 self.graph = graph
        if self.nodes_in_hive is None:
            self.nodes_in_hive = pd.Index(sorted(graph.nodes()))
        if (self.edges_in_hive is None) or (self.weights is None):
            self.weights = dict()
            for edge_data in graph.edges(data=True):
                edge = frozenset(edge_data[:-1])
                weight = edge_data[-1]["weight"]
                self.weights[edge] = weight
            self.weights = pd.Series(self.weights, name="Weights")#.sort_index()
            self.edges_in_hive = pd.Index(self.weights.index, name="Edges")

    # Built-ins
    
    def __repr__(self):
        pad = 4
        header = format_header("Hive(Name:{}, dtype: {})".format(self.name, self.weights.dtype),line_character="=")
        n = len(header.split("\n")[0])
        fields = [
            header,
            pad*" " + "* Number of nodes ({}): {}".format(self.node_type, len(self.nodes_in_hive)),
            pad*" " + "* Number of edges ({}): {}".format(self.edge_type, len(self.edges_in_hive)),
            pad*" " + "* Axes ({}): {}".format(self.axis_type, list(self.axes.keys())),
            pad*" " + "* Memory: {}".format(format_memory(self.memory)),
            pad*" " + "* Compiled: {}".format(self.compiled),
            ]
        if self.compiled:
            for field in map(lambda line:pad*" " + line, format_header("| Axes", "-", n=n-pad).split("\n")):
                fields.append(field)
            for field in map(lambda line: pad*" " + str(line), repr(self.axes_preview_).split("\n")[:-1]):
                fields.append(field)
        return "\n".join(fields)

    def __call__(self, name_axis=None):
        return self.get_axis_data(name_axis=name_axis)
    # def __getitem__(self, key):
    #     return self.weights[key]

    # Add axis to HivePlot
    def add_axis(self, name_axis, nodes, sizes=None, colors=None, split_axis:bool=False, node_style="o", scatter_kws=dict()):
        """
        Add or update axis
        
        nodes: Can be either an iterable of nodes or a dict-like with node positions {node:position}
        """
        # Initialize axis container
        self.axes[name_axis] = defaultdict(dict)
        self.axes[name_axis]["colors"] = None
        self.axes[name_axis]["sizes"] = None
        self.axes[name_axis]["split_axis"] = split_axis
        self.axes[name_axis]["node_style"] = node_style
        self.axes[name_axis]["scatter_kws"] = scatter_kws

        # Assign (preliminary) node positions
        if is_nonstring_iterable(nodes) and not isinstance(nodes, pd.Series):
            nodes = pd.Series(np.arange(len(nodes)), index=nodes)
        if is_dict(nodes):
            nodes = pd.Series(nodes)
        nodes = nodes.sort_values()
        assert set(nodes.index) <= set(self.nodes_in_hive), "All nodes in axis should be in the Hive and they aren't..."
        
        # Set values
        self.axes[name_axis]["node_positions"] = pd.Series(nodes, name=(name_axis, "node_positions"))
        self.axes[name_axis]["nodes"] = pd.Index(nodes.index, name=(name_axis, "nodes"))
        self.axes[name_axis]["number_of_nodes"] = nodes.size

        # Group node with axis
        self.node_mapping_.update(dict_build([(name_axis, self.axes[name_axis]["nodes"])]))

        # Assign component colors
        if colors is None:
            colors = "white"
        if is_color(colors):
            colors = dict_build([(colors, self.axes[name_axis]["nodes"])])
        if is_dict(colors):
            colors = pd.Series(colors)
        if not is_color(colors):
            if is_nonstring_iterable(colors) and not isinstance(colors, pd.Series):
                colors = pd.Series(colors, index=self.axes[name_axis]["nodes"])
        self.axes[name_axis]["colors"] = pd.Series(colors[self.axes[name_axis]["nodes"]], name=(name_axis, "node_colors"))

        # Assign component sizes
        if sizes is None:
            sizes = 100
        if is_number(sizes):
            sizes = dict_build([(sizes, self.axes[name_axis]["nodes"])])
        if is_dict(sizes):
            sizes = pd.Series(sizes)
        self.axes[name_axis]["sizes"] = pd.Series(sizes[nodes.index], name=(name_axis, "node_sizes"))

    # Compile the data for plotting
    def compile(self, split_theta_degree=None, inner_radius=None, theta_center=90, axis_normalize=True, axis_maximum=1000):
        """
        inner_radius should be similar units to axis_maximum
        """
        number_of_axes = len(self.axes)
        if split_theta_degree is None:
            split_theta_degree = (360/number_of_axes)*0.16180339887
        self.split_theta_degree = split_theta_degree
        self.axis_maximum = axis_maximum
        if inner_radius is None:
            if axis_normalize:
                inner_radius = (1/5)*self.axis_maximum
            else:
                inner_radius = 3
        self.inner_radius = inner_radius
        self.outer_radius = self.axis_maximum - self.inner_radius
        self.theta_center = theta_center
        # Adjust all of the node_positions
        for i, query_axis in enumerate(self.axes):
            # If the axis is normalized, force everything between the minimum position and the `outer_radius` (that is, the axis_maximum - inner_radius.  This ensures the axis_maximum is actually what is defined)
            if axis_normalize:
                node_positions = self.axes[query_axis]["node_positions"]
                self.axes[query_axis]["node_positions_normalized"] = normalize_minmax(node_positions, feature_range=(min(node_positions), self.outer_radius) )
            else:
                self.axes[query_axis]["node_positions_normalized"] = self.axes[query_axis]["node_positions"].copy()
            # Offset the node positions by the inner radius
            self.axes[query_axis]["node_positions_normalized"] = self.axes[query_axis]["node_positions_normalized"] + self.inner_radius

        # Adjust all of the axes angles
        for i, query_axis in enumerate(self.axes):
            # If the axis is in single mode
            if not self.axes[query_axis]["split_axis"]:
                # If the query axis is the first then the `theta_add` will be 0
                theta_add = (360/number_of_axes)*i
                self.axes[query_axis]["theta"] = np.array([self.theta_center + theta_add])
            else:
                theta_add = (360/number_of_axes)*i
                self.axes[query_axis]["theta"] = np.array([self.theta_center + theta_add - split_theta_degree,
                                                           self.theta_center + theta_add + split_theta_degree])
            self.axes[query_axis]["theta"] = np.deg2rad(self.axes[query_axis]["theta"])

        # Nodes
        self.nodes_ = list()
        for axes_data in self.axes.values():
            self.nodes_ += list(axes_data["nodes"])
        assert len(self.nodes_) == len(set(self.nodes_)), "Axes cannot contain duplicate nodes"
        self.number_of_nodes_ = len(self.nodes_)
        
        # Edges
        self.edges_ = list(map(frozenset, combinations(self.nodes_, r=2)))
        self.number_of_edges_ = len(self.edges_)
        
        # Axes
        self.axes_preview_ = pd.Series(dict(zip(self.axes.keys(), map(lambda data:list(data["nodes"]), self.axes.values()))), name="Axes preview")
        self.axes_preview_.index = self.axes_preview_.index.map(lambda name_axis: "{}. {} ({})".format(self.axes_preview_.index.get_loc(name_axis), name_axis, len(self.axes_preview_[name_axis])))

        # Compile
        self.compiled = True

    def _get_quadrant_info(self, theta_representative):
        # 0/360
        if theta_representative == np.deg2rad(0):
            horizontalalignment = "left"
            verticalalignment = "center"
            quadrant = 0
        # 90
        if theta_representative == np.deg2rad(90):
            horizontalalignment = "center"
            verticalalignment = "bottom"
            quadrant = 90
        # 180
        if theta_representative == np.deg2rad(180):
            horizontalalignment = "right"
            verticalalignment = "center"
            quadrant = 180
        # 270
        if theta_representative == np.deg2rad(270):
            horizontalalignment = "center"
            verticalalignment = "top"
            quadrant = 270

        # Quadrant 1
        if np.deg2rad(0) < theta_representative < np.deg2rad(90):
            horizontalalignment = "left"
            verticalalignment = "bottom"
            quadrant = 1
        # Quadrant 2
        if np.deg2rad(90) < theta_representative < np.deg2rad(180):
            horizontalalignment = "right"
            verticalalignment = "bottom"
            quadrant = 2
        # Quadrant 3
        if np.deg2rad(180) < theta_representative < np.deg2rad(270):
            horizontalalignment = "right"
            verticalalignment = "top"
            quadrant = 3
        # Quadrant 4
        if np.deg2rad(270) < theta_representative < np.deg2rad(360):
            horizontalalignment = "left"
            verticalalignment = "top"
            quadrant = 4
        return quadrant, horizontalalignment, verticalalignment

    def plot(self,
             title=None,
             # Arc style
             arc_style="curved",
             # Show components
             show_axis=True,
             show_nodes=True,
             show_edges=True,
             show_border = False,
             show_axis_labels=True,
             show_node_labels=False,
             show_polar_grid=False,
             show_cartesian_grid=False,
             # Colors
             axis_color=None,
             edge_colors=None,
             background_color=None,
             # Alphas
             edge_alpha=0.5,
             node_alpha=0.8,
             axis_alpha=0.618,
             # Keywords
             title_kws=dict(),
             axis_kws=dict(),
             axis_label_kws=dict(),
             node_label_kws=dict(),
             node_label_line_kws=dict(),
             node_kws=dict(),
             edge_kws=dict(),
             legend_kws=dict(),
             legend_label_kws=dict(),
             # Figure
             style="dark",
             edge_linestyle="-",
             axis_linestyle="-",
             node_label_linestyle=":",
             legend_markerstyle="s",
             legend=None,
#              polar=True,
             ax_polar=None,
             ax_cartesian=None,
             clip_edgeweight=5,
             granularity=100,
             func_edgeweight=None,
             figsize=(10,10),
             # Padding
             pad_axis_label = "infer",
             pad_node_label = 5,
#              pad_node_label_line = 0,
#              node_label_position_vertical_axis="right",
            ):
        polar = True #! Address this in future versions
        assert self.compiled == True, "Please `compile` before plotting"
        accepted_arc_styles = {"curved", "linear"}
        assert_acceptable_arguments(arc_style, accepted_arc_styles)
        if arc_style == "linear":
            granularity = 2
        if style in ["dark",  "black", "night",  "sith"]:
            style = "dark_background"
        if style in ["light", "white", "day", "jedi"] :
            style = "seaborn-white"
            
        with plt.style.context(style):
            # Create figure
            if ax_polar is not None:
                fig = plt.gcf()
                figsize = fig.get_size_inches()
            # Polar canvas
            if ax_polar is None:
                fig = plt.figure(figsize=figsize)
                ax_polar = plt.subplot(111, polar=polar)
            # Cartesian canvas
            if ax_cartesian is None:
                ax_cartesian = fig.add_axes(ax_polar.get_position(), frameon=False, polar=False)
            if polar == True:
                y = 0.95
            if polar == False:
                y = 1.1

            # Remove clutter from plot
            ax_polar.grid(show_polar_grid)
            ax_polar.set_xticklabels([])
            ax_polar.set_yticklabels([])
            
            ax_cartesian.grid(show_cartesian_grid)
            ax_cartesian.set_xticklabels([])
            ax_cartesian.set_yticklabels([])
            
            if not show_border: # Not using ax.axis('off') becuase it removes facecolor
                for spine in ax_polar.spines.values():
                    spine.set_visible(False)
                for spine in ax_cartesian.spines.values():
                    spine.set_visible(False)
                    
            node_padding = " "*pad_node_label

            # Default colors
            if axis_color is None:
                if style == "dark_background":
                    axis_color = "white"
                    axis_label_color = "white"
                else:
                    axis_color = "darkslategray"
                    axis_label_color = "black"
            if background_color is not None:
                ax_polar.set_facecolor(background_color)
                ax_cartesian.set_facecolor(background_color)

            # Title
            _title_kws = {"fontweight":"bold", "y":y}
            _title_kws.update(title_kws)
            if "fontsize" not in _title_kws:
                _title_kws["fontsize"] = figsize[0] * np.sqrt(figsize[0])/2 + 2
            # Axis labels
            _axis_label_kws = {"fontweight":None, "color":axis_label_color}
            _axis_label_kws.update(axis_label_kws)
            if "fontsize" not in _axis_label_kws:
                _axis_label_kws["fontsize"] = figsize[0] * np.sqrt(figsize[0])/2
            # Node labels
            _node_label_kws = {"fontsize":12}
            _node_label_kws.update(node_label_kws)
            _node_label_line_kws = {"linestyle":node_label_linestyle, "color":axis_color}
            _node_label_line_kws.update(node_label_line_kws)
            # Axis plotting
            _axis_kws = {"linewidth":3.382, "alpha":axis_alpha, "color":axis_color, "linestyle":axis_linestyle,  "zorder":0}
            _axis_kws.update(axis_kws)
            # Edge plotting
            _edge_kws = {"alpha":edge_alpha, "linestyle":edge_linestyle} #  "zorder", _node_kws["zorder"]+1}
            _edge_kws.update(edge_kws)
            # Node plotting
            _node_kws = {"linewidth":1.618, "edgecolor":axis_color, "alpha":node_alpha,"zorder":2}
            _node_kws.update(node_kws)

            # Legend plotting
            _legend_label_kws = {"marker":legend_markerstyle, "markeredgecolor":axis_color, "markeredgewidth":1, "linewidth":0}
            _legend_label_kws.update(legend_label_kws)
            _legend_kws = {'fontsize': 15, 'frameon': True, 'facecolor': background_color, 'edgecolor': axis_color, 'loc': 'center left', 'bbox_to_anchor': (1.1, 0.5), "markerscale":1.6180339887}
            _legend_kws.update(legend_kws)
            
            # Edge info
            edges = self.weights[self.edges_].abs()

            if func_edgeweight is not None:
                edges = func_edgeweight(edges)
            if clip_edgeweight is not None:
                edges = np.clip(edges, a_min=None, a_max=clip_edgeweight)
            if edge_colors is None:
                edge_colors = axis_color
            if is_color(edge_colors):
                edge_colors = dict_build([(edge_colors, edges.index)])
            if is_dict(edge_colors):
                edge_colors = pd.Series(edge_colors)
            if not is_color(edge_colors):
                if is_nonstring_iterable(edge_colors) and not isinstance(edge_colors, pd.Series):
                    edge_colors = pd.Series(edge_colors, index=edges.index)
            edge_colors = pd.Series(edge_colors[edges.index], name="edge_colors").to_dict()
            # ================
            # Plot edges
            # ================
            # Draw edges
            if show_edges:
                for (edge, weight) in edges.iteritems():
                    if abs(weight) > 0:
                        node_A, node_B = edge
                        name_axis_A = self.node_mapping_[node_A]
                        name_axis_B = self.node_mapping_[node_B]

                        # Check axis
                        intraaxis_edge = (name_axis_A == name_axis_B)

                        # Within axis edges
                        if intraaxis_edge:
                            name_consensus_axis = name_axis_A
                            # Plot edges on split axis
                            if self.axes[name_consensus_axis]["split_axis"]:
                                # print(type(edge), edge, edge in edge_colors)
                                color = edge_colors[edge]
                                # Draw edges between same axis
                                # Node A -> B
                                ax_polar.plot([*self.axes[name_consensus_axis]["theta"]], # Unpack
                                        [self.axes[name_consensus_axis]["node_positions_normalized"][node_A], self.axes[name_consensus_axis]["node_positions_normalized"][node_B]],
                                        c=color,
                                        linewidth=weight,
                                        **_edge_kws,
                                )
                                # Node B -> A
                                ax_polar.plot([*self.axes[name_consensus_axis]["theta"]], # Unpack
                                        [self.axes[name_consensus_axis]["node_positions_normalized"][node_B], self.axes[name_consensus_axis]["node_positions_normalized"][node_A]],
                                        c=color,
                                        linewidth=weight,
                                        **_edge_kws,
                            )

                        # Between axis
                        if not intraaxis_edge:
                            axes_ordered = list(self.axes.keys())
                            terminal_axis_edge = False
                            # Last connected to the first
                            if (name_axis_A == axes_ordered[-1]):
                                if (name_axis_B == axes_ordered[0]):
                                    thetas = [self.axes[name_axis_A]["theta"].max(), self.axes[name_axis_B]["theta"].min()]
                                    radii = [self.axes[name_axis_A]["node_positions_normalized"][node_A], self.axes[name_axis_B]["node_positions_normalized"][node_B]]
                                    terminal_axis_edge = True
                            # First connected to the last
                            if (name_axis_A == axes_ordered[0]):
                                if (name_axis_B == axes_ordered[-1]):
                                    thetas = [self.axes[name_axis_B]["theta"].max(), self.axes[name_axis_A]["theta"].min()]
                                    radii = [self.axes[name_axis_B]["node_positions_normalized"][node_B], self.axes[name_axis_A]["node_positions_normalized"][node_A]]
                                    terminal_axis_edge = True
                            if not terminal_axis_edge:
                                if axes_ordered.index(name_axis_A) < axes_ordered.index(name_axis_B):
                                    thetas = [self.axes[name_axis_A]["theta"].max(), self.axes[name_axis_B]["theta"].min()]
                                if axes_ordered.index(name_axis_A) > axes_ordered.index(name_axis_B):
                                    thetas = [self.axes[name_axis_A]["theta"].min(), self.axes[name_axis_B]["theta"].max()]
                                radii = [self.axes[name_axis_A]["node_positions_normalized"][node_A], self.axes[name_axis_B]["node_positions_normalized"][node_B]]

                            # Radii node positions
                            #
                            # Necessary to account for directionality of edge.
                            # If this doesn't happen then there is a long arc
                            # going counter clock wise instead of clockwise
                            # If straight lines were plotted then it would be thetas and radii before adjusting for the curve below
                            if terminal_axis_edge:
                                theta_end_rotation = thetas[0]
                                theta_next_rotation = thetas[1] + np.deg2rad(360)
                                thetas = [theta_end_rotation, theta_next_rotation]
                            # Create grid for thetas
                            t = np.linspace(start=thetas[0], stop=thetas[1], num=granularity)
                            # Get radii for thetas
                            radii = interp1d(thetas, radii)(t)
                            thetas = t
                            ax_polar.plot(thetas,
                                    radii,
                                    c=edge_colors[edge],
                                    linewidth=weight,
                                    **_edge_kws,
                                )
                        
            # ===================
            # Plot axis and nodes
            # ===================
            for name_axis, axes_data in self.axes.items():
                # Retrieve
                node_positions = axes_data["node_positions_normalized"]
                colors = axes_data["colors"].tolist() # Needs `.tolist()` for Matplotlib version < 2.0.0
                sizes = axes_data["sizes"].tolist()

                # Positions
                # =========
                # Get a theta value for each node on the axis
                if not axes_data["split_axis"]:
                    theta_single = np.repeat(axes_data["theta"][0], repeats=node_positions.size)
                    theta_vectors = [theta_single]
                # Split the axis so within axis interactions can be visualized
                if axes_data["split_axis"]:
                    theta_split_A = np.repeat(axes_data["theta"][0], repeats=node_positions.size)
                    theta_split_B = np.repeat(axes_data["theta"][1], repeats=node_positions.size)
                    theta_vectors = [theta_split_A, theta_split_B]
                theta_representative = np.mean(axes_data["theta"])

                # Quadrant
                # =======
                quadrant, horizontalalignment, verticalalignment = self._get_quadrant_info(theta_representative)

                # Plot axis
                # =========
                if show_axis:
                    for theta in axes_data["theta"]:
                        ax_polar.plot(
                            2*[theta],
                            [min(node_positions), max(node_positions)],
                            **_axis_kws,
                        )
                        
                # Plot axis labels
                # ================
                if show_axis_labels:
                    if pad_axis_label == "infer":
                        pad_axis_label = 0.06180339887*(node_positions.max() - node_positions.min())
                        
                    ax_polar.text(
                        s = name_axis,
                        x = theta_representative,  
                        y = node_positions.size + node_positions.max() + pad_axis_label,
                        horizontalalignment=horizontalalignment,
                        verticalalignment=verticalalignment,                        
                        **_axis_label_kws,
                    )

                # Plot nodes
                # ========
                if show_nodes:
                    for theta in theta_vectors:
                        # Filled
                        ax_polar.scatter(
                            theta,
                            node_positions,
                            c=axes_data["colors"],
                            s=axes_data["sizes"],
                            marker=axes_data["node_style"],
                            **_node_kws,
                        )
                        # Empty
                        ax_polar.scatter(
                            theta,
                            node_positions,
                            facecolors='none',
                            s=axes_data["sizes"],
                            marker=axes_data["node_style"],
                            alpha=1,
                            zorder=_node_kws["zorder"]+1,
                            # zorder=-1,
                            edgecolor=_node_kws["edgecolor"],
                            linewidth=_node_kws["linewidth"],
                        )

                    # Plot node labels
                    # ================
                    index_labels = node_positions.index
                    if is_nonstring_iterable(show_node_labels):
                        index_labels = pd.Index(show_node_labels) & index_labels
                        show_node_labels = True
                    
                    if show_node_labels:
                        if not polar:
                            warnings.warn("`show_node_labels` is not available in version: {}".format(__version__))
                        else:
                            horizontalalignment_nodelabels = None
                            for name_node, r in node_positions[index_labels].iteritems():  
                                #! Address this in future version
#                                 # Vertical axis case
#                                 vertical_axis_left = (quadrant in {90,270}) and (node_label_position_vertical_axis == "left")
#                                 vertical_axis_right = (quadrant in {90,270}) and (node_label_position_vertical_axis == "right")
#                                 if vertical_axis_left:
#                                     horizontalalignment_nodelabels = "right" # These are opposite b/c nodes should be on the left which means padding on the right
#                                 if vertical_axis_right:
#                                     horizontalalignment_nodelabels = "left" # Vice versa

                                # Pad on the right and push label to left
#                                 if (quadrant == 3) or vertical_axis_left:
#                                     node_label = "{}{}".format(name_node,node_padding)
#                                     theta_anchor_padding = max(axes_data["theta"])

#                                 # Pad on left and push label to the right
#                                 if (quadrant == 4) or vertical_axis_right:
#                                     node_label = "{}{}".format(node_padding,name_node)
#                                     theta_anchor_padding = min(axes_data["theta"])

                                # theta_anchor is where the padding ends up

                                # Pad on the right and push label to left
                                if quadrant in {2,3, 0, 180} :
                                    node_label = "{}{}".format(name_node,node_padding)
                                    theta_anchor_padding = max(axes_data["theta"])
                                    x, y = polar_to_cartesian(r, theta_anchor_padding)
                                    xs_line = [-self.axis_maximum, x]
                                    x_text = -self.axis_maximum
                                    horizontalalignment_nodelabels = "right" 
                                    
                                    
                                # Pad on the right and push label to left
                                if quadrant in {1,4, 90, 270} :
                                    node_label = "{}{}".format(node_padding,name_node)
                                    theta_anchor_padding = min(axes_data["theta"])
                                    x, y = polar_to_cartesian(r, theta_anchor_padding)
                                    xs_line = [x, self.axis_maximum]
                                    x_text = self.axis_maximum
                                    horizontalalignment_nodelabels = "left" 
                                    
                                # Node label line
                                ax_cartesian.plot(
                                    xs_line, 
                                    [y, y], 
                                    **_node_label_line_kws,
                                )
                                # Node label text
                                ax_cartesian.text(
                                    x=x_text, 
                                    y=y, 
                                    s=node_label, 
                                    horizontalalignment=horizontalalignment_nodelabels,
                                    verticalalignment="center",
                                    **_node_label_kws,
                                )


             # Adjust limits
             # ===========
            r_max = max(ax_polar.get_ylim()) 

            if title is not None:
                fig.suptitle(title, **_title_kws)
                
            ax_cartesian.set_xlim(-r_max, r_max)
            ax_cartesian.set_ylim(-r_max, r_max) 
            
            # Plot Legend
            # ===========
            if legend is not None:
                assert is_dict_like(legend), "`legend` must be dict-like"
                handles = list()
                for label, color in legend.items():
                    handle = plt.Line2D([0,0],[0,0], color=color, **_legend_label_kws)
                    handles.append(handle)
                ax_cartesian.legend(handles, legend.keys(), **_legend_kws)

            return fig, [ax_polar, ax_cartesian]


   # Axis data
    def get_axis_data(self, name_axis=None, field=None):
        if name_axis is None:
            print("Available axes:", set(self.axes.keys()), file=sys.stderr)
        else:
            assert name_axis in self.axes, "{} is not in the axes".format(name_axis)

            df =  pd.DataFrame(dict_filter(self.axes[name_axis], ["colors", "sizes", "node_positions", "node_positions_normalized"]))
            if self.compiled:
                df["theta"] = [self.axes[name_axis]["theta"]]*df.shape[0]
            df.index.name = name_axis
            if field is not None:
                return df[field]
            else:
                return df
    # Connections
    def get_axis_connections(self, name_axis=None,  sort_by=None, ascending=False, return_multiindex=False):
        assert self.compiled == True, "Please `compile` before getting connections"
        if name_axis is not None:
            assert name_axis in self.axes, "{} is not in the available axes for `name_axis`.  Please add and recompile or choose one of the available axes:\n{}".format(name_axis, list(self.axes.keys()))

        df_dense = condensed_to_dense(self.weights, index=self.nodes_)
        df_connections = df_dense.groupby(self.node_mapping_, axis=1).sum()
        if name_axis is not None:
            idx_axis_nodes = self.axes[name_axis]["nodes"]
            df_connections = df_connections.loc[idx_axis_nodes,:]
            df_connections.index.name = name_axis
        if sort_by is not None:
            assert sort_by in self.axes, f"{sort_by} is not in the available axes for `sort_by`.  Please add and recompile or choose one of the available axes:\n{self.axes.keys()}"
            df_connections = df_connections.sort_values(by=sort_by, axis=0, ascending=ascending)
        if return_multiindex:
            df_connections.index = pd.MultiIndex.from_tuples(df_connections.index.map(lambda id_node: (self.node_mapping_[id_node], id_node)))
        return df_connections

    # Stats
    # =====
    def compare(self, data, func_stats=mannwhitneyu, name_stat=None, tol=1e-10):
        """
        Compare the connections between 2 Hives or adjacencies using the specified axes assignments.
        """
        assert self.compiled == True, "Please `compile` before comparing adjacencies"
        assert_acceptable_arguments(type(data), {pd.DataFrame, Symmetric, Hive})
        if isinstance(data, (Hive, Symmetric)):
            df_dense__query = condensed_to_dense(data.weights)
        
        if isinstance(data, pd.DataFrame):
            assert is_symmetric(data, tol=tol)
            df_dense__query = data
        assert set(self.nodes_) <= set(df_dense__query.index), "`data` must contain all nodes from reference Hive"
        
        df_dense__reference = self.to_dense()
        
        d_statsdata = OrderedDict()

        # Get nodes
        d_statsdata = OrderedDict()
        for id_node in df_dense__reference.index:
            # Get axis groups
            stats_axes_data = list()
            for name_axis in self.axes:
                idx_axis_nodes = self.axes[name_axis]["nodes"]
                n = self.axes[name_axis]["number_of_nodes"]
                # Get comparison data
                u = df_dense__reference.loc[id_node,idx_axis_nodes]
                v = df_dense__query.loc[id_node,idx_axis_nodes]
                # Get stats
                stat, p = func_stats(u,v)
                if name_stat is None:
                    if hasattr(func_stats, "__name__"):
                        name_stat = func_stats.__name__
                    else:
                        name_stat = str(func_stats)
                # Store data
                row = pd.Series(OrderedDict([
                 ((name_axis, "number_of_nodes"), n),
                 ((name_axis, "∑(reference)"), u.sum()),
                 ((name_axis, "∑(query)"), v.sum()),
                 ((name_axis, name_stat), stat),
                 ((name_axis, "p_value"), p)
                ]))
                stats_axes_data.append(row)
            # Build pd.DataFrame
            d_statsdata[id_node] = pd.concat(stats_axes_data)
        return pd.DataFrame(d_statsdata).T

    # Exports
    # =======
    def to_networkx(self, into=None, **attrs):
        if into is None:
            into = nx.Graph
        metadata = { "node_type":self.node_type, "edge_type":self.edge_type}
        metadata.update(attrs)
        graph = into(name=self.name, **metadata)
        for (node_A, node_B), weight in self.weights.iteritems():
            graph.add_edge(node_A, node_B, weight=weight)
        return graph
    
    def to_symmetric(self, nodes=None, **symmetric_kws):
        _symmetric_kws = dict(node_type=self.node_type, edge_type=self.edge_type, association="network", name=self.name)
        _symmetric_kws.update(symmetric_kws)
        if nodes is not None:
            assert set(nodes) <= set(self.nodes_in_hive), "Not all `nodes` available in Hive"
            edges = list(combinations(nodes, r=2))
            weights = self.weights[edges]
        else:
            weights = self.weights
        return Symmetric(weights, **_symmetric_kws)

    def to_file(self, path:str, compression="infer"):
        write_object(self, path=path, compression=compression)
        return self

    def to_dense(self, nodes=None, fill_diagonal=np.nan):
        if nodes is not None:
            assert set(nodes) <= set(self.nodes_in_hive), "Not all `nodes` available in Hive"
        else:
            nodes = self.nodes_in_hive
        return condensed_to_dense(self.weights, index=nodes, fill_diagonal=fill_diagonal)
 
    def copy(self):
        return copy.deepcopy(self)
    

