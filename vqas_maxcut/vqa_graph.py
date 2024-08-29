import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class vqa_graph:
    """
    Class to represent a graph and perform operations on it.
    
    **Attributes:**
        graph: List of tuples, where each tuple represents an edge between two nodes.
        n_nodes: Number of nodes in the graph.
        n_edges: Number of edges in the graph.
        G: NetworkX graph object.

    **Methods:**
        draw_graph: Draws the graph with the given partition.
        compute_cut: Computes the cut of the given partition.    
    """
    def __init__(self, graph, n_nodes: int = None):
        plt.style.use('https://github.com/kaiuki2000/PitayaRemix/raw/main/PitayaRemix.mplstyle')
        self.graph   = graph
        self.G       = nx.Graph(self.graph)
        self.n_nodes = max(max(self.graph)) + 1 if n_nodes is None else n_nodes
        self.n_edges = len(self.graph)

    # Graph related methods:
    def draw_graph(self, partition: str = None, save_name: str = None, title: str = None) -> None:
        """
        Draws the graph with the given partition.
        
        **Args:**
            partition: List of strings, where each string represents the partition of the corresponding node.
            save_name: Name of the file to save the plot.
            title: Title of the plot.
            
        **Returns:**
            None
        """
        node_colors  = ["#FFC107"] * len(self.G.nodes()) if partition is None \
                  else ["#FFC107" if val == "0" else "#E91E63" for val in partition]
        node_size    = 500
        pos          = nx.spring_layout(self.G)

        # Default title
        plt.title("VQA_Graph")

        if(partition is not None):
            partition0_patch = Patch(color = '#FFC107', label = 'Set 0')
            partition1_patch = Patch(color = '#E91E63', label = 'Set 1')
            plt.legend(handles = [partition0_patch, partition1_patch], loc = 'best', frameon=True, fancybox=True)
            plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
            plt.title(f"VQA_Graph: Partition = {partition}")
            node_colors = [node_colors[node] for node in self.G.nodes()]

        # Overrides the default title, if 'title' is provided
        if(title is not None):
            plt.title(title)
            
        nx.draw_networkx(self.G, pos=pos, with_labels=True, node_color=node_colors, node_size = node_size, \
                         edge_color="#9E9E9E", edgecolors="black", linewidths=1)
        
        if(save_name is not None):
            plt.savefig(save_name, format='png', dpi=600, bbox_inches='tight')
        plt.show()

    def compute_cut(self, partition: str = None) -> int:
        """
        Computes the cut of the given partition.

        **Args:**
            partition: List of strings, where each string represents the partition of the corresponding node.

        **Returns:**
            int: The cut of the given partition.
        """
        assert(partition is not None), "Partition cannot be None."   
         
        cost = 0
        for edge in self.G.edges():
            if(partition[edge[0]] != partition[edge[1]]):
                cost += 1
        return cost
    
    # For computing the Poljak-Turzik lower bound (Regularization term, when using the 'Expectation' cost type, from https://arxiv.org/abs/2401.09421.)
    def poljak_turzik_lower_bound_unweighted(self):
        """
        Computes the Poljak-Turzik lower bound for unweighted graphs.

        **Returns:**
            float: The Poljak-Turzik lower bound for the graph.
        """
        # Compute the total weight of the graph w(G)
        # For unweighted graphs, each edge has a weight of 1
        w_G = self.G.number_of_edges()
        # Compute the minimum spanning tree of the graph
        T_min = nx.minimum_spanning_tree(self.G)
        # Compute the total weight of the minimum spanning tree w(T_min)
        # For unweighted graphs, each edge has a weight of 1
        w_T_min = T_min.number_of_edges()
        # Compute the Poljak-Turzík lower bound
        nu = w_G / 2 + w_T_min / 4
        return nu