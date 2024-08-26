# Required imports
import time # For timing
import math
import itertools
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import clear_output
import cvxpy as cp
from scipy.linalg import sqrtm
from scipy import optimize
from pennylane.operation import Operation

# RXX gate definition: 2-qubit Molmer-Sorensen gate
class RXX(Operation):
    num_params = 1
    num_wires = 2
    par_domain = "R"

    grad_method = "A"
    grad_recipe = None # This is the default, but we write it down explicitly here.

    generator = [(qml.PauliX(0) @ qml.PauliX(1)).matrix, -0.5]

    @staticmethod
    def compute_decomposition(theta, wires):
        return [qml.PauliRot(theta, 'XX', wires=wires)]

    @staticmethod
    def _matrix(*params):
        theta = params[0]
        c = np.cos(0.5 * theta)
        s = np.sin(0.5 * theta)
        return np.array(
            [
                [c, 0, 0, -s],
                [0, c, -s, 0],
                [0, -s, c, 0],
                [-s, 0, 0, c]
            ]
        )

    def adjoint(self):
        return RXX(-self.data[0], wires=self.wires)

class VQA_Graph:
    """
    This class implements the VQA_Graph class, which is used to solve the MaxCut problem.
    Has methods for drawing the graph, computing the cut of a given partition, and implementing
    the I-QA/QE, QAOA and QEMC VQAs, for the MaxCut problem.
    """

    def __init__(self, graph, n_nodes: int = None):
        """
        Initializes the VQA_Graph class.

        Args:
            graph (list[tuple] or list[list]): The graph to be used for the VQA problem.
            
        Returns:
            None
        """

        self.graph = graph; self.G = nx.Graph(self.graph)
        self.n_nodes = max(max(self.graph)) + 1 if n_nodes is None else n_nodes
        self.n_edges = len(self.graph)

        # # Matplolib style
        # plt.style.use('https://github.com/kaiuki2000/PitayaRemix/raw/main/PitayaRemix.mplstyle')

    # Graph related methods:
    def draw_graph(self, partition: str = None, save_name: str = None, title: str = None) -> None:
        """
        Draws the graph with the given partition if any.

        Args:
            partition (str, optional): The partition to be drawn. Defaults to None.

        Returns:
            None
        """
        
        node_colors, node_size = ["#FFC107" for node in self.G.nodes()] if partition is None else ["#FFC107" if val == "0" else "#E91E63" for val in partition], 500
        edge_colors, edge_width = ["#9E9E9E" for edge in self.G.edges()], 2
        pos = nx.spring_layout(self.G)
        
        if(partition is not None):
            partition0_patch = Patch(color = '#FFC107', label = 'Set 0')
            partition1_patch = Patch(color = '#E91E63', label = 'Set 1')
            plt.legend(handles = [partition0_patch, partition1_patch], loc = 'best', frameon=True, fancybox=True)
            plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
            plt.title(f"VQA_Graph: Partition = {partition}")
            node_colors = [node_colors[node] for node in self.G.nodes()]
        else:
            plt.title("VQA_Graph")

        # Overrides the default title, if 'title' is provided
        if(title is not None): plt.title(title)
            
        nx.draw_networkx(self.G, pos=pos, with_labels=True, node_color=node_colors, node_size = node_size, edge_color="#9E9E9E", edgecolors="black", linewidths=1)
        if(save_name is not None): plt.savefig(save_name, format='png', dpi=600, bbox_inches='tight')
        plt.show()

    def compute_cut(self, partition: str = None) -> int:
        """
        Computes the cut of the given partition if any.

        Args:
            partition (str, optional): The partition to be drawn. Defaults to None.

        Returns:
            int: The cut of the given partition.
        """
    
        assert(partition is not None), "Partition cannot be None."   
         
        cost = 0
        for edge in self.G.edges():
            if(partition[edge[0]] != partition[edge[1]]):
                cost += 1
        return cost
    
    def number_of_parameters(self, ansatz_type, n_layers, k):
        """
        Function to calculate the number of parameters in the ansatz.
        Only works for the Brickwork, Parity, and SEL ansatz types, within the semi-problem-informed    "Parity-like_Inspired" ansatz scheme.
    
        Args:
        - self: VQA_Graph object.
        - ansatz_type: str, specifying the ansatz type.
        - n_layers: int, specifying the number of layers.
        - k: int, specifying the desired order of polynomial compression.
    
        Returns:
        - int, specifying the number of parameters in the ansatz.
        """
    
        if(k == 2): n_qubits = math.ceil((1 + np.sqrt(1 + 8*self.n_nodes))/2) # k = 2
        else: n_qubits = math.ceil(optimize.newton(lambda n: n * (n - 1) * (n - 2) - 6 * self.n_nodes, self.n_nodes)) # k = 3
    
        if(ansatz_type == 'Brickwork'): 
            # *** Brickwork ansatz: ***       
            return n_layers * n_qubits * 4
            # *** Brickwork ansatz: ***
    
        elif(ansatz_type == 'Parity'):
            # *** Parity ansatz: ***
            def get_encoding(n_qubits, k):
                encoding = list(itertools.combinations(range(n_qubits), k))
                return encoding
            
            encoding = get_encoding(n_qubits, k)[:self.n_nodes]; terms = []
            for i, j in self.graph: terms.append(list(set(encoding[i]).union(encoding[j])))
            clean_terms = [ele for ind, ele in enumerate(terms) if ele not in terms[:ind]]
    
            return n_layers * ( n_qubits + len(clean_terms) )
            # *** Parity ansatz: ***
    
        elif(ansatz_type == 'SEL'): # All the Rx gates feature the same parameter, within each layer.
            # *** SEL ansatz: ***
            return n_layers * n_qubits * 3
            # *** SEL ansatz: ***
    
        else:
            print('Invalid ansatz type!')
            return None
    
    # VQA implementations:
    def iQAQE_MaxCut(self, n_qubits = None, n_layers = None, device = 'default.qubit', shots = None,
                     parameters = None, rs = None, draw_circuit = False, B = None, basis_states_lists = None,
                     non_deterministic_CNOT = False, rel_tol = 0, abs_tol = 0, MaxCut = None, max_iter = 100, 
                     cost_plot = True, raw_plot = False, nodes_probs_plot = True, **kwargs):
        """
        Implements the I-QA/QE VQA, for the MaxCut problem.

        Args:
            n_qubits (int, optional): The number of qubits to be used. Defaults to None.
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            rs (int, optional): Ranges for the \"Strongly Entangling Layers\" ansatz. Defaults to None.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            B (int, optional): The number of "blue" nodes. Defaults to None. (Actually, the "blue" nodes are the "yellow" ones, *I think*).
            basis_states_lists (list[list], optional): List of lists of basis states assigned to each graph node. Defaults to None.
            non_deterministic_CNOT (bool, optional): Whether to use the non-deterministic CNOT gate or not. Defaults to False.
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            raw_plot (bool, optional): Whether to plot the raw data (QCircuit output != Nodes' probabilities) or not. Defaults to False.
            nodes_probs_plot (bool, optional): Whether to plot the nodes' probabilities or not. Defaults to True.
        Returns:
            list: The probabilities of each basis state.
            list: The probabilities of each node.
            float: The "optimal" cost function value.
            list: The cost function values for each iteration.
            list: The approximation ratio values for each iteration.
            list: The "optimal" parameters.
            list: The "optimal" partition.
            float: The training time.
        """

        assert(n_qubits is not None and n_layers is not None and parameters is not None and B is not None and basis_states_lists is not None), "n_qubits, n_layers, parameters, B and basis_states_lists cannot be None."
        assert(len(basis_states_lists) == self.n_nodes), "The number of basis states lists must be equal to the graph's number of nodes."
        if(non_deterministic_CNOT): assert(parameters.shape == (n_layers, n_qubits, 4)), "The parameters shape must be (n_layers, n_qubits, 4). [Non-deterministic CNOT-based ansatz]."
        else: assert(parameters.shape == (n_layers, n_qubits, 3)), "The parameters shape must be (n_layers, n_qubits, 3). [Deterministic CNOT-based ansatz]."
        if(non_deterministic_CNOT): print("Using non-deterministic CNOT gates. [RX-paramaterized.]") # Status message
            
        # Default 'rs' values
        if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]
        
        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)
        
        # Circuit setup
        @qml.qnode(dev)
        def circuit(params):
            global c; c = 0
            # I-QA/QE (Interpolated-QAOA/QEMC) layer definition:
            def iQAQE_layer(params_layer):
                global c
                for i in range(n_qubits):
                    qml.RX(params_layer[i][0], wires = i)
                    qml.RY(params_layer[i][1], wires = i)
                    qml.RZ(params_layer[i][2], wires = i)
                for j in range(n_qubits):
                    if(non_deterministic_CNOT): qml.RX(params_layer[j][3], wires = j)
                    qml.CNOT(wires = [j, (j+rs[c]) % n_qubits])
                c += 1

            # Actual "Strongly Entangling Layers" ansatz:
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            qml.layer(iQAQE_layer, n_layers, params)

            return qml.probs()

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_qubits = {n_qubits}, n_layers = {n_layers}, rs = {rs}.')
            fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters)
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        # cte = kwargs.get('cte', 1.0); print(f"Softmax scaling constant: {cte}.")
        print(f"iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        # # Testing using the 'softmax' function, instead of mere normalization.
        # def softmax(x, cte = 1.0):
        #     """Compute softmax values for each sets of scores in x."""
        #     return np.exp(cte * x) / np.sum(np.exp(cte * x), axis=0)

        def compute_nodes_probs(probabilities, basis_states_lists):
            """
            Computes the probabilities associated with each node.

            Args:
                probabilities (list): The probabilities of each basis state.
                basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.

            Returns:
                list: The probabilities associated with each node.
            """

            nodes_probs = []
            for sublist in basis_states_lists:
                node_prob = 0
                for basis_state in sublist: node_prob += probabilities[int(basis_state, 2)]
                nodes_probs.append(node_prob)
            nodes_probs = [prob/sum(nodes_probs) for prob in nodes_probs]
            # nodes_probs = softmax(np.array(nodes_probs), cte)
            return nodes_probs
        
        # Define the cost function
        def objective(params):
            """
            This function computes the value of the cost function.

            Args:
                params (array): Array of parameters.

            Returns:
                float: The value of the cost function.
            """

            # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
            probs = circuit(params); cost  = 0
            nodes_probs = compute_nodes_probs(probs, basis_states_lists)

            # Second - Compute the cost function itself
            for edge in self.graph:
                # j and k are the nodes connected by the edge
                # 0: j, 1: k
                d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                cost += edge_cost
            return cost
        
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end='\r')
        while(True):
            parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            if (MaxCut is not None):
                # Compute the approximation ratio, from the computed partition's cut
                approx_ratio = 0
                # Just take the exact distribution
                probabilities = circuit(parameters)
                # Computing the probaility associated to each node, from the 'n_qubit' probability distribution
                nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
                # Get the computed partition
                partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
                # Compute the approximation ratio, from the computed partition's cut
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
            
            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # abs_tol and rel_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            # Plotting the cost function
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        # "Sample measured bitstrings 3*N**2 times". No need to to this, since I'm doing this analytically:
        # Just take the exact distribution
        probabilities = circuit(parameters)
        
        # Computing the probaility associated to each node, from the 'n_qubit' probability distribution
        nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)

        # Get the computed partition
        partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

        # Only 'meaningful'/'readable' for small graphs.
        if(raw_plot):
            # Get the basis states
            basis_states = [format(i, '0'+str(n_qubits)+'b') for i in range(2**n_qubits)]
    
            # Plot the probabilities
            plt.figure(figsize=(8, 6))
            plt.bar(basis_states, probabilities)
            plt.xlabel("Basis States")
            plt.ylabel("Probabilities")
            plt.title("Probabilities vs. Basis States")
            plt.show()

            # Print most frequently sampled bitstring
            most_freq_bit_string = np.argmax(probabilities)
            print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

        if(nodes_probs_plot):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(nodes_probs); x = range(num_outcomes); threshold = 1 / (2*B)
            plt.hlines(threshold, -1, num_outcomes, colors='C2', linestyles='dashed',
                       label=r'$\frac{1}{2B} =$' + f'{threshold:.3f} (B = {int(B)})')
            plt.xlim(-1, num_outcomes); plt.xticks(x)

            # Plot the probability distribution
            plt.bar(x, nodes_probs, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers})')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        return probabilities, nodes_probs, cost, cost_vec, ar_vec, parameters, partition, train_time
    
    def QEMC_MaxCut(self, n_layers = None, device = 'default.qubit', shots = None, parameters = None, 
                    rs = None, draw_circuit = False, B = None, rel_tol = 0, abs_tol = 0, MaxCut = None,
                    non_deterministic_CNOT = False, max_iter = 100, cost_plot = True, nodes_probs_plot = True, **kwargs):
        """
        Implements the QEMC VQA, for the MaxCut problem.

        Args:
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            rs (int, optional): Ranges for the \"Strongly Entangling Layers\" ansatz. Defaults to None.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            B (int, optional): The number of "blue" nodes. Defaults to None. (Actually, the "blue" nodes are the "yellow" ones, *I think*).
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            non_deterministic_CNOT (bool, optional): Whether to use the non-deterministic CNOT gate or not. Defaults to False.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            nodes_probs_plot (bool, optional): Whether to plot the nodes' probabilities or not. Defaults to True.

        Returns:
            list: The probabilities of each basis state/node.
            float: The "optimal" cost function value.
            list: The cost function values for each iteration.
            list: The approximation ratio values for each iteration.
            list: The "optimal" parameters.
            list: The "optimal" partition.
            float: The training time.
        """

        assert(n_layers is not None and parameters is not None and B is not None), "n_layers, parameters and B cannot be None."
        if(non_deterministic_CNOT): print("Using non-deterministic CNOT gates. [RX-paramaterized.]") # Status message

        # Define the number of qubits
        n_qubits = math.ceil(np.log2(self.n_nodes))

        # Default 'rs' values
        if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]
        
        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)
        
        # Circuit setup
        @qml.qnode(dev)
        def circuit(params):
            global c; c = 0
            # QEMC layer definition (same as iQAQE's [Interpolated-QAOA/QEMC]):
            def QEMC_layer(params_layer): # QEMC_layer, actually.
                global c
                for i in range(n_qubits):
                    qml.RX(params_layer[i][0], wires = i)
                    qml.RY(params_layer[i][1], wires = i)
                    qml.RZ(params_layer[i][2], wires = i)
                for j in range(n_qubits):
                    if(non_deterministic_CNOT): qml.RX(params_layer[j][3], wires = j)
                    qml.CNOT(wires = [j, (j+rs[c]) % n_qubits])
                c += 1

            # Actual "Strongly Entangling Layers" ansatz:
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            qml.layer(QEMC_layer, n_layers, params)

            return qml.probs()

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_qubits = {n_qubits}, n_layers = {n_layers}, rs = {rs}.')
            fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters)
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        print(f"QEMC level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        # Define the cost function
        def objective(params):
            """
            This function computes the value of the cost function.

            Args:
                params (array): Array of parameters.

            Returns:
                float: The value of the cost function.
            """

            probs = circuit(params); cost  = 0
            for edge in self.graph:
                # j and k are the nodes connected by the edge
                # 0: j, 1: k
                d_jk = np.abs(probs[edge[0]] - probs[edge[1]]); s_jk = probs[edge[0]] + probs[edge[1]]
                edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                cost += edge_cost
            return cost
    
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end = '\r')
        while(True):
            parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            # Compute the approximation ratio, from the computed partition's cut
            if (MaxCut is not None):
                # Compute the approximation ratio, from the computed partition's cut
                approx_ratio = 0
                # Just take the exact distribution
                probabilities = circuit(parameters)
                # Get the computed partition
                partition = ['0' if probability < 1 / (2*B) else '1' for probability in probabilities]
                # Compute the approximation ratio, from the computed partition's cut
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)

            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end = '\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # rel_tol and abs_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break  

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            # Plotting the cost function
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        # "Sample measured bitstrings 3*N**2 times". No need to to this, since I'm doing this analytically:
        # Just take the exact distribution
        probabilities = circuit(parameters)
        
        # Get the computed partition
        partition = ['0' if prob < 1 / (2*B) else '1' for prob in probabilities]

        # Print most frequently sampled bitstring
        most_freq_bit_string = np.argmax(probabilities)
        print(f'Highest probability bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')
        
        if(nodes_probs_plot):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(probabilities); x = range(num_outcomes); threshold = 1 / (2*B)
            plt.hlines(threshold, -1, num_outcomes, colors='C2', linestyles='dashed',
                       label=r'$\frac{1}{2B} =$' + f'{threshold:.3f} (B = {int(B)})')
            plt.xlim(-1, num_outcomes); plt.xticks(x)

            # Plot the probability distribution
            plt.bar(x, probabilities, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers})')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        return probabilities, cost, cost_vec, ar_vec, parameters, partition, train_time

    def QAOA_MaxCut(self, n_layers = None, device = 'default.qubit', shots = None, 
                    parameters = None, draw_circuit = False, rel_tol = 0, abs_tol = 0,
                    MaxCut = None, max_iter = 100, cost_plot = True, probs_plot = True, **kwargs):
        """
        Implements the QAOA VQA, for the MaxCut problem.

        Args:
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            probs_plot (bool, optional): Whether to plot the output's probabilities or not. Defaults to True.

        Returns:
            list: The probabilities of each basis state.
            float: The "optimal" cost function value.
            list: The cost function values for each iteration.
            list: The approximation ratio values for each iteration.
            list: The "optimal" parameters.
            list: The "optimal" partition.
            float: The training time.
        """

        assert(n_layers is not None and parameters is not None), "n_layers and parameters cannot be None."

        # Define the number of qubits
        n_qubits = self.n_nodes

        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)

        # Preliminary definitions: circuit setup
        # Unitary operator 'U_C' (Cost/Problem) with parameters 'gamma'.
        def U_C(gamma):
            for edge in self.graph:
                wire1, wire2 = edge[0], edge[1]
                # Usual terms
                qml.CNOT(wires = [wire1, wire2])
                qml.RZ(gamma, wires = wire2)
                qml.CNOT(wires = [wire1, wire2])

        # Unitary operator 'U_B' (Bias) with parameters 'beta'.
        def U_B(beta):
            for wire in range(n_qubits):
                qml.RX(2 * beta, wires=wire)

        # QAOA circuit
        @qml.qnode(dev)
        def circuit(gammas, betas, probs_flag = False):
            # Apply Hadamards to get the n qubit |+> state.
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)

            # Apply p instances (layers) of unitary operators.
            for i in range(n_layers):
                U_C(gammas[i])
                U_B(betas[i])
            if probs_flag:
                # Measurement phase
                return qml.probs()
            
            # During the optimization phase we are evaluating a term the objective using 'expval'
            H = 0
            for edge in self.graph:
                H += 0.5 * (1 - qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
            return qml.expval(H)

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_qubits = {n_qubits}, n_layers = {n_layers}.')
            fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters[0], parameters[1])
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        print(f"QAOA level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        # Define the cost function
        def objective(params):
            """
            This function computes the value of the cost function.

            Args:
                params (array): Array of parameters.

            Returns:
                float: The value of the cost function.
            """

            # Get the parameters
            gammas, betas = params[0], params[1]
            # Objective for the MaxCut problem
            neg_obj = -1 * circuit(gammas, betas)
            return neg_obj
    
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end='\r')
        while(True):
            parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            # Compute the approximation ratio, from the computed partition's cut
            if (MaxCut is not None):
                # Compute the approximation ratio, from the computed partition's cut
                approx_ratio = 0
                # Just take the exact distribution
                probabilities = circuit(parameters[0], parameters[1], probs_flag = True)
                # Get the computed partition
                partition = format(np.argmax(probabilities), "0"+str(n_qubits)+"b")
                # Compute the approximation ratio, from the computed partition's cut
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)

            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # rel_tol and abs_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break  

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            # Plotting the cost function
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        # "Sample measured bitstrings 3*N**2 times". No need to to this, since I'm doing this analytically:
        # Just take the exact distribution
        probabilities = circuit(parameters[0], parameters[1], probs_flag = True)

        # Print most frequently sampled bitstring
        most_freq_bit_string = np.argmax(probabilities)
        print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')
        # Get the computed partition
        partition = format(most_freq_bit_string, "0"+str(n_qubits)+"b")

        if(probs_plot):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(probabilities); x = range(num_outcomes)

            # Sort the probabilities and the outcomes, to plot only the 10 most probable outcomes
            sorted_zip = sorted(zip(probabilities, x), reverse = True)[:10]
            sorted_outcomes, sorted_probs = [bin(outcome)[2:].zfill(n_qubits) for _, outcome in sorted_zip], [prob for prob, _ in sorted_zip]

            plt.xticks(fontsize = 8)

            # Plot the probability distribution
            plt.bar(sorted_outcomes, sorted_probs, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers}): 10 most likely outcomes')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        return probabilities, cost, cost_vec, ar_vec, parameters, partition, train_time

    # Multi-angle QAOA (MA-QAOA) for the MaxCut problem
    def MA_QAOA_MaxCut(self, diff_Rx = False, n_layers = None, device = 'default.qubit', shots = None, 
                       expectation_flag = False, parameters = None, seed = None, draw_circuit = False,
                       rel_tol = 0, abs_tol = 0, MaxCut = None, max_iter = 100, cost_plot = True,
                       probs_plot = True, **kwargs):
        """
        Implements the Multi-angle QAOA VQA, for the MaxCut problem.

        Args:
            diff_Rx (bool, optional): Whether to use different Rx parameters or not. Defaults to False.
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            expectation_flag (bool, optional): Whether to use the 'Expectation' cost type or not. Defaults to False (Usual ma-QAOA cost.)
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            probs_plot (bool, optional): Whether to plot the output's probabilities or not. Defaults to True.

        Returns:
            list: The probabilities of each basis state.
            float: The "optimal" cost function value.
            list: The cost function values for each iteration.
            list: The approximation ratio values for each iteration.
            list: The "optimal" parameters.
            list: The "optimal" partition.
            float: The training time.
        """

        assert(n_layers is not None), "n_layers cannot be None."

        # Define the 'parameters' array, if it is None
        if(parameters is None):
            assert(seed is not None), "parameters cannot be None, without a seed."
            np.random.seed(seed) # Set the seed, for reproducibility
            parameters = 2 * np.pi * np.random.rand(self.n_edges + self.n_nodes, n_layers, requires_grad=True) if diff_Rx else 2 * np.pi * np.random.rand(self.n_edges + 1, n_layers, requires_grad=True)
        else:
            # This is to ensure that the 'parameters' array, when provided, has the correct shape.
            assert(parameters.shape == (self.n_edges + self.n_nodes, n_layers) if diff_Rx else (self.n_edges + 1, n_layers)), "parameters has the wrong shape."

        # Define the number of qubits
        n_qubits = self.n_nodes

        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)

        # Preliminary definitions: circuit setup
        # Unitary operator 'U_C' (Cost/Problem) with parameters 'gamma'.
        def U_C(gamma):
            for i, edge in enumerate(self.graph):
                wire1, wire2 = edge[0], edge[1]
                # Usual terms
                qml.CNOT(wires = [wire1, wire2])
                qml.RZ(gamma[i], wires = wire2)
                qml.CNOT(wires = [wire1, wire2])

        # Unitary operator 'U_B' (Bias) with parameters 'beta'.
        def U_B(beta):
            for wire in range(n_qubits):
                if(diff_Rx): qml.RX(2 * beta[wire], wires=wire)
                else: qml.RX(2 * beta[0], wires=wire)

        # QAOA circuit
        @qml.qnode(dev)
        def circuit(gammas, betas, probs_flag = False):
            # Apply Hadamards to get the n qubit |+> state.
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)

            # Apply p instances (layers) of unitary operators.
            for i in range(n_layers):
                U_C(gammas[:, i])
                U_B(betas[:, i])
            if probs_flag:
                # Measurement phase
                return qml.probs()
            
            if(not expectation_flag):
                # During the optimization phase we are evaluating a term the objective using 'expval'
                H = 0
                for edge in self.graph:
                    H += 0.5 * (1 - qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
                return qml.expval(H)
            else:
                # To do: Implement the 'Expectation' cost type.
                P_lst = [qml.PauliZ(P) for P in range(self.n_nodes)]
                return [qml.expval(P) for P in P_lst]

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_qubits = {n_qubits}, n_layers = {n_layers}.')
            if(diff_Rx): fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters[:-self.n_nodes], parameters[-self.n_nodes:]) # Change thie.
            else: fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters[:-1], parameters[-1:])
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        print(f"QAOA level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        # For computing the Poljak-Turzik lower bound (Regularization term, when using the 'Expectation' cost type)
        def poljak_turzik_lower_bound_unweighted():
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
        
        # Define the cost function
        def objective(params):
            """
            This function computes the value of the cost function.

            Args:
                params (array): Array of parameters.

            Returns:
                float: The value of the cost function.
            """

            # Get the parameters
            gammas = params[:-self.n_nodes] if diff_Rx else params[:-1]; betas = params[-self.n_nodes:] if diff_Rx else params[-1:]

            if(not expectation_flag):
                # Objective for the MaxCut problem
                neg_obj = -1 * circuit(gammas, betas)
                return neg_obj
            else:
                # To do: Implement the 'Expectation' cost type.
                # Objective for the MaxCut problem
                exp_vals = circuit(gammas, betas)

                # First part of the cost function
                avg_H = 0; k = 2; alpha = n_qubits**(np.floor(k/2)) # Arbitrary value for 'alpha': k = 2
                for i, j in self.graph:
                    avg_H += ( np.tanh(alpha * exp_vals[i]) * np.tanh(alpha * exp_vals[j]) ) # Missing the non-linearity here!

                # Regularization term
                beta = 0.5
                nu = poljak_turzik_lower_bound_unweighted()
                L_reg = beta * nu * (1/self.n_nodes * np.sum([np.tanh(alpha * exp_val)**2 for exp_val in exp_vals]))**2

                # Total cost
                cost = avg_H + L_reg

                return cost
    
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end='\r')
        while(True):
            parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            # Compute the approximation ratio, from the computed partition's cut
            if (MaxCut is not None):
                # Compute the approximation ratio, from the computed partition's cut
                approx_ratio = 0
                # Just take the exact distribution
                probabilities = circuit(parameters[:-self.n_nodes], parameters[-self.n_nodes:], probs_flag = True) if diff_Rx else circuit(parameters[:-1], parameters[-1:], probs_flag = True)
                # Get the computed partition
                partition = format(np.argmax(probabilities), "0"+str(n_qubits)+"b")
                # Compute the approximation ratio, from the computed partition's cut
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)

            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # rel_tol and abs_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break  

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            # Plotting the cost function
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        # "Sample measured bitstrings 3*N**2 times". No need to to this, since I'm doing this analytically:
        # Just take the exact distribution
        probabilities = circuit(parameters[:-self.n_nodes], parameters[-self.n_nodes:], probs_flag = True) if diff_Rx else circuit(parameters[:-1], parameters[-1:], probs_flag = True)

        # Print most frequently sampled bitstring
        most_freq_bit_string = np.argmax(probabilities)
        print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')
        # Get the computed partition
        partition = format(most_freq_bit_string, "0"+str(n_qubits)+"b")

        if(probs_plot):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(probabilities); x = range(num_outcomes)

            # Sort the probabilities and the outcomes, to plot only the 10 most probable outcomes
            sorted_zip = sorted(zip(probabilities, x), reverse = True)[:10]
            sorted_outcomes, sorted_probs = [bin(outcome)[2:].zfill(n_qubits) for _, outcome in sorted_zip], [prob for prob, _ in sorted_zip]

            plt.xticks(fontsize = 8)

            # Plot the probability distribution
            plt.bar(sorted_outcomes, sorted_probs, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers}): 10 most likely outcomes')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        return probabilities, cost, cost_vec, ar_vec, parameters, partition, train_time

    # Multi-angle QAOA (MA-QAOA) for the MaxCut problem, using the iQAQE Framework
    def MA_QAOA_iQAQE_MaxCut(self, diff_Rx = False, n_layers = None, device = 'default.qubit', 
                             draw_circuit = False, B = None, basis_states_lists = None, seed = None,
                             rel_tol = 0, abs_tol = 0, MaxCut = None, max_iter = 100, shots = None,
                             parameters = None, cost_plot = True, raw_plot = False, nodes_probs_plot = True, **kwargs):
        """
        Implements the Multi-angle QAOA VQA, for the MaxCut problem, using the iQAQE Framework.

        Args:
            diff_Rx (bool, optional): Whether to use different Rx parameters or not. Defaults to False.
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            B (int, optional): The number of "blue" nodes. Defaults to None.
            basis_states_lists (list[list]): The basis states assigned to each graph node. Defaults to None.
            seed (int, optional): The seed to be used. Defaults to None.
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            raw_plot (bool, optional): Whether to plot the raw probabilities or not. Defaults to False.
            nodes_probs_plot (bool, optional): Whether to plot the nodes' probabilities or not. Defaults to True.

        Returns:
            list: The probabilities of each basis state.
            float: The "optimal" cost function value.
            list: The cost function values for each iteration.
            list: The approximation ratio values for each iteration.
            list: The "optimal" parameters.
            list: The "optimal" partition.
            float: The training time.
        """

        # Start with the iQAQE method, and then modify it to include the MA-QAOA ansatz.

        assert(n_layers is not None and B is not None and basis_states_lists is not None), "n_layers, B and basis_states_lists cannot be None."
        assert(len(basis_states_lists) == self.n_nodes), "The number of basis states lists must be equal to the graph's number of nodes."
        
        # Define the 'parameters' array, if it is None
        if(parameters is None):
            assert(seed is not None), "parameters cannot be None, without a seed."
            np.random.seed(seed) # Set the seed, for reproducibility
            parameters = 2 * np.pi * np.random.rand(self.n_edges + self.n_nodes, n_layers, requires_grad=True) if diff_Rx else 2 * np.pi * np.random.rand(self.n_edges + 1, n_layers, requires_grad=True)
        else:
            # This is to ensure that the 'parameters' array, when provided, has the correct shape.
            assert(parameters.shape == (self.n_edges + self.n_nodes, n_layers) if diff_Rx else (self.n_edges + 1, n_layers)), "parameters has the wrong shape."

        # Define the number of qubits
        n_qubits = self.n_nodes

        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)

        # Preliminary definitions: circuit setup
        # Unitary operator 'U_C' (Cost/Problem) with parameters 'gamma'.
        def U_C(gamma):
            for i, edge in enumerate(self.graph):
                wire1, wire2 = edge[0], edge[1]
                # Usual terms
                qml.CNOT(wires = [wire1, wire2])
                qml.RZ(gamma[i], wires = wire2)
                qml.CNOT(wires = [wire1, wire2])

        # Unitary operator 'U_B' (Bias) with parameters 'beta'.
        def U_B(beta):
            for wire in range(n_qubits):
                if(diff_Rx): qml.RX(2 * beta[wire], wires=wire)
                else: qml.RX(2 * beta[0], wires=wire)

        # QAOA-like circuit
        @qml.qnode(dev)
        def circuit(params):
            gammas = params[:-self.n_nodes] if diff_Rx else params[:-1]; betas = params[-self.n_nodes:] if diff_Rx else params[-1:]
            # Apply Hadamards to get the n qubit |+> state.
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)

            # Apply p instances (layers) of unitary operators.
            for i in range(n_layers):
                U_C(gammas[:, i])
                U_B(betas[:, i])

            return qml.probs()

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_layers = {n_layers}.')
            fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters)
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        print(f"iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        def compute_nodes_probs(probabilities, basis_states_lists):
            """
            Computes the probabilities associated with each node.

            Args:
                probabilities (list): The probabilities of each basis state.
                basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.

            Returns:
                list: The probabilities associated with each node.
            """

            nodes_probs = []
            for sublist in basis_states_lists:
                node_prob = 0
                for basis_state in sublist: node_prob += probabilities[int(basis_state, 2)]
                nodes_probs.append(node_prob)
            nodes_probs = [prob/sum(nodes_probs) for prob in nodes_probs]
            return nodes_probs
        
        # Define the cost function
        def objective(params):
            """
            This function computes the value of the cost function.

            Args:
                params (array): Array of parameters.

            Returns:
                float: The value of the cost function.
            """

            # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
            probs = circuit(params); cost  = 0
            nodes_probs = compute_nodes_probs(probs, basis_states_lists)

            # Second - Compute the cost function itself
            for edge in self.graph:
                # j and k are the nodes connected by the edge
                # 0: j, 1: k
                d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                cost += edge_cost
            return cost
        
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end='\r')
        while(True):
            parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            if (MaxCut is not None):
                approx_ratio = 0
                probabilities = circuit(parameters)
                nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
                partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
            
            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # abs_tol and rel_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        probabilities = circuit(parameters)
        nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
        partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

        # Only 'meaningful'/'readable' for small graphs.
        if(raw_plot):
            basis_states = [format(i, '0'+str(n_qubits)+'b') for i in range(2**n_qubits)]
            plt.figure(figsize=(8, 6))
            plt.bar(basis_states, probabilities)
            plt.xlabel("Basis States")
            plt.ylabel("Probabilities")
            plt.title("Probabilities vs. Basis States")
            plt.show()

            # Print most frequently sampled bitstring
            most_freq_bit_string = np.argmax(probabilities)
            print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

        if(nodes_probs_plot):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(nodes_probs); x = range(num_outcomes); threshold = 1 / (2*B)
            plt.hlines(threshold, -1, num_outcomes, colors='C2', linestyles='dashed',
                       label=r'$\frac{1}{2B} =$' + f'{threshold:.3f} (B = {int(B)})')
            plt.xlim(-1, num_outcomes); plt.xticks(x)

            # Plot the probability distribution
            plt.bar(x, nodes_probs, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers})')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        return probabilities, nodes_probs, cost, cost_vec, ar_vec, parameters, partition, train_time

    def Exponential_QEMC_iQAQE_MaxCut(self, diff_Rx = False, n_layers = None, device = 'default.qubit', 
                                      draw_circuit = False, B = None, basis_states_lists = None, seed = None,
                                      rel_tol = 0, abs_tol = 0, MaxCut = None, max_iter = 100, shots = None,
                                      parameters = None, cost_plot = True, raw_plot = False, nodes_probs_plot = True, **kwargs):
        """
        Implements the QEMC limit of Bence's suggested problem-inspired ansatz, for the MaxCut problem.
        Uses an exponential number of parameters (number of parameters = number of graph nodes).

        Args:
            diff_Rx (bool, optional): Whether to use different Rx parameters or not. Defaults to False.
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            B (int, optional): The number of "blue" nodes. Defaults to None.
            basis_states_lists (list[list]): The basis states assigned to each graph node. Defaults to None.
            seed (int, optional): The seed to be used. Defaults to None.
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            raw_plot (bool, optional): Whether to plot the raw probabilities or not. Defaults to False.
            nodes_probs_plot (bool, optional): Whether to plot the nodes' probabilities or not. Defaults to True.

        Returns:
            list: The probabilities of each basis state.
            list: The probabilities associated with each node.
            float: The "optimal" cost function value.
            list: The cost function values for each iteration.
            list: The approximation ratio values for each iteration.
            list: The "optimal" parameters.
            list: The "optimal" partition.
            float: The training time.
        """

        # Start with the iQAQE method, and then modify it to include the QEMC limit of Bence's suggested problem-inspired ansatz.

        assert(n_layers is not None and B is not None and basis_states_lists is not None), "n_layers, B and basis_states_lists cannot be None."
        assert(len(basis_states_lists) == self.n_nodes), "The number of basis states lists must be equal to the graph's number of nodes."
        
        # Define the number of qubits
        n_qubits = math.ceil(np.log2(self.n_nodes))

        # Define the 'parameters' array, if it is None
        if(parameters is None):
            assert(seed is not None), "parameters cannot be None, without a seed."
            np.random.seed(seed) # Set the seed, for reproducibility
            parameters = 2 * np.pi * np.random.rand(2**n_qubits - 1 + n_qubits, n_layers, requires_grad=True) if diff_Rx else 2 * np.pi * np.random.rand(2**n_qubits - 1 + 1, n_layers, requires_grad=True)
        else:
            # This is to ensure that the 'parameters' array, when provided, has the correct shape.
            assert(parameters.shape == (2**n_qubits - 1 + n_qubits, n_layers) if diff_Rx else (2**n_qubits - 1 + 1, n_layers)), "parameters has the wrong shape."

        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)

        # Preliminary definitions: circuit setup
        # Unitary operator 'U_C' (Cost/Problem) with parameters 'gamma'.
        def U_C(gamma):
            c = 0
            for i in range(1, n_qubits + 1):
                for j in itertools.combinations(range(n_qubits), i):
                    qml.MultiRZ(gamma[c], wires=j); c += 1

        # Unitary operator 'U_B' (Bias) with parameters 'beta'.
        def U_B(beta):
            for wire in range(n_qubits):
                if(diff_Rx): qml.RX(2 * beta[wire], wires=wire)
                else: qml.RX(2 * beta[0], wires=wire)

        # QAOA-like circuit
        @qml.qnode(dev)
        def circuit(params):
            s = 2**n_qubits - 1
            # Apply Hadamards to get the n qubit |+> state.
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)

            # Apply p instances (layers) of unitary operators.
            for i in range(n_layers):
                U_C(params[:s, i])
                U_B(params[s:, i])

            return qml.probs()

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_layers = {n_layers}.')
            fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters)
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        print(f"iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        def compute_nodes_probs(probabilities, basis_states_lists):
            """
            Computes the probabilities associated with each node.

            Args:
                probabilities (list): The probabilities of each basis state.
                basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.

            Returns:
                list: The probabilities associated with each node.
            """

            nodes_probs = []
            for sublist in basis_states_lists:
                node_prob = 0
                for basis_state in sublist: node_prob += probabilities[int(basis_state, 2)]
                nodes_probs.append(node_prob)
            nodes_probs = [prob/sum(nodes_probs) for prob in nodes_probs]
            return nodes_probs
        
        # Define the cost function
        def objective(params):
            """
            This function computes the value of the cost function.

            Args:
                params (array): Array of parameters.

            Returns:
                float: The value of the cost function.
            """

            # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
            probs = circuit(params); cost  = 0
            nodes_probs = compute_nodes_probs(probs, basis_states_lists)
            # You can also grab the values from an ArrayBox by using its “_value” attribute.
            # Example usage, to retrive the numerical values of 'nodes_probs': [Box._value for Box in nodes_probs]

            # Second - Compute the cost function itself
            for edge in self.graph:
                # j and k are the nodes connected by the edge
                # 0: j, 1: k
                d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                cost += edge_cost
            return cost
        
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end='\r')
        while(True):
            parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            if (MaxCut is not None):
                approx_ratio = 0
                probabilities = circuit(parameters)
                nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
                partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
            
            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # abs_tol and rel_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        probabilities = circuit(parameters)
        nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
        partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

        # Only 'meaningful'/'readable' for small graphs.
        if(raw_plot):
            basis_states = [format(i, '0'+str(n_qubits)+'b') for i in range(2**n_qubits)]
            plt.figure(figsize=(8, 6))
            plt.bar(basis_states, probabilities)
            plt.xlabel("Basis States")
            plt.ylabel("Probabilities")
            plt.title("Probabilities vs. Basis States")
            plt.show()

            # Print most frequently sampled bitstring
            most_freq_bit_string = np.argmax(probabilities)
            print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

        if(nodes_probs_plot):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(nodes_probs); x = range(num_outcomes); threshold = 1 / (2*B)
            plt.hlines(threshold, -1, num_outcomes, colors='C2', linestyles='dashed',
                       label=r'$\frac{1}{2B} =$' + f'{threshold:.3f} (B = {int(B)})')
            plt.xlim(-1, num_outcomes); plt.xticks(x)

            # Plot the probability distribution
            plt.bar(x, nodes_probs, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers})')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        return probabilities, nodes_probs, cost, cost_vec, ar_vec, parameters, partition, train_time

    def Goemans_Williamson(self, MaxCut = None):
        """
        Computes the Goemans-Williamson partition for the MaxCut problem.
        Uses a semi-definite programming relaxation.

        Returns:
            list: The partition.
            int: The cut value.
            float: The obtained approximation ratio, if 'MaxCut' is given.
            float: The running time.
        """

        # Definitions:
        n = self.n_nodes; edges = self.graph

        # Semi-definite programming relaxation + Constraints
        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [X[i, i] == 1 for i in range(n)]

        # Objective function and solving
        objective = sum(0.5 * (1 - X[i, j]) for (i, j) in edges)
        prob = cp.Problem(cp.Maximize(objective), constraints)

        # Solve the problem
        start_time = time.time()
        prob.solve()
        run_time = time.time() - start_time # Time taken for running
        print("--- Goemans-Williamson algorithm took %s seconds to run. ---" % (run_time))

        # Retrieve x, and random-plane projection - to obtain the sign.
        x = sqrtm(X.value)
        u = np.random.randn(n)
        partition = np.sign(x @ u)

        # Compute the cut value
        cut = self.compute_cut(partition)

        # Get the approximation ratio, if 'MaxCut' is given
        if (MaxCut is not None): approx_ratio = cut/MaxCut
        else: approx_ratio = None
            
        # Return the partition
        return partition, cut, approx_ratio, run_time
    
    # Parity-like QAOA inspired. Built on the iQAQE Framework, to be more problem-inspired.
    def Parity_Inspired_MaxCut(self, k = None, n_layers = None, device = 'default.qubit', shots = None,
                               parameters = None, ansatz_type = None, cost_type = None, draw_circuit = False,
                               B = None, basis_states_lists = None, rel_tol = 0, abs_tol = 0, 
                               MaxCut = None, max_iter = 100, seed = 0, verbose = False, rs = None,
                               cost_plot = True, raw_plot = False, nodes_probs_plot = True, **kwargs):
        """
        Implements the Parity-inspired VQA, for the MaxCut problem, using the iQAQE Framework. Doesn't support ND-CNOTs, yet, nor the 'Brickwork' circuit ansatz.

        Args:
            k (int, optional): Order of polynomial compression. Defaults to None.
            n_layers (int, optional): The number of layers to be used. Defaults to None.
            device (str, optional): The device to be used. Defaults to 'default.qubit'.
            shots (int, optional): The number of shots to be used. Defaults to None (Analytical calculations).
            parameters (list, optional): The initial parameters to be used. Defaults to None.
            ansatz_type (str, optional): The type of ansatz to be used. Defaults to None.
            cost_type (str, optional): The type of cost function to be used. Defaults to None.
            draw_circuit (bool, optional): Whether to draw the circuit or not. Defaults to False.
            B (int, optional): The number of "blue" nodes. Defaults to None.
            basis_states_lists (list[list]): The basis states assigned to each graph node. Defaults to None.
            rel_tol (int, optional): The relative tolerance to be used for the convergence criteria. Defaults to 0.
            abs_tol (int, optional): The absolute tolerance to be used for the convergence criteria. Defaults to 0.
            MaxCut (int, optional): The maximum cut value. Defaults to None.
            max_iter (int, optional): The maximum number of iterations to be used. Defaults to 100.
            seed (int, optional): The seed to be used. Defaults to 0.
            verbose (bool, optional): Whether to print the results or not. Defaults to False.
            rs (list, optional): The 'rs' values to be used. Defaults to None.
            cost_plot (bool, optional): Whether to plot the cost function or not. Defaults to True.
            raw_plot (bool, optional): Whether to plot the raw probabilities or not. Defaults to False.
            nodes_probs_plot (bool, optional): Whether to plot the nodes' probabilities or not. Defaults to True.
        """

        assert(k == 2 or k == 3), "The order of polynomial compression 'k' must be 2 or 3. [Higher orders are not supported yet.]"
        assert(ansatz_type == 'SEL' or ansatz_type == 'Brickwork' or ansatz_type == 'Parity'), "The 'ansatz_type' must be 'SEL', 'Brickwork' or 'Parity'."
        assert(cost_type == 'QEMC' or cost_type == 'Expectation'), "The 'cost_type' must be 'QEMC' or 'Expectation'."
        assert(k is not None and n_layers is not None), "k and n_layers cannot be None."
        if(cost_type == 'QEMC'):
            assert(B is not None and basis_states_lists is not None), "B and basis_states_lists cannot be None. [For QEMC cost type.]"
            assert(len(basis_states_lists) == self.n_nodes), "The number of basis states lists must be equal to the graph's number of nodes. [For QEMC cost type.]"
        
        def combinations(n, k):
            return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))
        
        if(k == 2):
            # Quadradic compression
            n_qubits = math.ceil((1 + np.sqrt(1 + 8*self.n_nodes))/2)
            if(verbose): print(f'Combinations of {n_qubits} taken 2 at a time (Quadratic compression): {combinations(n_qubits, 2)} >= {self.n_nodes}.')
        else:
            # Cubic compression
            n_qubits = math.ceil(optimize.newton(lambda n: n * (n - 1) * (n - 2) - 6 * self.n_nodes, self.n_nodes))
            if(verbose): print(f'Combinations of {n_qubits} taken 3 at a time (Cubic compression): {combinations(n_qubits, 3)} >= {self.n_nodes}.')
                    
        # Function to generate the encoding
        def get_encoding(n_qubits, k):
            encoding = list(itertools.combinations(range(n_qubits), k))
            return encoding

        # Get the encoding
        encoding = get_encoding(n_qubits, k)[:self.n_nodes]
        if(verbose): print(f'Encoding: {encoding}.')
        
        # Original terms:
        terms = []
        for i, j in self.graph:
            terms.append(list(set(encoding[i]).union(encoding[j])))
        if(verbose): print(f'terms: {terms}.')

        # Removing duplicates # 'consecutive' duplicates
        # This method preserves the order of the elements.
        clean_terms = [ele for ind, ele in enumerate(terms) if ele not in terms[:ind]] # [value for i, value in enumerate(terms) if i == 0 or value != terms[i-1]]
        if(verbose): print(f'clean_terms: {clean_terms}.')

        if(ansatz_type == 'Parity'):
            if(parameters is not None):
                assert(parameters[0].shape == (n_layers, len(clean_terms))), "'parameters[0]': shape must be (n_layers,len(clean_terms))."
                assert(parameters[1].shape == (n_layers, n_qubits)), "'parameters[1]': shape must be (n_layers, n_qubits)."
            else:
                np.random.seed(seed)
                parameters = [2 * np.pi * np.random.rand(n_layers, len(clean_terms), requires_grad=True), 2 * np.pi * np.random.rand(n_layers, n_qubits, requires_grad=True)]
        elif(ansatz_type == 'SEL'):
            # Default 'rs' values
            if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]
            if(parameters is not None):
                assert(parameters.shape == (n_layers, n_qubits, 3)), "'parameters': shape must be (n_layers, n_qubits, 3)."
            else:
                np.random.seed(seed)
                parameters = 2 * np.pi * np.random.rand(n_layers, n_qubits, 3, requires_grad=True)
        elif(ansatz_type == 'Brickwork'):
            # To do: Implement the 'Brickwork' circuit ansatz.
            if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]
            if(parameters is not None):
                assert(parameters.shape == (n_layers, n_qubits, 4)), "'parameters': shape must be (n_layers, n_qubits, 4)."
            else:
                np.random.seed(seed)
                parameters = 2 * np.pi * np.random.rand(n_layers, n_qubits, 4, requires_grad=True)

        # Device setup
        dev = qml.device(device, wires = n_qubits, shots = shots)
        
        def parity_layer(gammas, betas):
            for i, tpl in enumerate(clean_terms): qml.MultiRZ(gammas[i], wires=tpl)
            for j in range(n_qubits): qml.RX(betas[j], wires=j)
        
        if(ansatz_type == 'SEL'):
            @qml.qnode(dev)
            def circuit(params, probs_flag = False):
                # Apply Hadamards to get the n qubit |+> state.
                for wire in range(n_qubits): qml.Hadamard(wires=wire)

                # Apply p instances (layers) of unitary operators.
                global c; c = 0
                # iQAQE layer definition:
                def iQAQE_layer(params_layer):
                    global c
                    for i in range(n_qubits):
                        qml.RX(params_layer[i][0], wires = i); qml.RY(params_layer[i][1], wires = i); qml.RZ(params_layer[i][2], wires = i)
                    for j in range(n_qubits):
                        qml.CNOT(wires = [j, (j+rs[c]) % n_qubits])
                    c += 1

                # Actual "Strongly Entangling Layers" ansatz:
                qml.layer(iQAQE_layer, n_layers, params)
                if(cost_type == 'QEMC'): # 'QEMC' cost is to be understood as iQAQE-like!
                    return qml.probs()
                elif(cost_type == 'Expectation'):
                    if(probs_flag):
                        return qml.probs()
                    else:
                        # To do: Implement the 'Expectation' cost type.
                        P_str_lst = []
                        for P_str in encoding:
                            P = qml.PauliZ(P_str[0])
                            for i in range(1, len(P_str)):
                                P = P @ qml.PauliZ(P_str[i])
                            P_str_lst.append(P)
                        return [qml.expval(P) for P in P_str_lst]
        elif(ansatz_type == 'Brickwork'):
            @qml.qnode(dev)
            def circuit(params, probs_flag = False):
                # To do: Implement the 'Brickwork' circuit ansatz.
                # Apply Hadamards to get the n qubit |+> state.
                for wire in range(n_qubits): qml.Hadamard(wires=wire)

                # Apply p instances (layers) of unitary operators.
                global c; c = 0
                # iQAQE layer definition:
                def Brickwork_layer(params_layer):
                    global c
                    for i in range(n_qubits):
                        qml.RX(params_layer[i][0], wires = i); qml.RY(params_layer[i][1], wires = i); qml.RZ    (params_layer[i][2], wires = i)
                    for j in range(n_qubits):
                        # The 'wires' here were chosen somewhat 'arbitrarily'. This is because not much information is given about the 'brickwork' ansatz (https://arxiv.org/pdf/2401.09421v2).
                        RXX(params_layer[j][3], wires = [(2*j + rs[c] - 1) % n_qubits, (2*j + rs[c]) % n_qubits])
                    c += 1

                # Actual "Strongly Entangling Layers" ansatz:
                qml.layer(Brickwork_layer, n_layers, params)
                if(cost_type == 'QEMC'):
                    return qml.probs()
                elif(cost_type == 'Expectation'):
                    if(probs_flag):
                        return qml.probs()
                    else:
                        # To do: Implement the 'Expectation' cost type.
                        P_str_lst = []
                        for P_str in encoding:
                            P = qml.PauliZ(P_str[0])
                            for i in range(1, len(P_str)):
                                P = P @ qml.PauliZ(P_str[i])
                            P_str_lst.append(P)
                        return [qml.expval(P) for P in P_str_lst]
        elif(ansatz_type == 'Parity'):
            @qml.qnode(dev)
            def circuit(gammas, betas, probs_flag = False):
                # Apply Hadamards to get the n qubit |+> state.
                for wire in range(n_qubits): qml.Hadamard(wires=wire)
                for n in range(n_layers): parity_layer(gammas[n], betas[n])
                if(cost_type == 'QEMC'):
                    return qml.probs()
                elif(cost_type == 'Expectation'):
                    if(probs_flag):
                        return qml.probs()
                    else:
                        # To do: Implement the 'Expectation' cost type.
                        P_str_lst = []
                        for P_str in encoding:
                            P = qml.PauliZ(P_str[0])
                            for i in range(1, len(P_str)):
                                P = P @ qml.PauliZ(P_str[i])
                            P_str_lst.append(P)
                        return [qml.expval(P) for P in P_str_lst]

        if draw_circuit:
            qml.drawer.use_style("pennylane") # Set the default style
            print(f'Quantum circuit drawing: n_qubits = {n_qubits}, n_layers = {n_layers}, ansatz_type = {ansatz_type}.')
            if(ansatz_type == 'SEL'):
                fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters)
            elif(ansatz_type == 'Parity'):
                fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters[0], parameters[1])
            elif(ansatz_type == 'Brickwork'):
                # To do: Implement the 'Brickwork' circuit ansatz.
                fig, ax = qml.draw_mpl(circuit, decimals=3)(parameters)
            plt.show()
            
        # Status message, before optimization
        step_size = kwargs.get('step_size', 0.99)
        # cte = kwargs.get('cte', 1.0); print(f"Softmax scaling constant: {cte}.")
        print(f"iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

        # # Testing using the 'softmax' function, instead of mere normalization.
        # def softmax(x, cte = 1.0):
        #     """Compute softmax values for each sets of scores in x."""
        #     return np.exp(cte * x) / np.sum(np.exp(cte * x), axis=0)

        def compute_nodes_probs(probabilities, basis_states_lists):
            """
            Computes the probabilities associated with each node.

            Args:
                probabilities (list): The probabilities of each basis state.
                basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.

            Returns:
                list: The probabilities associated with each node.
            """

            nodes_probs = []
            for sublist in basis_states_lists:
                node_prob = 0
                for basis_state in sublist: node_prob += probabilities[int(basis_state, 2)]
                nodes_probs.append(node_prob)
            nodes_probs = [prob/sum(nodes_probs) for prob in nodes_probs]
            # nodes_probs = softmax(np.array(nodes_probs), cte)
            return nodes_probs
        
        # For computing the Poljak-Turzik lower bound (Regularization term, when using the 'Expectation' cost type)
        def poljak_turzik_lower_bound_unweighted():
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

        # Define the cost function
        if(ansatz_type == 'Parity'):
            def objective(gammas, betas):
                """
                This function computes the value of the cost function.

                Args:
                    params (array): Array of parameters.

                Returns:
                    float: The value of the cost function.
                """
                if(cost_type == 'QEMC'): # This can be simplified! The structure is a bit too ugly.
                    # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
                    probs = circuit(gammas, betas); cost  = 0
                    nodes_probs = compute_nodes_probs(probs, basis_states_lists)

                    # Second - Compute the cost function itself
                    for edge in self.graph:
                        # j and k are the nodes connected by the edge
                        # 0: j, 1: k
                        d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                        edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                        cost += edge_cost
                    return cost
                elif(cost_type == 'Expectation'):
                    # To do: Implement the 'Expectation' cost type.
                    # Objective for the MaxCut problem
                    exp_vals = circuit(gammas, betas)

                    # First part of the cost function
                    avg_H = 0; alpha = n_qubits**(np.floor(k/2))
                    for i, j in self.graph:
                        avg_H += ( np.tanh(alpha * exp_vals[i]) * np.tanh(alpha * exp_vals[j]) ) # Missing the non-linearity here!

                    # Regularization term
                    beta = 0.5
                    nu = poljak_turzik_lower_bound_unweighted()
                    L_reg = beta * nu * (1/self.n_nodes * np.sum([np.tanh(alpha * exp_val)**2 for exp_val in exp_vals]))**2

                    # Total cost
                    cost = avg_H + L_reg

                    return cost
        elif(ansatz_type == 'SEL'):
            def objective(params):
                """
                This function computes the value of the cost function.

                Args:
                    params (array): Array of parameters.

                Returns:
                    float: The value of the cost function.
                """
                if(cost_type == 'QEMC'):
                    # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
                    probs = circuit(params); cost  = 0
                    nodes_probs = compute_nodes_probs(probs, basis_states_lists)

                    # Second - Compute the cost function itself
                    for edge in self.graph:
                        # j and k are the nodes connected by the edge
                        # 0: j, 1: k
                        d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                        edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                        cost += edge_cost
                    return cost
                elif(cost_type == 'Expectation'):
                    # To do: Implement the 'Expectation' cost type.
                    # Objective for the MaxCut problem
                    exp_vals = circuit(params)
                    
                    # First part of the cost function
                    avg_H = 0; alpha = n_qubits**(np.floor(k/2))
                    for i, j in self.graph:
                        avg_H += ( np.tanh(alpha * exp_vals[i]) * np.tanh(alpha * exp_vals[j]) ) # Missing the non-linearity here!

                    # Regularization term
                    beta = 0.5
                    nu = poljak_turzik_lower_bound_unweighted()
                    L_reg = beta * nu * (1/self.n_nodes * np.sum([np.tanh(alpha * exp_val)**2 for exp_val in exp_vals]))**2

                    # Total cost
                    cost = avg_H + L_reg

                    return cost
        elif(ansatz_type == 'Brickwork'):
            # To do: Implement the 'Brickwork' circuit ansatz.
            def objective(params):
                """
                This function computes the value of the cost function.

                Args:
                    params (array): Array of parameters.

                Returns:
                    float: The value of the cost function.
                """
                if(cost_type == 'QEMC'):
                    # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
                    probs = circuit(params); cost  = 0
                    nodes_probs = compute_nodes_probs(probs, basis_states_lists)

                    # Second - Compute the cost function itself
                    for edge in self.graph:
                        # j and k are the nodes connected by the edge
                        # 0: j, 1: k
                        d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                        edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                        cost += edge_cost
                    return cost
                elif(cost_type == 'Expectation'):
                    # To do: Implement the 'Expectation' cost type.
                    # Objective for the MaxCut problem
                    exp_vals = circuit(params)
                    
                    # First part of the cost function
                    avg_H = 0; alpha = n_qubits**(np.floor(k/2))
                    for i, j in self.graph:
                        avg_H += ( np.tanh(alpha * exp_vals[i]) * np.tanh(alpha * exp_vals[j]) ) # Missing the non-linearity here!

                    # Regularization term
                    beta = 0.5
                    nu = poljak_turzik_lower_bound_unweighted()
                    L_reg = beta * nu * (1/self.n_nodes * np.sum([np.tanh(alpha * exp_val)**2 for exp_val in exp_vals]))**2

                    # Total cost
                    cost = avg_H + L_reg

                    return cost        
        
        # Initialize optimizer: Adagrad works well empirically. We use Adam, though.
        opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

        # Optimize parameters in objective
        start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
        print("\nOptimizing parameters...", end='\r')
        while(True):
            if(ansatz_type == 'Parity'):
                parameters, cost = opt.step_and_cost(objective, parameters[0], parameters[1]); i += 1; cost_vec.append(cost)
            elif(ansatz_type == 'SEL'):
                parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
            elif(ansatz_type == 'Brickwork'):
                # To do: Implement the 'Brickwork' circuit ansatz.
                parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)

            if (MaxCut is not None):
                # Compute the approximation ratio, from the computed partition's cut
                approx_ratio = 0

                # Updated!
                if(cost_type == 'Expectation'):
                    # Re-defining this, so it works for arbitrary k-value
                    def get_parity_partition(Exp_vals: list) -> str:
                        partition = []
                        for n in range(len(Exp_vals)):
                            n_k = np.sign(Exp_vals[n]); # print(f"n_k: {n_k}. Exp_vals[n] = {Exp_vals[n]}.") # Debugging.
                            if(n_k == 1):    partition.append(0)
                            elif(n_k == -1): partition.append(1)
                            else:            partition.append(1) # Default to 1. Bugfix [Sometimes, n_k = 0.]
                            # print(f"Partition (*inside*): {partition}.") # Debugging.
                        return "".join([str(p) for p in partition])
                    if(ansatz_type == 'Parity'):
                        probabilities = circuit(parameters[0], parameters[1], probs_flag = True)
                        exp_vals = circuit(parameters[0], parameters[1])
                    elif(ansatz_type == 'SEL'):
                        probabilities = circuit(parameters, probs_flag = True)
                        exp_vals = circuit(parameters)
                    elif(ansatz_type == 'Brickwork'):
                        # To do: Implement the 'Brickwork' circuit ansatz.
                        probabilities = circuit(parameters, probs_flag = True)
                        exp_vals = circuit(parameters)
                    nodes_probs = None # Relevant only for the 'QEMC' cost type
                    # 'exp_vals' is already computed!
                    partition = get_parity_partition(exp_vals)

                elif(cost_type == 'QEMC'):
                    if(ansatz_type == 'Parity'):
                        probabilities = circuit(parameters[0], parameters[1], probs_flag = True)
                    elif(ansatz_type == 'SEL'):
                        probabilities = circuit(parameters)
                    elif(ansatz_type == 'Brickwork'):
                        # To do: Implement the 'Brickwork' circuit ansatz.
                        probabilities = circuit(parameters)

                    # Computing the probaility associated to each node, from the 'n_qubit' probability distribution
                    nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
                    exp_vals = None # Relevant only for the 'Expectation' cost type
                    # Get the computed partition
                    partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

                # Compute the approximation ratio, from the computed partition's cut
                # print(f'Partition: {partition}.') # Debugging.
                cut = self.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
            
            if i % 5 == 0:
                print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
            
            # Check convergence criteria. This is structured to be as efficient as possible in the case where both
            # abs_tol and rel_tol are 0. "Legacy" reasons.
            if(abs_tol != 0):
                if (i - 1) >= 1:
                    abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
                    if (abs_diff <= abs_tol):
                        print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                        break
            if(rel_tol != 0):
                if (i - 1) >= 1:
                    rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
                    if (rel_diff <= rel_tol):
                        print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                        break

            # Check maximum iterations
            if i == max_iter:
                print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
                break
        train_time = time.time() - start_time # Time taken for training
        print("--- Training took %s seconds ---" % (train_time))

        if cost_plot:
            # Plotting the cost function
            plt.figure(figsize=(8, 6))
            plt.plot(cost_vec, label="Cost function")
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Cost function")
            plt.title("Cost function evolution")
            plt.show()

        # Just take the exact distribution
        if(cost_type == 'Expectation'):
            # Re-defining this, so it works for arbitrary k-value
            def get_parity_partition(Exp_vals: list) -> str:
                partition = []
                for n in range(len(Exp_vals)):
                    n_k = np.sign(Exp_vals[n])
                    if(n_k == 1):    partition.append(0)
                    elif(n_k == -1): partition.append(1)
                return "".join([str(p) for p in partition])
            if(ansatz_type == 'Parity'):
                probabilities = circuit(parameters[0], parameters[1], probs_flag = True)
                exp_vals = circuit(parameters[0], parameters[1])
            elif(ansatz_type == 'SEL'):
                probabilities = circuit(parameters, probs_flag = True)
                exp_vals = circuit(parameters)
            elif(ansatz_type == 'Brickwork'):
                # To do: Implement the 'Brickwork' circuit ansatz.
                probabilities = circuit(parameters, probs_flag = True)
                exp_vals = circuit(parameters)
            nodes_probs = None # Relevant only for the 'QEMC' cost type
            # 'exp_vals' is already computed!
            partition = get_parity_partition(exp_vals)

        elif(cost_type == 'QEMC'):
            if(ansatz_type == 'Parity'):
                probabilities = circuit(parameters[0], parameters[1], probs_flag = True)
            elif(ansatz_type == 'SEL'):
                probabilities = circuit(parameters)
            elif(ansatz_type == 'Brickwork'):
                # To do: Implement the 'Brickwork' circuit ansatz.
                probabilities = circuit(parameters)

            # Computing the probaility associated to each node, from the 'n_qubit' probability distribution
            nodes_probs = compute_nodes_probs(probabilities, basis_states_lists)
            exp_vals = None # Relevant only for the 'Expectation' cost type
            # Get the computed partition
            partition = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

        # Only 'meaningful'/'readable' for small graphs.
        if(raw_plot):
            # Get the basis states
            basis_states = [format(i, '0'+str(n_qubits)+'b') for i in range(2**n_qubits)]
    
            # Plot the probabilities
            plt.figure(figsize=(8, 6))
            plt.bar(basis_states, probabilities)
            plt.xlabel("Basis States")
            plt.xticks(rotation=90)
            plt.ylabel("Probabilities")
            plt.title("Probabilities vs. Basis States")
            plt.show()

            # Print most frequently sampled bitstring
            most_freq_bit_string = np.argmax(probabilities)
            print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

        if(nodes_probs_plot and cost_type == 'QEMC'):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(nodes_probs); x = range(num_outcomes); threshold = 1 / (2*B)
            plt.hlines(threshold, -1, num_outcomes, colors='C2', linestyles='dashed',
                       label=r'$\frac{1}{2B} =$' + f'{threshold:.3f} (B = {int(B)})')
            plt.xlim(-1, num_outcomes); plt.xticks(x)

            # Plot the probability distribution
            plt.bar(x, nodes_probs, label='Probability', color = 'C1')
            plt.xlabel('Outcome')
            plt.ylabel('Probability')
            plt.title(f'Discrete Probability Distribution (n_layers={n_layers})')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()
        elif(nodes_probs_plot and cost_type == 'Expectation'):
            # Definitions (for plotting)
            plt.figure(figsize=(8, 6))
            num_outcomes = len(exp_vals); x = range(num_outcomes)

            # Sort the probabilities and the outcomes, to plot only the 10 most probable outcomes
            plt.xticks(fontsize = 8, rotation = 90)

            # Plot the probability distribution
            plt.bar(x, exp_vals, label='Probability', color = 'C1'); plt.xticks(x)
            plt.xlabel('Node number')
            plt.ylabel(r'Associated $\left<\Pi_i\right>$')
            plt.title(f'Nodes' + r' $\left<\Pi_i\right>$' + ' distribution')
            plt.legend(loc = 'best', frameon = True, fancybox=True)
            plt.show()

        # Maybe I could make it so that we specify the number of parameters that we want, not the number of layers.
        # This way, we could have a more flexible approach to the number of parameters.

        return probabilities, nodes_probs, exp_vals, cost, cost_vec, ar_vec, parameters, partition, train_time
    
    def Avg_best_so_far(self, VQA = None, n_qubits = None, n_layers = None, device = 'default.qubit',
                        shots = None, rs = None, B = None, basis_states_lists = None, MaxCut = None,
                        non_deterministic_CNOT = False, max_iter = 100, plot_flag = True, k = None,
                        params_silent = True, repeat = 10, method = 'Avg_Bsf', diff_Rx = False, **kwargs):
        """
        Computes the average of the best-so-far results, for a given VQA.

        Args:
            VQA (str): The VQA to be used. Defaults to None.
            n_qubits (int): The number of qubits to be used. Defaults to None.
            n_layers (int): The number of layers to be used. Defaults to None.
            device (str): The device to be used. Defaults to 'default.qubit'.
            shots (int): The number of shots to be used. Defaults to None.
            rs (list): The random seeds to be used. Defaults to None.
            B (int): The number of "blue" nodes. Defaults to None.
            basis_states_lists (list[list]): The basis states assigned to each graph node. Defaults to None.
            MaxCut (int): The maximum cut value. Defaults to None.
            non_deterministic_CNOT (bool): Whether to use non-deterministic CNOT gates or not. Defaults to False.
            max_iter (int): The maximum number of iterations to be used. Defaults to 100.
            plot_flag (bool): Whether to plot the results or not. Defaults to True.
            k (int): The order of polynomial compression. Defaults to None. ('Parity_Inspired' only VQA.)
            params_silent (bool): Whether to print the parameters or not. Defaults to True.
            repeat (int): The number of times to repeat the process. Defaults to 10.
            method (str): The method to be used. Defaults to 'Avg_Bsf'. (Used only for plotting!)
            diff_Rx (bool): Whether to use different Rx parameters or not. Defaults to False. (Used in Exponential_QEMC_iQAQE.)
            **kwargs: Additional arguments.

        Raises:
            Exception: VQA, n_layers and MaxCut cannot be None.
            Exception: n_qubits, rs, B and basis_states_lists cannot be None (iQAQE).
            Exception: B cannot be None (QEMC).
            Exception: B cannot be None (Exponential_QEMC_iQAQE).

        Returns:
            ar_vec: The approximation ratios.
            med_bsf_vec: The median of the best-so-far values.
            avg_bsf_vec: The average of the best-so-far values.
            bsf_avg_vec: The best-so-far of the average values.
            avg_vec: The average values.
            std_vec: The standard deviation values.
            avg_cost_vec: The average cost values.
            min_ar_vec: The minimum approximation ratios.
            max_ar_vec: The maximum approximation ratios.
            avg_train_time: The average training time.
        """

        # Error handling
        if(VQA == 'G-W'): # Exception for the Goemans-Williamson algorithm, since it's not a VQA!
            assert(MaxCut is not None), "MaxCut cannot be None (Goemans-Williamson)."
            plot_flag = False
        else:
            assert(VQA is not None and n_layers is not None and MaxCut is not None), "VQA, n_layers and MaxCut cannot be None."
            if((VQA == 'iQAQE' or VQA == 'iQAQE_QAOA_Ansatz') and (n_qubits is None and B is None or basis_states_lists is None)): raise Exception("n_qubits, rs, B and basis_states_lists cannot be None (iQAQE).")
            if(VQA == 'QEMC' and (B is None)): raise Exception("B cannot be None (QEMC).")
            if(VQA == 'Exponential_QEMC_iQAQE' and (B is None)): raise Exception("B cannot be None (Exponential_QEMC_iQAQE).")
            if(VQA == 'Parity_Inspired' and (B is None) and (k is None)): raise Exception("B cannot be None (Parity_Inspired).")
        # Default 'step_size' value
        # step_size = kwargs.get('step_size', 0.99) # This is supposed to enter as part of **kwargs.
        
        # Averaged over 'repeat' (defaults to 10) runs
        ar_vec, avg_train_time = [], 0
        cost_vec = [] if VQA != 'G-W' else None
        for i_ in range(repeat):
            print(f"--- Run #{i_+1} of {repeat}: ---")
            if(VQA == 'G-W'):
                _, _, approx_ratio, temp_train_time = self.Goemans_Williamson(MaxCut = MaxCut)
                ar_vec.append(approx_ratio) # Remember, no cost for G-W!
                avg_train_time += temp_train_time; print()
                # For 'G-W', 'cost_vec' is None, and 'avg_train_time' is to be seen as 'avg_run_time'.
                print(f"Approximation ratio: {approx_ratio}. (Cut = {approx_ratio*MaxCut}; MaxCut = {MaxCut}).")
            elif(VQA == 'iQAQE'):
                # Default 'rs' values
                if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]

                # Parameters initialization
                params = 2 * np.pi * np.random.rand(n_layers, n_qubits, 4, requires_grad=True) if non_deterministic_CNOT == True else \
                         2 * np.pi * np.random.rand(n_layers, n_qubits, 3, requires_grad=True)
                if(not params_silent): print(f'Parameters = {params}.\n')

                _, _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                self.iQAQE_MaxCut(n_qubits = n_qubits, n_layers = n_layers,
                                  device = device, shots = shots,
                                  parameters = params, rs = rs, draw_circuit = False,
                                  basis_states_lists = basis_states_lists, B = B,
                                  max_iter = max_iter, cost_plot = False,
                                  MaxCut = MaxCut, nodes_probs_plot = False,
                                  non_deterministic_CNOT = non_deterministic_CNOT, **kwargs)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            elif(VQA == 'Exponential_QEMC_iQAQE'):
                # Re-define the number of qubits
                n_qubits = math.ceil(np.log2(self.n_nodes))

                # 'diff_Rx' value
                print(f"diff_Rx = {diff_Rx}.")

                # Parameters initialization
                params = 2 * np.pi * np.random.rand(2**n_qubits - 1 + n_qubits, n_layers, requires_grad=True) if diff_Rx else 2 * np.pi * np.random.rand(2**n_qubits - 1 + 1, n_layers, requires_grad=True)
                if(not params_silent): print(f'Parameters = {params}.\n')

                _, _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                self.Exponential_QEMC_iQAQE_MaxCut(n_layers = n_layers, device = device,
                                                  shots = shots, parameters = params,
                                                  draw_circuit = False, B = B,
                                                  basis_states_lists = basis_states_lists,
                                                  MaxCut = MaxCut, max_iter = max_iter, diff_Rx = diff_Rx,
                                                  cost_plot = False, nodes_probs_plot = False, **kwargs)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            elif(VQA == 'Parity_Inspired'):
                # 'seed' initialization
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                self.Parity_Inspired_MaxCut(k = k, n_layers = n_layers,
                                  device = device, shots = shots,
                                  seed = seed, draw_circuit = False,
                                  basis_states_lists = basis_states_lists, B = B,
                                  max_iter = max_iter, cost_plot = False, verbose = False,
                                  MaxCut = MaxCut, nodes_probs_plot = False, **kwargs)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            elif(VQA == 'QEMC'):
                # Re-define the number of qubits
                n_qubits = math.ceil(np.log2(self.n_nodes))

                # Default 'rs' values
                if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]

                # Parameters initialization
                params = 2 * np.pi * np.random.rand(n_layers, n_qubits, 4, requires_grad=True) if non_deterministic_CNOT == True else \
                         2 * np.pi * np.random.rand(n_layers, n_qubits, 3, requires_grad=True)
                if(not params_silent): print(f'Parameters = {params}.\n')

                _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                self.QEMC_MaxCut(n_layers = n_layers, device = device,
                                 shots = shots, parameters = params,
                                 rs = rs, draw_circuit = False,
                                 B = B, max_iter = max_iter,
                                 cost_plot = False, MaxCut = MaxCut,
                                 nodes_probs_plot = False,
                                 non_deterministic_CNOT = non_deterministic_CNOT, **kwargs)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            elif(VQA == 'QAOA'):
                # Parameters initialization
                params = 2 * np.pi * np.random.rand(2, n_layers, requires_grad=True)
                if(not params_silent): print(f'Parameters = {params}.\n')

                _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                self.QAOA_MaxCut(n_layers = n_layers, device = device,
                                 shots = shots, parameters = params,
                                 draw_circuit = False, max_iter = max_iter,
                                 cost_plot = False, MaxCut = MaxCut,
                                 probs_plot = False, **kwargs)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            time.sleep(0.5); clear_output(wait=True)
        avg_train_time /= 10

        # Populating the min-max approximation ratio vector
        min_ar_vec = np.min(ar_vec, axis = 0); max_ar_vec = np.max(ar_vec, axis = 0)

        # Compute the average approximation ratio vector and the standard deviation vector
        avg_vec = np.mean(ar_vec, axis = 0); std_vec = np.std(ar_vec, axis = 0)

        # Auxiliary function: Best-so-far transformation (Used in method: '1_Bsf_2_Avg'.)
        def BSF_Transform(vec):
            """
            Transforms a vector into a best-so-far vector.
            """
            bsf_vec = vec.copy()
            for i in range(1, len(bsf_vec)):
                if bsf_vec[i] < bsf_vec[i-1]:
                    bsf_vec[i] = bsf_vec[i-1]
            return bsf_vec

        # Best-so-far approximation ratio vector. 'np.mean(ar_vec, axis = 0)' is the actual average!
        if(VQA != 'G-W'):
            # First we average the AR_Vec, then we compute the BSF metric: Best-so-far of the average.
            bsf_avg_vec = BSF_Transform(avg_vec)

            # First we compute the BSF metric, for each run, then we average them: Avg. best-so-far. I think that's what is done in the paper: https://arxiv.org/abs/2308.10383
            bsf_vec = [BSF_Transform(ar_vec[i]) for i in range(repeat)]
            avg_bsf_vec = np.mean(bsf_vec, axis = 0)

            # Median best-so-far approximation ratio vector
            med_bsf_vec = np.median(bsf_vec, axis = 0)
        else:
            bsf_avg_vec, avg_bsf_vec, med_bsf_vec = None, None, None

        # Average cost vector. 'np.mean(cost_vec, axis = 0)' is the actual average!
        avg_cost_vec = np.mean(cost_vec, axis = 0) if VQA != 'G-W' else None

        if(plot_flag):
            # Deciding which 'method' to plot.
            if(method == 'Bsf_Avg'): y = bsf_avg_vec; label = f"Best-so-far Avg. approximation ratio ({VQA})"
            elif(method == 'Avg_Bsf'): y = avg_bsf_vec; label = f"Avg. best-so-far approximation ratio ({VQA})" # This is the default.

            # Plot the avg. best-so-far approximation ratio
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(y) + 1), y, label=label)
            plt.legend()
            plt.xlabel("# Iterations")
            plt.ylabel(label)
            plt.title(label + f" evolution (MaxCut = {MaxCut})")
            plt.show()

        return ar_vec, med_bsf_vec, avg_bsf_vec, bsf_avg_vec, avg_vec, std_vec, avg_cost_vec, min_ar_vec, max_ar_vec, avg_train_time