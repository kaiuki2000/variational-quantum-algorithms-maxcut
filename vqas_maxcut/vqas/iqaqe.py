import sys; sys.path.append("../..") # Adds higher, higher directory to python modules path.
from vqas_maxcut.vqa_graph import vqa_graph
from vqas_maxcut.utils import compute_nodes_probs

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time

# Interpolated QAOA/QEMC (iQAQE) algorithm for the MaxCut problem
def iqaqe(vqa_graph_instance: vqa_graph, n_layers = None, shots = None,  device = 'default.qubit',
          basis_states_lists = None, rs = None, non_deterministic_CNOT = False, B = None,
          max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = None,
          draw_circuit = False, draw_cost_plot = False, draw_probs_plot = False, draw_nodes_probs_plot = True,
          MaxCut = None, step_size = 0.99, **kwargs):
    """
    Implements the iQAQE algorithm for the MaxCut problem.

    **Args:**
        vqa_graph_instance (vqa_graph): The VQA graph instance.
        n_layers (int): The number of layers.
        shots (int): The number of shots.
        device (str): The device to use.
        basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.
        rs (list): List of 'rs' values.
        non_deterministic_CNOT (bool): Whether to use non-deterministic CNOT gates.
        B (int): The value of 'B'.
        max_iter (int): The maximum number of iterations.
        rel_tol (float): The relative tolerance.
        abs_tol (float): The absolute tolerance.
        parameters (array): The parameters.
        seed (int): The seed.
        draw_circuit (bool): Whether to draw the circuit.
        draw_cost_plot (bool): Whether to draw the cost plot.
        draw_probs_plot (bool): Whether to draw the probabilities plot.
        draw_nodes_probs_plot (bool): Whether to draw the nodes probabilities plot.
        MaxCut (int): The maximum cut value.
        step_size (float): The step size.

    **Returns:**
        array: The probabilities.
        list: The nodes probabilities.
        float: The cost.
        list: The cost vector.
        list: The approximation ratio vector.
        array: The parameters.
        list: The partition.
        float: The training time.
    """
    
    assert(n_layers is not None and basis_states_lists is not None), "n_layers and basis_states_lists cannot be None."
    assert(len(basis_states_lists) == vqa_graph_instance.n_nodes), "The number of basis states lists must be equal to the graph's number of nodes."
    n_qubits = len(basis_states_lists[0][0]) # Extract the number of qubits from the basis states lists
    
    if(parameters is None):
        assert(seed is not None), "parameters cannot be None, without a seed."
        np.random.seed(seed) # Set the seed, for reproducibility
        parameters = 2 * np.pi * np.random.rand(n_layers, n_qubits, 4, requires_grad=True) if non_deterministic_CNOT == True else \
                     2 * np.pi * np.random.rand(n_layers, n_qubits, 3, requires_grad=True)
    else:
        # This is to ensure that the 'parameters' array, when provided, has the correct shape.
        assert(parameters.shape == (n_layers, n_qubits, 4) if non_deterministic_CNOT == True else (n_layers, n_qubits, 3)), "parameters has the wrong shape."
    if(non_deterministic_CNOT): print("[Info.] Using non-deterministic CNOT gates. [RX-paramaterized.]") # Status message

    if(B is None):
        B = vqa_graph_instance.n_nodes//2 # Default value for B
    
    # Default 'rs' values
    if rs is None: rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]

    # Status messages
    print(f"[Info.] rs values: {rs}.")
    print(f"[Info.] B value: {B}.")
    
    # Device setup
    dev = qml.device(device, wires = n_qubits, shots = shots)
    
    # Circuit setup
    @qml.qnode(dev)
    def circuit(params):
        global c; c = 0
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
        for wire in range(n_qubits):
            qml.Hadamard(wires = wire)
        qml.layer(iQAQE_layer, n_layers, params)
        return qml.probs()

    if(draw_circuit):
        fig, ax = qml.draw_mpl(circuit, decimals=3, style='pennylane')(parameters)
        plt.close(fig)
        fig.savefig(f'iqaqe_circuit_n{n_qubits}_p{n_layers}.png', dpi = 200, bbox_inches = 'tight')
        print('[Info.] Circuit plot saved to:', f'iqaqe_circuit_n{n_qubits}_p{n_layers}.png')
        
    # Status message, before optimization
    print(f"iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")
    
    # Define the cost function
    def objective(params):
        """
        This function computes the value of the cost function.

        **Args:**
            params (array): Array of parameters.

        **Returns:**
            float: The value of the cost function.
        """
        probs = circuit(params); cost  = 0
        nodes_probs = compute_nodes_probs(probs, basis_states_lists, **kwargs)
        for edge in vqa_graph_instance.graph:
            # j and k are the nodes connected by the edge
            # 0: j, 1: k
            d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
            edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
            cost += edge_cost
        return cost
    
    # Initialize optimizer: Adam.
    opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

    # Optimize parameters in objective
    start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
    print("Optimizing parameters...", end='\r')

    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        if (MaxCut is not None):
            approx_ratio  = 0
            probabilities = circuit(parameters)
            nodes_probs   = compute_nodes_probs(probabilities, basis_states_lists)
            partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
            cut           = vqa_graph_instance.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
        if i % 5 == 0:
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
        
        # Check convergence criteria
        if abs_tol != 0 and i >= 2:
            abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
            if abs_diff <= abs_tol:
                print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                break

        if rel_tol != 0 and i >= 2:
            rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
            if rel_diff <= rel_tol:
                print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                break

        # Check maximum iterations
        if i == max_iter:
            print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
            break

    train_time = time.time() - start_time # Time taken for training
    print("--- Training took %s seconds ---" % (train_time))

    probabilities = circuit(parameters)
    nodes_probs   = compute_nodes_probs(probabilities, basis_states_lists)
    partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

    if(draw_cost_plot):
        # Plotting the cost function
        plt.figure(figsize=(8, 6))
        plt.plot(cost_vec, label="Cost function")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.title("Cost function evolution")
        plt.savefig(f'iqaqe_cost_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Cost plot saved to:', f'iqaqe_cost_n{n_qubits}_p{n_layers}.png')

    # Only 'meaningful'/'readable' for small graphs.
    if(draw_probs_plot):
        basis_states         = [format(i, '0'+str(n_qubits)+'b') for i in range(2**n_qubits)]
        most_freq_bit_string = np.argmax(probabilities)
        print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')
        
        plt.figure(figsize=(8, 6))
        plt.bar(basis_states, probabilities)
        plt.xlabel("Basis States")
        plt.ylabel("Probabilities")
        plt.title("Probabilities vs. Basis States")
        plt.savefig(f'iqaqe_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Probabilities plot saved to:', f'iqaqe_probs_n{n_qubits}_p{n_layers}.png')

    if(draw_nodes_probs_plot):
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
        plt.savefig(f'iqaqe_nodes_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Nodes probabilities plot saved to:', f'iqaqe_nodes_probs_n{n_qubits}_p{n_layers}.png')

    return probabilities, nodes_probs, cost, cost_vec, ar_vec, parameters, partition, train_time


















# Multi-angle QAOA (ma-QAOA) for the MaxCut problem, using the iQAQE Framework
def ma_qaoa_iqaqe(vqa_graph_instance: vqa_graph, n_layers = None, shots = None, device = 'default.qubit',
                 diff_Rx = False, B = None, basis_states_lists = None, 
                 max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = None,
                 draw_circuit = False, draw_cost_plot = False, draw_probs_plot = True,
                 MaxCut = None, step_size = 0.99, **kwargs):
    """
    Implements the ma-QAOA algorithm for the MaxCut problem, using the iQAQE framework.
    Note: This was removed from the main code, as it proved to be useless. Kept here for reference.
          The reason for that has to do with how the cost function would be constant, due to the
          specific 'bsl' that is used.
    Explanation: the ma-QAOA ansatz generates symmetric states, which results in a constant cost.
          Example: |\psi> = a|00> + b|01> + b|10> + a|11>, such that |a|^2 + |b|^2 = 1/2. (Normalization.)
          Then, P(Node 0) = P(|10>) + P(|11>) = 1/2. (Constant cost.)
          And, P(Node 1) = P(|01>) + P(|11>) = 1/2. (Constant cost.)
    Previous implementation: 'MA_QAOA_iQAQE_MaxCut' in 'VQA_Graph.py', in the 'HQCC_Beta' GitHub Repository.
    """
    pass
                
