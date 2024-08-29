import sys; sys.path.append("../..") # Adds higher, higher directory to python modules path.
from vqas_maxcut.vqa_graph import vqa_graph

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
import math
import cvxpy as cp
from scipy.linalg import sqrtm

# Quantum Approximate Optimization Algorithm (QAOA)
def qaoa(vqa_graph_instance: vqa_graph, n_layers = None, shots = None, device = 'default.qubit',
        max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = None,
        draw_circuit = False, draw_cost_plot = False, draw_probs_plot = True,
        MaxCut = None, step_size = 0.99):
    """
    This function implements the Quantum Approximate Optimization Algorithm (QAOA) for the MaxCut problem.

    **Args:**
        vqa_graph_instance (vqa_graph): An instance of the vqa_graph class.
        n_layers (int): Number of layers (p) in the QAOA circuit.
        shots (int): Number of shots for the quantum device.
        device (str): Name of the device to use.
        max_iter (int): Maximum number of iterations for the optimization.
        rel_tol (float): Relative tolerance for the optimization.
        abs_tol (float): Absolute tolerance for the optimization.
        parameters (array): Array of parameters for the QAOA circuit.
        seed (int): Seed for the random number generator.
        draw_circuit (bool): Flag to draw the QAOA circuit.
        draw_cost_plot (bool): Flag to draw the cost function plot.
        draw_probs_plot (bool): Flag to draw the probabilities plot.
        MaxCut (int): Maximum cut value for the graph.
        step_size (float): Step size for the optimizer.

    **Returns:**
        array: Probabilities of the most likely outcomes.
        float: Cost function value.
        array: Cost function values over the iterations.
        array: Approximation ratio over the iterations.
        array: Parameters of the QAOA circuit.
        str: Most likely partition.
        float: Time taken for training.
    """ 
    assert(n_layers is not None), "n_layers cannot be None."

    if(parameters is None):
        assert(seed is not None), "parameters cannot be None, without a seed."
        np.random.seed(seed) # Set the seed, for reproducibility
        parameters = 2 * np.pi * np.random.rand(2, n_layers, requires_grad=True)
    else:
        # This is to ensure that the 'parameters' array, when provided, has the correct shape.
        assert(parameters.shape == (2, n_layers)), "parameters has the wrong shape."

    # Define the number of qubits
    n_qubits = vqa_graph_instance.n_nodes

    # Device setup
    dev = qml.device(device, wires = n_qubits, shots = shots)

    # Unitary operator 'U_C' (Cost/Problem) with parameters 'gamma'.
    def U_C(gamma):
        for edge in vqa_graph_instance.graph:
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
        for edge in vqa_graph_instance.graph:
            H += 0.5 * (1 - qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
        return qml.expval(H)
    
    if(draw_circuit):
        fig, ax = qml.draw_mpl(circuit, decimals=3, style='pennylane')(parameters[0], parameters[1])
        plt.close(fig)
        fig.savefig(f'qaoa_circuit_n{n_qubits}_p{n_layers}.png', dpi = 200, bbox_inches = 'tight')
        print('[Info.] Circuit plot saved to:', f'qaoa_circuit_n{n_qubits}_p{n_layers}.png')
        
    # Status message, before optimization
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
        neg_obj = -1 * circuit(gammas, betas)
        return neg_obj

    # Initialize optimizer: Adam.
    opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

    # Optimize parameters in objective
    start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
    print("Optimizing parameters...", end='\r')

    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        # Compute the approximation ratio, from the computed partition's cut
        if(MaxCut is not None):
            approx_ratio  = 0
            probabilities = circuit(parameters[0], parameters[1], probs_flag = True)
            partition     = format(np.argmax(probabilities), "0"+str(n_qubits)+"b")
            cut           = vqa_graph_instance.compute_cut(partition)
            approx_ratio  = cut/MaxCut
            ar_vec.append(approx_ratio)
        if(i % 5 == 0):
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
        
        # Check convergence criteria
        if abs_tol != 0 and i >= 2:
            abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
            if abs_diff <= abs_tol:
                print()
                print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                break

        if rel_tol != 0 and i >= 2:
            rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
            if rel_diff <= rel_tol:
                print()
                print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                break

        if(i == max_iter):
            print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
            break

    train_time = time.time() - start_time # Time taken for training
    print("--- Training took %s seconds ---" % (train_time))

    probabilities        = circuit(parameters[0], parameters[1], probs_flag = True)
    most_freq_bit_string = np.argmax(probabilities)
    partition            = format(most_freq_bit_string, "0"+str(n_qubits)+"b")
    print(f'Most frequently sampled bitstring: {partition} (0b) = {most_freq_bit_string} (0f).')

    if(draw_cost_plot):
        # Plotting the cost function
        plt.figure(figsize=(8, 6))
        plt.plot(cost_vec, label="Cost function")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.title("Cost function evolution")
        plt.savefig(f'qaoa_cost_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Cost plot saved to:', f'qaoa_cost_n{n_qubits}_p{n_layers}.png')

    if(draw_probs_plot):
        plt.figure(figsize=(8, 6))
        num_outcomes = len(probabilities); x = range(num_outcomes)

        # Sort the probabilities and the outcomes, to plot only the 10 most probable outcomes
        sorted_zip      = sorted(zip(probabilities, x), reverse = True)[:10]
        sorted_outcomes = [bin(outcome)[2:].zfill(n_qubits) for _, outcome in sorted_zip]
        sorted_probs    = [prob for prob, _ in sorted_zip]
        plt.xticks(fontsize = 8)
        plt.bar(sorted_outcomes, sorted_probs, label='Probability', color = 'C1')
        plt.xlabel('Outcome')
        plt.ylabel('Probability')
        plt.title(f'Discrete Probability Distribution (n_layers={n_layers}): 10 most likely outcomes')
        plt.legend(loc = 'best', frameon = True, fancybox=True)
        plt.savefig(f'qaoa_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Probabilities plot saved to:', f'qaoa_probs_n{n_qubits}_p{n_layers}.png')
        
    return probabilities, cost, cost_vec, ar_vec, parameters, partition, train_time



















# Qubit Efficient Max-Cut (QEMC) Algorithm
def qemc(vqa_graph_instance: vqa_graph, n_layers = None, shots = None, device = 'default.qubit',
        rs = None, non_deterministic_CNOT = False, B = None,
        max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = None,
        draw_circuit = False, draw_cost_plot = False, draw_nodes_probs_plot = True,
        MaxCut = None, step_size = 0.99):
    """
    This function implements the Qubit Efficient Max-Cut (QEMC) algorithm for the MaxCut problem.

    **Args:**
        vqa_graph_instance (vqa_graph): An instance of the vqa_graph class.
        n_layers (int): Number of layers (p) in the QEMC circuit.
        shots (int): Number of shots for the quantum device.
        device (str): Name of the device to use.
        rs (list): List of integers, where each integer represents the distance between the qubits in the CNOT gate. One integer per layer.
        non_deterministic_CNOT (bool): Flag to use non-deterministic CNOT gates.
        B (int): Parameter for the QEMC algorithm.
        max_iter (int): Maximum number of iterations for the optimization.
        rel_tol (float): Relative tolerance for the optimization.
        abs_tol (float): Absolute tolerance for the optimization.
        parameters (array): Array of parameters for the QEMC circuit.
        seed (int): Seed for the random number generator.
        draw_circuit (bool): Flag to draw the QEMC circuit.
        draw_cost_plot (bool): Flag to draw the cost function plot.
        draw_nodes_probs_plot (bool): Flag to draw the nodes probabilities plot.
        MaxCut (int): Maximum cut value for the graph.
        step_size (float): Step size for the optimizer.

    **Returns:**
        array: Probabilities of the most likely outcomes.
        float: Cost function value.
        array: Cost function values over the iterations.
        array: Approximation ratio over the iterations.
        array: Parameters of the QEMC circuit.
        str: Most likely partition.
        float: Time taken for training.
    """
    assert(n_layers is not None and B is not None), "n_layers and B cannot be None."

    # Define the number of qubits
    n_qubits = math.ceil(np.log2(vqa_graph_instance.n_nodes))

    if(parameters is None):
        assert(seed is not None), "parameters cannot be None, without a seed."
        np.random.seed(seed) # Set the seed, for reproducibility
        parameters = 2 * np.pi * np.random.rand(n_layers, n_qubits, 4, requires_grad=True) if non_deterministic_CNOT == True else \
                     2 * np.pi * np.random.rand(n_layers, n_qubits, 3, requires_grad=True)
    else:
        # This is to ensure that the 'parameters' array, when provided, has the correct shape.
        assert(parameters.shape == (n_layers, n_qubits, 4) if non_deterministic_CNOT == True else (n_layers, n_qubits, 3)), "parameters has the wrong shape."
    if(non_deterministic_CNOT): print("[Info.] Using non-deterministic CNOT gates. [RX-paramaterized.]") # Status message

    # Default 'rs' values
    if(rs is None): rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]

    # Status messages
    print(f"[Info.] rs values: {rs}.")
    
    # Device setup
    dev = qml.device(device, wires = n_qubits, shots = shots)
    
    # Circuit setup: Strongly Entangling Layers
    @qml.qnode(dev)
    def circuit(params):
        global c; c = 0
        def QEMC_layer(params_layer):
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
        qml.layer(QEMC_layer, n_layers, params)
        return qml.probs()
    
    if(draw_circuit):
        fig, ax = qml.draw_mpl(circuit, decimals=3, style='pennylane')(parameters)
        plt.close(fig)
        fig.savefig(f'qemc_circuit_n{n_qubits}_p{n_layers}.png', dpi = 200, bbox_inches = 'tight')
        print('[Info.] Circuit plot saved to:', f'qemc_circuit_n{n_qubits}_p{n_layers}.png')

    # Status message, before optimization
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
        for edge in vqa_graph_instance.graph:
            # j and k are the nodes connected by the edge
            # 0: j, 1: k
            d_jk = np.abs(probs[edge[0]] - probs[edge[1]]); s_jk = probs[edge[0]] + probs[edge[1]]
            edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
            cost += edge_cost
        return cost
    
    # Adam optimizer
    opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

    # Optimize parameters in objective
    start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
    print("Optimizing parameters...", end = '\r')

    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        # Compute the approximation ratio, from the computed partition's cut
        if (MaxCut is not None):
            approx_ratio = 0
            probabilities = circuit(parameters)
            partition = ['0' if probability < 1 / (2*B) else '1' for probability in probabilities]
            cut = vqa_graph_instance.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
        if i % 5 == 0:
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end = '\r')
        
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

    probabilities        = circuit(parameters)
    partition            = ['0' if prob < 1 / (2*B) else '1' for prob in probabilities]
    most_freq_bit_string = np.argmax(probabilities)
    print(f'Highest probability bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

    if(draw_cost_plot):
        # Plotting the cost function
        plt.figure(figsize=(8, 6))
        plt.plot(cost_vec, label="Cost function")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.title("Cost function evolution")
        plt.savefig(f'qemc_cost_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Cost plot saved to:', f'qemc_cost_n{n_qubits}_p{n_layers}.png')

    if(draw_nodes_probs_plot):
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
        plt.savefig(f'qemc_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Probabilities plot saved to:', f'qemc_probs_n{n_qubits}_p{n_layers}.png')

    return probabilities, cost, cost_vec, ar_vec, parameters, partition, train_time


















# Multi-angle Quantum Approximate Optimization Algorithm (ma-QAOA)
def ma_qaoa(vqa_graph_instance: vqa_graph, n_layers = None, shots = None, device = 'default.qubit',
            max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = None,
            draw_circuit = False, draw_cost_plot = False, draw_probs_plot = True,
            MaxCut = None, step_size = 0.99,
            diff_Rx = False):
    """
    This function implements the Multi-angle Quantum Approximate Optimization Algorithm (ma-QAOA) for the MaxCut problem.
    Note: There's a slightly different implementation - 'MA_QAOA_MaxCut' in 'VQA_Graph.py', in the 'HQCC_Beta' GitHub Repository.
          This other implementation considers the possibility of using an expectation value-based cost function.
          However, due to how it is defined, it yields a constant cost all the time, hindering training (Explained in 'MA_QAOA.ipynb', in the same Repository).
          If, for any reason, you want to use that implementation, you can find it in the 'VQA_Graph.py' file.
          That specific part of the code has been removed from this implementation, due to the aforementioned issue.

    **Args:**
        vqa_graph_instance (vqa_graph): An instance of the vqa_graph class.
        n_layers (int): Number of layers (p) in the ma-QAOA circuit.
        shots (int): Number of shots for the quantum device.
        device (str): Name of the device to use.
        max_iter (int): Maximum number of iterations for the optimization.
        rel_tol (float): Relative tolerance for the optimization.
        abs_tol (float): Absolute tolerance for the optimization.
        parameters (array): Array of parameters for the ma-QAOA circuit.
        seed (int): Seed for the random number generator.
        draw_circuit (bool): Flag to draw the ma-QAOA circuit.
        draw_cost_plot (bool): Flag to draw the cost function plot.
        draw_probs_plot (bool): Flag to draw the probabilities plot.
        MaxCut (int): Maximum cut value for the graph.
        step_size (float): Step size for the optimizer.
        diff_Rx (bool): Flag to use different parameters for each Rx gate.

    **Returns:**
        array: Probabilities of the most likely outcomes.
        float: Cost function value.
        array: Cost function values over the iterations.
        array: Approximation ratio over the iterations.
        array: Parameters of the ma-QAOA circuit.
        str: Most likely partition.
        float: Time taken for training.
    """
    assert(n_layers is not None), "n_layers cannot be None."

    if(parameters is None):
        assert(seed is not None), "parameters cannot be None, without a seed."
        np.random.seed(seed) # Set the seed, for reproducibility
        parameters = 2 * np.pi * np.random.rand(vqa_graph_instance.n_edges + vqa_graph_instance.n_nodes, n_layers, requires_grad=True) if diff_Rx else 2 * np.pi * np.random.rand(vqa_graph_instance.n_edges + 1, n_layers, requires_grad=True)
    else:
        # This is to ensure that the 'parameters' array, when provided, has the correct shape.
        assert(parameters.shape == (vqa_graph_instance.n_edges + vqa_graph_instance.n_nodes, n_layers) if diff_Rx else (vqa_graph_instance.n_edges + 1, n_layers)), "parameters has the wrong shape."
    
    # Define the number of qubits
    n_qubits = vqa_graph_instance.n_nodes

    # Device setup
    dev = qml.device(device, wires = n_qubits, shots = shots)

    # Unitary operator 'U_C' (Cost/Problem) with parameters 'gamma'.
    def U_C(gamma):
        for i, edge in enumerate(vqa_graph_instance.graph):
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

    # ma-QAOA circuit
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
        
        # During the optimization phase we are evaluating a term the objective using 'expval'
        H = 0
        for edge in vqa_graph_instance.graph:
            H += 0.5 * (1 - qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
        return qml.expval(H)
    
    if(draw_circuit):
        A = vqa_graph_instance.n_nodes if diff_Rx else 1
        fig, ax = qml.draw_mpl(circuit, decimals=3, style='pennylane')(parameters[:-A], parameters[-A:]) # Change thie.
        plt.close(fig)
        fig.savefig(f'ma_qaoa_circuit_n{n_qubits}_p{n_layers}.png', dpi = 200, bbox_inches = 'tight')
        print('[Info.] Circuit plot saved to:', f'ma_qaoa_circuit_n{n_qubits}_p{n_layers}.png')
        
    # Status message, before optimization
    print(f"ma-QAOA level (# of layers): p = {n_layers}. (Step size: {step_size}.)")
    
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
        gammas = params[:-vqa_graph_instance.n_nodes] if diff_Rx else params[:-1]; betas = params[-vqa_graph_instance.n_nodes:] if diff_Rx else params[-1:]
        neg_obj = -1 * circuit(gammas, betas)
        return neg_obj
    
    # Initialize optimizer: Adam.
    opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)
    
    # Optimize parameters in objective
    start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
    print("Optimizing parameters...", end='\r')

    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        # Compute the approximation ratio, from the computed partition's cut
        if(MaxCut is not None):
            approx_ratio  = 0
            probabilities = circuit(parameters[:-vqa_graph_instance.n_nodes], parameters[-vqa_graph_instance.n_nodes:], probs_flag = True) if diff_Rx else circuit(parameters[:-1], parameters[-1:], probs_flag = True)
            partition     = format(np.argmax(probabilities), "0"+str(n_qubits)+"b")
            cut           = vqa_graph_instance.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
        if i % 5 == 0:
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
        
        # Check convergence criteria
        if abs_tol != 0 and i >= 2:
            abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
            if abs_diff <= abs_tol:
                print()
                print(f"Convergence criteria reached: abs_tol <= {abs_tol}")
                break

        if rel_tol != 0 and i >= 2:
            rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
            if rel_diff <= rel_tol:
                print()
                print(f"Convergence criteria reached: rel_tol <= {rel_tol}")
                break

        # Check maximum iterations
        if i == max_iter:
            print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
            break

    train_time = time.time() - start_time # Time taken for training
    print("--- Training took %s seconds ---" % (train_time))

    probabilities        = circuit(parameters[:-vqa_graph_instance.n_nodes], parameters[-vqa_graph_instance.n_nodes:], probs_flag = True) if diff_Rx else circuit(parameters[:-1], parameters[-1:], probs_flag = True)
    most_freq_bit_string = np.argmax(probabilities)
    partition            = format(most_freq_bit_string, "0"+str(n_qubits)+"b")
    print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

    if(draw_cost_plot):
        # Plotting the cost function
        plt.figure(figsize=(8, 6))
        plt.plot(cost_vec, label="Cost function")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.title("Cost function evolution")
        plt.savefig(f'ma_qaoa_cost_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Cost plot saved to:', f'ma_qaoa_cost_n{n_qubits}_p{n_layers}.png')

    if(draw_probs_plot):
        plt.figure(figsize=(8, 6))
        num_outcomes = len(probabilities); x = range(num_outcomes)
        
        # Sort the probabilities and the outcomes, to plot only the 10 most probable outcomes
        sorted_zip      = sorted(zip(probabilities, x), reverse = True)[:10]
        sorted_outcomes = [bin(outcome)[2:].zfill(n_qubits) for _, outcome in sorted_zip]
        sorted_probs    = [prob for prob, _ in sorted_zip]
        plt.xticks(fontsize = 8)
        plt.bar(sorted_outcomes, sorted_probs, label='Probability', color = 'C1')
        plt.xlabel('Outcome')
        plt.ylabel('Probability')
        plt.title(f'Discrete Probability Distribution (n_layers={n_layers}): 10 most likely outcomes')
        plt.legend(loc = 'best', frameon = True, fancybox=True)
        plt.savefig(f'ma_qaoa_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Probabilities plot saved to:', f'ma_qaoa_probs_n{n_qubits}_p{n_layers}.png')

    return probabilities, cost, cost_vec, ar_vec, parameters, partition, train_time


















def Goemans_Williamson(self, MaxCut = None):
    """
    This function implements the Goemans-Williamson algorithm for the MaxCut problem.
    (Uses a semi-definite programming relaxation.)

    **Args:**
        MaxCut (int): Maximum cut value for the graph.

    **Returns:**
        array: Partition of the nodes.
        float: Cut value.
        float: Approximation ratio.
        float: Time taken for running.
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
    if (MaxCut is not None):
        approx_ratio = cut/MaxCut
    else:
        approx_ratio = None
        
    # Return the partition
    return partition, cut, approx_ratio, run_time