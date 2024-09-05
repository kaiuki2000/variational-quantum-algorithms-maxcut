import sys; sys.path.append("../..") # Adds higher, higher directory to python modules path.

from vqas_maxcut.vqa_graph import *
import vqas_maxcut.utils as ut
    
# Interpolated QAOA/QEMC (iQAQE) algorithm for the MaxCut problem
def iqaqe(vqa_graph_instance: vqa_graph, n_layers = None, shots = None,  device = 'default.qubit',
          basis_states_lists = None, rs = None, non_deterministic_CNOT = False, B = None,
          max_iter = 100, rel_tol = 0, abs_tol = 0, consecutive_count = 5, parameters = None, seed = None,
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
        consecutive_count (int): The number of consecutive counts for the convergence criteria.
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
        nodes_probs = ut.compute_nodes_probs(probs, basis_states_lists, **kwargs)
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

    # Auxiliary variable for convergence criteria
    count = consecutive_count

    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        if (MaxCut is not None):
            approx_ratio  = 0
            probabilities = circuit(parameters)
            nodes_probs   = ut.compute_nodes_probs(probabilities, basis_states_lists)
            partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
            cut           = vqa_graph_instance.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
        
        if i % 5 == 0:
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
        
        # Check convergence criteria - absolute tolerance
        if abs_tol != 0 and i >= 2:
            abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
            if abs_diff <= abs_tol:
                count += 1
                if count == 5:
                    print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol} (For {count} consecutive iterations; At i = {i}.)")
                    break
            else:
                count = 0

        # Check convergence criteria - relative tolerance
        if rel_tol != 0 and i >= 2:
            rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
            if rel_diff <= rel_tol:
                count += 1
                if count == 5:
                    print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol} (For {count} consecutive iterations; At i = {i}.)")
                    break
            else:
                count = 0

        # Check maximum iterations
        if i == max_iter:
            print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
            break

    train_time = time.time() - start_time # Time taken for training
    print("--- Training took %s seconds ---" % (train_time))

    probabilities = circuit(parameters)
    nodes_probs   = ut.compute_nodes_probs(probabilities, basis_states_lists)
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


















def exp_iqaqe(vqa_graph_instance: vqa_graph, n_layers = None, shots = None, device = 'default.qubit', 
             basis_states_lists = None, diff_Rx = False, B = None,
             max_iter = 100, rel_tol = 0, abs_tol = 0, consecutive_count = 5, parameters = None, seed = None,
             draw_circuit = False, draw_cost_plot = True, draw_probs_plot = False, draw_nodes_probs_plot = True,
             MaxCut = None, step_size = 0.99):
    """
    Implements the Exponential iQAQE algorithm for the MaxCut problem.
    This scheme implements a different ansatz, with an exponential number of parameters.
    It's supposed to be the 'QEMC-limit' of the following 'semi-problem-informed-ansatz' that follows,
    and, thus, should use the 'QEMC-limit' basis states lists. (Check 'utils.py' for more details - `build_qemc_bsl`.)
    [Although, we do leave the option to utilize the iQAQE framework, hence why we call it 'Exponential iQAQE'.]

    **Args:**
        vqa_graph_instance (vqa_graph): The VQA graph instance.
        n_layers (int): The number of layers.
        shots (int): The number of shots.
        device (str): The device to use.
        basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.
        diff_Rx (bool): Whether to use different Rx parameters for each qubit.
        B (int): The value of 'B'.
        max_iter (int): The maximum number of iterations.
        rel_tol (float): The relative tolerance.
        abs_tol (float): The absolute tolerance.
        consecutive_count (int): The number of consecutive counts for the convergence criteria.
        parameters (array): The parameters.
        seed (int): The seed.
        draw_circuit (bool): Whether to draw the circuit.
        draw_cost_plot (bool): Whether to draw the cost plot.
        draw_probs_plot (bool): Whether to draw the probabilities plot.

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
        parameters = 2 * np.pi * np.random.rand(2**n_qubits - 1 + n_qubits, n_layers, requires_grad=True) if diff_Rx else 2 * np.pi * np.random.rand(2**n_qubits - 1 + 1, n_layers, requires_grad=True)
    else:
        # This is to ensure that the 'parameters' array, when provided, has the correct shape.
        assert(parameters.shape == (2**n_qubits - 1 + n_qubits, n_layers) if diff_Rx else (2**n_qubits - 1 + 1, n_layers)), "parameters has the wrong shape."
    
    if(B is None):
        B = vqa_graph_instance.n_nodes//2 # Default value for B

    # Status messages
    print(f"[Info.] B value: {B}.")

    # Device setup
    dev = qml.device(device, wires = n_qubits, shots = shots)
    
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
    
    if(draw_circuit):
        fig, ax = qml.draw_mpl(circuit, decimals=3, style='pennylane')(parameters)
        plt.close(fig)
        fig.savefig(f'exp_iqaqe_circuit_n{n_qubits}_p{n_layers}.png', dpi = 200, bbox_inches = 'tight')
        print('[Info.] Circuit plot saved to:', f'exp_iqaqe_circuit_n{n_qubits}_p{n_layers}.png')
        
    # Status message, before optimization
    print(f"Exponential iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

    # Define the cost function
    def objective(params):
        """
        This function computes the value of the cost function.

        **Args:**
            params (array): Array of parameters.

        **Returns:**
            float: The value of the cost function.
        """
        # First - Compute the probaility associated to each node, from the 'n_qubit' probability distribution
        probs = circuit(params); cost  = 0
        nodes_probs = ut.compute_nodes_probs(probs, basis_states_lists)
        # You can also grab the values from an ArrayBox by using its “_value” attribute.
        # Example usage, to retrive the numerical values of 'nodes_probs': [Box._value for Box in nodes_probs]
        # Second - Compute the cost function itself
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
    
    # Auxiliary variable for convergence criteria
    count = consecutive_count
    
    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        if (MaxCut is not None):
            approx_ratio  = 0
            probabilities = circuit(parameters)
            nodes_probs   = ut.compute_nodes_probs(probabilities, basis_states_lists)
            partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
            cut           = vqa_graph_instance.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)
        
        if i % 5 == 0:
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
        
        # Check convergence criteria - absolute tolerance
        if abs_tol != 0 and i >= 2:
            abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
            if abs_diff <= abs_tol:
                count += 1
                if count == 5:
                    print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol} (For {count} consecutive iterations; At i = {i}.)")
                    break
            else:
                count = 0

        # Check convergence criteria - relative tolerance
        if rel_tol != 0 and i >= 2:
            rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
            if rel_diff <= rel_tol:
                count += 1
                if count == 5:
                    print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol} (For {count} consecutive iterations; At i = {i}.)")
                    break
            else:
                count = 0

        # Check maximum iterations
        if i == max_iter:
            print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
            break

    train_time = time.time() - start_time # Time taken for training
    print("--- Training took %s seconds ---" % (train_time))
    
    probabilities = circuit(parameters)
    nodes_probs   = ut.compute_nodes_probs(probabilities, basis_states_lists)
    partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]

    if(draw_cost_plot):
        plt.figure(figsize=(8, 6))
        plt.plot(cost_vec, label="Cost function")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.title("Cost function evolution")
        plt.savefig(f'exp_iqaqe_cost_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Cost plot saved to:', f'exp_iqaqe_cost_n{n_qubits}_p{n_layers}.png')

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
        plt.savefig(f'exp_iqaqe_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Probabilities plot saved to:', f'exp_iqaqe_probs_n{n_qubits}_p{n_layers}.png')

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
        plt.savefig(f'exp_iqaqe_nodes_probs_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Nodes probabilities plot saved to:', f'exp_iqaqe_nodes_probs_n{n_qubits}_p{n_layers}.png')

    return probabilities, nodes_probs, cost, cost_vec, ar_vec, parameters, partition, train_time


















# Semi-problem-informed ansatz (SPIA) for the MaxCut problem, using the iQAQE Framework (Custom-build.)
def smpi_iqaqe(vqa_graph_instance: vqa_graph, n_layers = None, shots = None, device = 'default.qubit',
               ansatz_type = None, cost_type = None, encoding = None, B = None, rs = None, diff_Rx = False, diff_Rz = True,
               max_iter = 100, rel_tol = 0, abs_tol = 0, consecutive_count = 5, seed = None,
               draw_circuit = False, draw_cost_plot = True, draw_probs_plot = False, draw_nodes_probs_plot = True,
               MaxCut = None, step_size = 0.99, verbose = False):
    # Haven't implemented 'non-deterministic CNOT' for this ansatz.
    # Inpsect the arguments
    assert(ansatz_type == 'SEL' or ansatz_type == 'Brickwork' or ansatz_type == 'Parity'), "The 'ansatz_type' must be 'SEL', 'Brickwork' or 'Parity'."
    assert(cost_type == 'iQAQE' or cost_type == 'Expectation'), "The 'cost_type' must be 'iQAQE' or 'Expectation'."
    assert(n_layers is not None and encoding is not None and seed is not None), "n_layers, encoding and seed cannot be None."
    assert(encoding is not None), "encoding cannot be None."
    if(cost_type == 'iQAQE'):
        if(B is None):
            B = vqa_graph_instance.n_nodes//2 # Default value for B
    else:
        def get_expectation_partition(Exp_vals: list) -> str:
            partition = []
            for n in range(len(Exp_vals)):
                n_k = np.sign(Exp_vals[n]); # print(f"n_k: {n_k}. Exp_vals[n] = {Exp_vals[n]}.") # Debugging.
                if(n_k == 1):    partition.append(0)
                elif(n_k == -1): partition.append(1)
                else:            partition.append(1) # Default to 1. Bugfix [Sometimes, n_k = 0.]
                # print(f"Partition (*inside*): {partition}.") # Debugging.
            return "".join([str(p) for p in partition])
    
    # Extract the number of qubits from the encoding
    # We assume the provided 'encoding' uses all of the qubits.
    # (Will always happen for a valid 'encoding', as we're interested in reducing the number of qubits.)
    n_qubits = max(num for tup in encoding for num in tup) + 1
    if(verbose):
        print(f"Number of qubits: {n_qubits}.")

    # Original terms:
    terms = []
    for i, j in vqa_graph_instance.graph:
        terms.append(encoding[i] + encoding[j])
    if(verbose):
        print(f'terms: {terms}.')

    # Removing duplicates. This method preserves the order of the elements.
    clean_terms = [ut.remove_duplicates(T) for T in terms] # [value for i, value in enumerate(terms) if i == 0 or value != terms[i-1]]
    if(verbose):
        print(f'clean_terms: {clean_terms}.')
    
    # Generating the basis states lists, from the provided encoding
    bsl = ut.build_parity_bsl(encoding, n_qubits)

    # Parameters initialization
    match ansatz_type:
        case 'Parity':
            np.random.seed(seed)
            # 'diff_Rx' is a flag that indicates whether to use different 'Rx' parameters for each 'RX' gate.
            # 'diff_Rz' is a flag that indicates whether to use different 'Rz' parameters for each 'MultiRZ' gate.
            # Only for 'Parity'. You can specify them, but they'll be ignored for the other ansatz types. (When not 'Parity'.)
            if(diff_Rx and diff_Rz):
                parameters = 2 * np.pi * np.random.rand(n_layers, len(clean_terms) + n_qubits, requires_grad=True)
            elif(not diff_Rx and diff_Rz):
                parameters = 2 * np.pi * np.random.rand(n_layers, len(clean_terms) + 1, requires_grad=True)
            elif(diff_Rx and not diff_Rz):
                parameters = 2 * np.pi * np.random.rand(n_layers, 1 + n_qubits, requires_grad=True)
            else: # (not diff_Rx and not diff_Rz)
                parameters = 2 * np.pi * np.random.rand(n_layers, 1 + 1, requires_grad=True)

            # Parity layer definition
            def Parity_layer(params_layer):
                if(diff_Rz):
                    gammas = params_layer[:len(clean_terms)]; betas = params_layer[len(clean_terms):]
                else:
                    gammas = params_layer[:1]; betas = params_layer[1:]
                for i, tpl in enumerate(clean_terms):
                    if(diff_Rz):
                        qml.MultiRZ(gammas[i], wires=tpl)
                    else:
                        qml.MultiRZ(gammas[0], wires=tpl)
                for j in range(n_qubits):
                    if(diff_Rx):
                        qml.RX(betas[j], wires=j)
                    else:
                        qml.RX(betas[0], wires=j)
        case 'SEL':
            if rs is None:
                rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]
            np.random.seed(seed)
            parameters = 2 * np.pi * np.random.rand(n_layers, n_qubits, 3, requires_grad=True)
            # SEL layer definition:
            def SEL_layer(params_layer):
                global a
                for i in range(n_qubits):
                    qml.RX(params_layer[i][0], wires = i)
                    qml.RY(params_layer[i][1], wires = i)
                    qml.RZ(params_layer[i][2], wires = i)
                for j in range(n_qubits):
                    qml.CNOT(wires = [j, (j+rs[a]) % n_qubits])
                a += 1

        case 'Brickwork':
            if rs is None:
                rs = [i % (n_qubits-1) + 1 for i in range(n_layers)]
                np.random.seed(seed)
                parameters = 2 * np.pi * np.random.rand(n_layers, n_qubits, 4, requires_grad=True)
                # Brickwork layer definition:
                def Brickwork_layer(params_layer):
                    global b
                    for i in range(n_qubits):
                        qml.RX(params_layer[i][0], wires = i)
                        qml.RY(params_layer[i][1], wires = i)
                        qml.RZ(params_layer[i][2], wires = i)
                    for j in range(n_qubits):
                        # The 'wires' here were chosen somewhat 'arbitrarily'.
                        # This is because not much information is given about the 'brickwork' ansatz (https://arxiv.org/pdf/2401.09421v2).
                        ut.RXX(params_layer[j][3], wires = [(2*j + rs[b] - 1) % n_qubits, (2*j + rs[b]) % n_qubits])
                    b += 1

    # Device setup
    dev = qml.device(device, wires = n_qubits, shots = shots)
    
    # Circuit setup
    @qml.qnode(dev)
    def circuit(params, probs_flag = False):
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)

        # Building the ansatz
        match ansatz_type:
            case 'SEL':
                global a; a = 0
                qml.layer(SEL_layer, n_layers, params)
            case 'Brickwork':
                global b; b = 0
                qml.layer(Brickwork_layer, n_layers, params)
            case 'Parity':
                for n in range(n_layers):
                    Parity_layer(params[n])

        # "circuit"'s output
        match cost_type:
            case 'iQAQE':
                return qml.probs()
            case 'Expectation':
                if(probs_flag):
                    return qml.probs()
                else:
                    P_str_lst = []
                    for P_str in encoding:
                        P = qml.PauliZ(P_str[0])
                        for i in range(1, len(P_str)):
                            P = P @ qml.PauliZ(P_str[i])
                        P_str_lst.append(P)
                    return [qml.expval(P) for P in P_str_lst]

    if(draw_circuit):
        fig, ax = qml.draw_mpl(circuit, decimals=3, style='pennylane')(parameters)
        plt.close(fig)
        fig.savefig(f'smpi_iqaqe_circuit_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png', dpi = 200, bbox_inches = 'tight')
        print('[Info.] Circuit plot saved to:', f'smpi_iqaqe_circuit_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png')
        
    # Status message, before optimization
    print(f"semi-problem-informed-ansatz iQAQE level (# of layers): p = {n_layers}. (Step size: {step_size}.)")

    # Define the cost function
    def objective(params):
        match cost_type:
            case 'iQAQE':
                probs       = circuit(params)
                cost        = 0
                nodes_probs = ut.compute_nodes_probs(probs, bsl)
                for edge in vqa_graph_instance.graph:
                    d_jk = np.abs(nodes_probs[edge[0]] - nodes_probs[edge[1]]); s_jk = nodes_probs[edge[0]] + nodes_probs[edge[1]]
                    edge_cost = (d_jk - 1/B)**2 + (s_jk - 1/B)**2
                    cost += edge_cost
                return cost
            case 'Expectation':
                exp_vals = circuit(params, probs_flag = False)

                # First part of the cost function
                avg_H = 0
                k     = max([len(T) for T in encoding])
                # Although we allow for "different" k's, we'll use the maximum one.
                # In "normal" circumstances, all the terms (in the 'encoding') should have the same length.
                alpha = n_qubits**(np.floor(k/2)) # This assumes polynomial compression of order 'k'

                for i, j in vqa_graph_instance.graph:
                    avg_H += ( np.tanh(alpha * exp_vals[i]) * np.tanh(alpha * exp_vals[j]) )
                
                # Regularization term
                beta  = 0.5
                nu    = vqa_graph_instance.poljak_turzik_lower_bound_unweighted()
                L_reg = beta * nu * (1/vqa_graph_instance.n_nodes * np.sum([np.tanh(alpha * exp_val)**2 for exp_val in exp_vals]))**2
                
                # Total cost
                cost = avg_H + L_reg

                return cost

    # Initialize optimizer: Adam.
    opt = qml.AdamOptimizer(stepsize = step_size, beta1 = 0.9, beta2 = 0.99, eps = 1e-08)

    # Optimize parameters in objective
    start_time = time.time(); i, cost_vec, ar_vec = 0, [], [] # For timing
    print("Optimizing parameters...", end='\r')
    
    # Auxiliary variable for convergence criteria
    count = consecutive_count

    while(True):
        parameters, cost = opt.step_and_cost(objective, parameters); i += 1; cost_vec.append(cost)
        if (MaxCut is not None):
            approx_ratio = 0
            # We need this 'match' statement, as different 'cost_type' requires different read-outs. (For obtaining the partition.)
            match cost_type:
                case 'iQAQE':
                    probabilities = circuit(parameters)
                    nodes_probs   = ut.compute_nodes_probs(probabilities, bsl)
                    exp_vals      = None # Relevant only for the 'Expectation' cost type
                    partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
                case 'Expectation':
                    probabilities = circuit(parameters, probs_flag = True)
                    exp_vals      = circuit(parameters, probs_flag = False)
                    nodes_probs   = None # Relevant only for the 'iQAQE' cost type
                    partition     = get_expectation_partition(exp_vals)
            cut = vqa_graph_instance.compute_cut(partition); approx_ratio = cut/MaxCut; ar_vec.append(approx_ratio)

        if i % 5 == 0:
            print(f"Optimizing parameters... Objective after step {i:4d}: {cost:.7f}", end='\r')
        
        # Check convergence criteria - absolute tolerance
        if abs_tol != 0 and i >= 2:
            abs_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2])
            if abs_diff <= abs_tol:
                count += 1
                if count == 5:
                    print(); print(f"Convergence criteria reached: abs_tol <= {abs_tol} (For {count} consecutive iterations; At i = {i}.)")
                    break
            else:
                count = 0

        # Check convergence criteria - relative tolerance
        if rel_tol != 0 and i >= 2:
            rel_diff = np.abs(cost_vec[i - 1] - cost_vec[i - 2]) / np.abs(cost_vec[i - 2])
            if rel_diff <= rel_tol:
                count += 1
                if count == 5:
                    print(); print(f"Convergence criteria reached: rel_tol <= {rel_tol} (For {count} consecutive iterations; At i = {i}.)")
                    break
            else:
                count = 0

        # Check maximum iterations
        if i == max_iter:
            print(); print(f"Maximum number of iterations reached: max_iter = {max_iter}")
            break

    train_time = time.time() - start_time # Time taken for training
    print("--- Training took %s seconds ---" % (train_time))

    match cost_type:
        case 'iQAQE':
            probabilities = circuit(parameters)
            nodes_probs   = ut.compute_nodes_probs(probabilities, bsl)
            exp_vals      = None # Relevant only for the 'Expectation' cost type
            partition     = ['0' if node_prob < 1 / (2*B) else '1' for node_prob in nodes_probs]
        case 'Expectation':
            probabilities = circuit(parameters, probs_flag = True)
            exp_vals      = circuit(parameters, probs_flag = False)
            nodes_probs   = None # Relevant only for the 'iQAQE' cost type
            partition     = get_expectation_partition(exp_vals)
    
    if(draw_cost_plot):
        # Plotting the cost function
        plt.figure(figsize=(8, 6))
        plt.plot(cost_vec, label="Cost function")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Cost function")
        plt.title("Cost function evolution")
        plt.savefig(f'smpi_iqaqe_cost_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Cost plot saved to:', f'smpi_iqaqe_cost_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png')

    # Only 'meaningful'/'readable' for small graphs.
    if(draw_probs_plot):
        basis_states = [format(i, '0'+str(n_qubits)+'b') for i in range(2**n_qubits)]
        most_freq_bit_string = np.argmax(probabilities)
        print(f'Most frequently sampled bitstring: {format(most_freq_bit_string, "0"+str(n_qubits)+"b")} (0b) = {most_freq_bit_string} (0f).')

        # Plot the probabilities
        plt.figure(figsize=(8, 6))
        plt.bar(basis_states, probabilities)
        plt.xlabel("Basis States")
        plt.xticks(rotation=90)
        plt.ylabel("Probabilities")
        plt.title("Probabilities vs. Basis States")
        plt.savefig(f'smpi_iqaqe_probs_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
        print('[Info.] Probabilities plot saved to:', f'smpi_iqaqe_probs_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png')

    if(draw_nodes_probs_plot):
        match cost_type:
            case 'iQAQE':
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
                plt.savefig(f'smpi_iqaqe_nodes_probs_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
                print('[Info.] Nodes probabilities plot saved to:', f'smpi_iqaqe_nodes_probs_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png')
            case 'Expectation':
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
                plt.savefig(f'smpi_iqaqe_nodes_probs_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
                print('[Info.] Nodes probabilities plot saved to:', f'smpi_iqaqe_nodes_probs_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png')

    # Idea: Maybe I could make it so that we specify the number of parameters that we want, not the number of layers.
    # This way, we could have a more flexible approach to the number of parameters.

    return probabilities, nodes_probs, exp_vals, cost, cost_vec, ar_vec, parameters, partition, train_time