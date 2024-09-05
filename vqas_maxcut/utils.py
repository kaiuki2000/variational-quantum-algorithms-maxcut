# Description: This file contains utility functions used in the 'variational-quantum-algorithms-maxcut' project.
import sys; sys.path.append("../..") # Adds higher, higher directory to python modules path.

from vqas_maxcut.vqa_graph import *
import vqas_maxcut.vqas.base as bs
import vqas_maxcut.vqas.iqaqe as iq

def compute_nodes_probs(probabilities, basis_states_lists, **kwargs):
    """
    Computes the probabilities associated with each node.

    **Args:**
        probabilities (list): The probabilities of each basis state.
        basis_states_lists (list[list]): List of lists of basis states assigned to each graph node.

    **Returns:**
        list: The probabilities associated with each node.
    """
    # # Testing using the 'softmax' function, instead of mere normalization.
    # cte = kwargs.get('cte', 1.0); print(f"Softmax scaling constant: {cte}.")
    # def softmax(x, cte = 1.0):
    #     """Compute softmax values for each sets of scores in x."""
    #     return np.exp(cte * x) / np.sum(np.exp(cte * x), axis=0)
    
    nodes_probs = []
    for sublist in basis_states_lists:
        node_prob = 0
        for basis_state in sublist: node_prob += probabilities[int(basis_state, 2)]
        nodes_probs.append(node_prob)
    nodes_probs = [prob/sum(nodes_probs) for prob in nodes_probs]
    # nodes_probs = softmax(np.array(nodes_probs), cte) # Testing using the 'softmax' function, instead of mere normalization.
    return nodes_probs








def build_qaoa_bsl(n_qubits):
    """
    Builds the basis states lists for the QAOA algorithm, in the iQAQE formalism.
    This fixed one qubit to '1' and permutes the remaining qubits. The fixed qubit is different for each node.
    Example: For n_nodes = 3, the basis states lists are:
    [['100', '101', '110', '111'], ['010', '011', '000', '001'], ['001', '000', '011', '010']].

    **Args:**
        n_qubits (int): The number of qubits.

    **Returns:**
        list: The basis states lists for the QAOA algorithm, in the iQAQE formalism.
    """
    bsl          = [] # Basis states lists
    permutations = ["".join(seq) for seq in itertools.product("01", repeat = n_qubits - 1)]
    for i in range(n_qubits):
        individual_list = []
        for perm in permutations:
            individual_list.append(perm[:i] + "1" + perm[i:])
        bsl.append(individual_list)
    return bsl








def build_parity_bsl(encoding, n_qubits):
    """
    Builds the basis states lists for the 'Parity-like' encoding, in the iQAQE formalism.
    Example: For 'encoding = [(0), (1, 2, 3), (0, 3), (1, 2), (1, 3), (2, 3)]',          \
    the basis states lists are: (For 'n_qubits = 4')
    [['1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'], ['0111', '1111'], \
    ['1001', '1011', '1101', '1111'], ['0110', '0111', '1110', '1111'],                  \
    ['0101', '0111', '1101', '1111'], ['0011', '0111', '1011', '1111']].

    **Args:**
        encoding (list): The encoding of the basis states.
        n_qubits (int): The number of qubits.

    **Returns:**
        list: The basis states lists for the 'Parity-like' encoding, in the iQAQE formalism.
    """
    parity_bsl = []
    for T in encoding:
        # Exception handling for the case when 'T' is an integer.
        # E.g.: encoding = [(0), (1, 2)].
        # In this case, for the first tuple of the list, 'T' is an integer, and not a list.
        T = [T] if isinstance(T, int) else T

        individual_lst = []
        n_free         = n_qubits - len(T)
        free_list      = [''.join(seq) for seq in itertools.product("01", repeat=n_free)]
        for element in free_list:
            element_tmp = element
            for index in T:
                element_tmp = element_tmp[:index] + '1' + element_tmp[index:]
            individual_lst.append(element_tmp)
        parity_bsl.append(individual_lst)
    return parity_bsl








def build_qemc_bsl(n_qubits):
    """
    Builds the basis states lists for the QEMC algorithm, in the iQAQE formalism.
    Example: For n_nodes = 8, the basis states lists are:
    [['000'], ['001'], ['010'], ['011'], ['100'], ['101'], ['110'], ['111']].

    **Args:**
        n_qubits (int): The number of qubits.
        
    **Returns:**
        list: The basis states lists for the QEMC algorithm, in the iQAQE formalism.
    """
    tmp_bsl = [bin(i)[2:] for i in range(2**n_qubits)]
    bsl = [[str(i).zfill(n_qubits)] for i in tmp_bsl]
    return bsl








def combinations(n, k):
    """
    Computes the number of combinations of 'n' items taken 'k' at a time.

    **Args:**
        n (int): The number of items.
        k (int): The number of items to take at a time.

    **Returns:**
        int: The number of combinations of 'n' items taken 'k' at a time.
    """
    return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))








def remove_duplicates(lst):
    """
    Removes duplicates from a list, two by two. (Keeps the order of the elements.)

    **Args:**
        lst (list): The list to remove duplicates from.

    **Returns:**
        list: The list without duplicates
    """
    seen = set()
    result = []
    for num in lst:
        if num not in seen:
            seen.add(num)
            result.append(num)
        else:
            seen.remove(num)
            result.remove(num)
    return result
    







# RXX gate definition: 2-qubit Molmer-Sorensen gate
class RXX(Operation):
    num_params = 1
    num_wires = 2
    par_domain = "R"

    grad_method = "A"
    grad_recipe = None # This is the default but we write it down explicitly here.

    def generator(self):
        return -0.5 * (qml.PauliX(0) @ qml.PauliX(1))

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







def polynomial_compression_n_qubits(vqa_graph_instance: vqa_graph, k: int = None) -> int:
    """
    Computes the number of qubits needed for the polynomial compression (of order k) the graph.

    **Args:**
        vqa_graph_instance (vqa_graph): The VQA graph instance.
        k (int): The order of the polynomial compression.

    **Returns:**
        int: The number of qubits needed for the polynomial compression.
    """
    match k:
        case 2: # Exact formula for k = 2
            n_qubits = math.ceil((1 + np.sqrt(1 + 8*vqa_graph_instance.n_nodes))/2)
        case _: # For k > 2, we solve a polynomial equation
            def polynomial(n):
                result = 1
                for i in range(k):
                    result *= (n - i)
                return result - math.factorial(k) * vqa_graph_instance.n_nodes
            
            # Find the root using Newton's method
            n_qubits = math.ceil(optimize.newton(polynomial, vqa_graph_instance.n_nodes))
    return n_qubits








def generate_polynomial_encoding(vqa_graph_instance: vqa_graph, k: int = None) -> list:
    """
    Generates the polynomial encoding for the polynomial compression (of order k) of the graph.

    **Args:**
        vqa_graph_instance (vqa_graph): The VQA graph instance.
        k (int): The order of the polynomial compression.

    **Returns:**
        list: The polynomial encoding for the polynomial compression
    """
    # assert(k == 2 or k == 3), "The 'k' value must be 2 or 3. (Only these two values are supported.)"
    # This has been updated to support any value of 'k'.
    n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k)
    encoding = list(itertools.combinations(range(n_qubits), k))[:vqa_graph_instance.n_nodes]
    return encoding







# Need to implement 'diff_Rz' for the 'Parity' ansatz!
def number_of_parameters(vqa_graph_instance: vqa_graph, ansatz_type, n_layers, k, diff_Rx = False) -> int:
    """
    Computes the number of parameters needed for the VQA algorithm. (For the given ansatz type.)
    This is framed in the context of the 'Parity-inspired' ansatz, hence we always assume a polynomial compression
    in the number of qubits, of order 'k' (k = 2 or k = 3), despite the ansatz type.

    **Args:**
        vqa_graph_instance (vqa_graph): The VQA graph instance.
        ansatz_type (str): The ansatz type. ('Brickwork', 'SEL', 'Parity')
        n_layers (int): The number of layers.
        k (int): The order of the polynomial compression. (k = 2 or k = 3)
        diff_Rx (bool): Flag to use different parameters for each 'Rx' gate. (Only used in the 'Parity' ansatz.)

    **Returns:**
        int: The number of parameters needed for the VQA algorithm.
    """
    # Number of qubits
    n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k)

    # Number of parameters
    match ansatz_type:
        case 'Brickwork':
            return n_layers * n_qubits * 4
        case 'SEL':
            return n_layers * n_qubits * 3
        case 'Parity':
            encoding = generate_polynomial_encoding(vqa_graph_instance, k)[:vqa_graph_instance.n_nodes]
            terms = []
            for i, j in vqa_graph_instance.graph:
                terms.append(encoding[i] + encoding[j])
            clean_terms = [remove_duplicates(T) for T in terms]
            if(diff_Rx):
                return n_layers * ( n_qubits + len(clean_terms) )
            else:
                return n_layers * ( n_qubits + 1 )
        case _:
            raise Exception("Invalid ansatz type!")








# Auxiliary function: Best-so-far transformation (Used in method: '1_Bsf_2_Avg'.)
def bsf_transform(vec):
    """
    Transforms a vector into a best-so-far vector.
    """
    bsf_vec = vec.copy()
    for i in range(1, len(bsf_vec)):
        if bsf_vec[i] < bsf_vec[i-1]:
            bsf_vec[i] = bsf_vec[i-1]
    return bsf_vec
    





    

def avg_best_so_far(vqa_graph_instance: vqa_graph, vqa = None, n_qubits = None, n_layers = None, device = 'default.qubit',
                    ansatz_type = None, cost_type = None, # For 'smpi_iqaqe'.
                    shots = None, rs = None, B = None, basis_states_lists = None, encoding = None,
                    non_deterministic_CNOT = False, max_iter = 100, draw_plot = True,
                    repeat = 10, method = 'Avg_Bsf', diff_Rx = False, # 'diff_Rx' is only used in the 'ma_qaoa', 'exp_iqae' and 'smpi' ansatzë.
                    MaxCut = None, step_size = 0.99):
    """
    Possible values for 'vqa':
    - 'qaoa': Quantum Approximate Optimization Algorithm.
    - 'ma_qaoa': Multi-angle Quantum Approximate Optimization Algorithm.
    - 'qemc': Qubit-Efficient MaxCut Algorithm.
    - 'gw': Goemans-Williamson algorithm.
    - 'iqaqe': Interpolated qaoa/qemc;
    - 'exp_iqaqe': Exponential iqaqe;
    - 'smpi_iqaqe': Semi-Problem-Informed Ansatz iQAQE.
    """
    # Error handling
    assert(vqa is not None and n_layers is not None and MaxCut is not None), "vqa, n_layers and MaxCut cannot be None."
    assert(vqa == 'gw' or vqa == 'qaoa' or vqa == 'ma_qaoa' or vqa == 'qemc' or vqa == 'iqaqe' or vqa == 'exp_iqaqe' or vqa == 'smpi_iqaqe'), "Invalid 'vqa' algorithm."
    if(vqa == 'gw'):
        draw_plot = False # 'gw' is not a 'vqa', so we don't draw the plot.
    elif((vqa == 'iqaqe' or vqa == 'exp_iqaqe') and (basis_states_lists is None)):
        raise Exception("basis_states_lists cannot be None ('iqaqe' & 'exp_iqaqe').")
    elif((vqa == 'smpi_iqaqe') and (encoding is None)):
        raise Exception("encoding cannot be None ('smpi_iqaqe').")
    else:
        pass # Everything is fine.
    
    # Averaged over 'repeat' (defaults to 10) runs
    ar_vec, avg_train_time = [], 0
    cost_vec = [] if vqa != 'gw' else None

    for i_ in range(repeat):
        print(f"--- Run #{i_+1} of {repeat}: ---")
        match vqa:
            case 'qaoa':
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                bs.qaoa(vqa_graph_instance, n_layers = n_layers, shots = shots, device = device,
                     max_iter = max_iter, seed = seed,
                     draw_circuit = False, draw_cost_plot = False, draw_probs_plot = False,
                     MaxCut = MaxCut, step_size = step_size)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            case 'ma_qaoa':
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                bs.ma_qaoa(vqa_graph_instance, n_layers = n_layers, shots = shots, device = device,
                        max_iter = max_iter, seed = seed,
                        draw_circuit = False, draw_cost_plot = False, draw_probs_plot = False,
                        MaxCut = MaxCut, step_size = step_size, diff_Rx = diff_Rx)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()
            case 'qemc':
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                bs.qemc(vqa_graph_instance, n_layers = n_layers, shots = shots, device = device,
                     B = B,
                     max_iter = max_iter, seed = seed,
                     draw_circuit = False, draw_cost_plot = False, draw_nodes_probs_plot = False,
                     MaxCut = MaxCut, step_size = step_size)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()             
            case 'gw':
                _, _, approx_ratio, temp_train_time = bs.goemans_williamson(vqa_graph_instance, MaxCut)
                print(f"Approximation ratio: {approx_ratio}. (Cut = {approx_ratio*MaxCut}; MaxCut = {MaxCut}).")
                ar_vec.append(approx_ratio) # Remember, there is no 'cost_vec' for 'gw'.
                avg_train_time += temp_train_time; print()                
            case 'iqaqe':
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                iq.iqaqe(vqa_graph_instance, n_layers = n_layers, shots = shots, device = device,
                      basis_states_lists = basis_states_lists, rs = rs, non_deterministic_CNOT = non_deterministic_CNOT, B = B,
                      max_iter = max_iter, seed = seed,
                      draw_circuit = False, draw_cost_plot = False, draw_probs_plot = False, draw_nodes_probs_plot = False,
                      MaxCut = MaxCut, step_size = step_size)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()                
            case 'exp_iqaqe':
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                iq.exp_iqaqe(vqa_graph_instance, n_layers = n_layers, shots = shots, device = device,
                          basis_states_lists = basis_states_lists, diff_Rx = diff_Rx, B = B,
                          max_iter = max_iter, seed = seed,
                          draw_circuit = False, draw_cost_plot = False, draw_probs_plot = False, draw_nodes_probs_plot = False,
                          MaxCut = MaxCut, step_size = step_size)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()   
            case 'smpi_iqaqe':
                seed = np.random.randint(2**(32-1) - 1) # Random seed.
                _, _, _, _, cost_vec_tmp, ar_vec_tmp, _, _, temp_train_time = \
                iq.smpi_iqaqe(vqa_graph_instance, n_layers = n_layers, shots = shots, device = device,
                           ansatz_type = ansatz_type, cost_type = cost_type, encoding = encoding, B = B, rs = rs, diff_Rx = diff_Rx,
                           max_iter = max_iter, seed = seed,
                           draw_circuit = False, draw_cost_plot = False, draw_probs_plot = False, draw_nodes_probs_plot = False,
                           MaxCut = MaxCut, step_size = step_size, verbose = False)
                ar_vec.append(ar_vec_tmp); cost_vec.append(cost_vec_tmp)
                avg_train_time += temp_train_time; print()

    # Post-processing
    avg_train_time /= 10
    # Populating the min/max approximation ratio vector(s):
    min_ar_vec = np.min(ar_vec, axis = 0)
    max_ar_vec = np.max(ar_vec, axis = 0)

    # Compute the average approximation ratio vector and the standard deviation vector
    avg_vec = np.mean(ar_vec, axis = 0)
    std_vec = np.std(ar_vec, axis = 0)

    if(vqa != 'gw'):
        # First, we average the AR_Vec, then we compute the BSF metric: Best-so-far of the average.
        bsf_avg_vec = bsf_transform(avg_vec)

        # First, we compute the BSF metric, for each run, then we average them: Avg. best-so-far. I think that's what is done in the paper: https://arxiv.org/abs/2308.10383
        bsf_vec = [bsf_transform(ar_vec[i]) for i in range(repeat)]
        avg_bsf_vec = np.mean(bsf_vec, axis = 0)

        # Median best-so-far approximation ratio vector
        med_bsf_vec = np.median(bsf_vec, axis = 0)
    else:
        bsf_avg_vec, avg_bsf_vec, med_bsf_vec = None, None, None

    # Average cost vector. 'np.mean(cost_vec, axis = 0)' is the actual average.
    avg_cost_vec = np.mean(cost_vec, axis = 0) if vqa != 'gw' else None

    if(draw_plot):
        # Deciding which 'method' to plot.
        if(method == 'Bsf_Avg'):
            y = bsf_avg_vec
            label = f"Best-so-far Avg. approx. ratio (v{vqa} - a{ansatz_type} - c{cost_type})"
        elif(method == 'Avg_Bsf'):
            y = avg_bsf_vec
            label = f"Avg. best-so-far approx. ratio (v{vqa} - a{ansatz_type} - c{cost_type})" # This is the default.

        # Plot the avg. best-so-far approximation ratio
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(y) + 1), y, label=label)
        plt.legend()
        plt.xlabel("# Iterations")
        plt.ylabel(label)
        plt.title(label + f" evolution (MaxCut = {MaxCut})")
        if(vqa != 'smpi_iqaqe'):
            plt.savefig(f'{method}_v{vqa}_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
            print(f"[Info.] Plot saved as '{method}_v{vqa}_n{n_qubits}_p{n_layers}.png'.")
        else:
            # For 'smpi_iqaqe', we need to specify the 'ansatz_type' and 'cost_type'.
            # The 'encoding' isn't specified in the 'file_name', but it should be stored somewhere too.
            plt.savefig(f'{method}_v{vqa}_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png', dpi = 600, bbox_inches = 'tight')
            print(f"[Info.] Plot saved as '{method}_v{vqa}_a{ansatz_type}_c{cost_type}_n{n_qubits}_p{n_layers}.png'.")
        
    return ar_vec, med_bsf_vec, avg_bsf_vec, bsf_avg_vec, avg_vec, std_vec, avg_cost_vec, min_ar_vec, max_ar_vec, avg_train_time