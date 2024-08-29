# Description: This file contains utility functions used in the 'variational-quantum-algorithms-maxcut' project.
import sys; sys.path.append("../..") # Adds higher, higher directory to python modules path.
from vqas_maxcut.vqa_graph import vqa_graph

import math
from scipy import optimize
from pennylane import numpy as np
import itertools

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
