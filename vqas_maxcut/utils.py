# Description: This file contains utility functions used in the 'variational-quantum-algorithms-maxcut' project.
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