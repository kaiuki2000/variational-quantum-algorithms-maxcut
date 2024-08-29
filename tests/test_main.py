import sys; sys.path.append("..") # Adds higher directory to python modules path.

from vqas_maxcut.version import __version__
from vqas_maxcut.vqa_graph import vqa_graph
from vqas_maxcut.vqas.base import *
from vqas_maxcut.vqas.iqaqe import iqaqe

def print_version():
    print(f"\n*** Using 'vqas_maxcut' version: {__version__} ***\n")
    return None

if __name__ == "__main__":
    print_version()
    n_nodes = 8
    graph   = [(0, 1), (0, 2), (0, 6), (1, 2), \
               (1, 6), (3, 2), (3, 4), (3, 5), \
               (4, 5), (4, 7), (5, 7), (6, 7)  ]
    vqa_graph_instance = vqa_graph(graph, n_nodes)

    # Test the QAOA function
    print("Testing the QAOA function:")
    qaoa_probabilities, qaoa_cost, qaoa_cost_vec, qaoa_ar_vec, qaoa_parameters, qaoa_partition, qaoa_train_time = \
    qaoa(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
        max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = 0,
        draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True,
        MaxCut = None, step_size = 0.99)
    print(f"Partition using QAOA: {qaoa_partition}. Cut = {vqa_graph_instance.compute_cut(qaoa_partition)}"); print()
    
    # Test the QEMC function
    print("Testing the QEMC function:")
    qemc_probabilities, qemc_cost, qemc_cost_vec, qemc_ar_vec, qemc_parameters, qemc_partition, qemc_train_time = \
    qemc(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
        B = 4,
        max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = 0,
        draw_circuit = True, draw_cost_plot = True, draw_nodes_probs_plot = True,
        MaxCut = None, step_size = 0.99)
    print(f"Partition using QEMC: {qemc_partition}. Cut = {vqa_graph_instance.compute_cut(qemc_partition)}"); print()

    # Test the ma-QAOA function
    print("Testing the ma-QAOA function:")
    ma_qaoa_probabilities, ma_qaoa_cost, ma_qaoa_cost_vec, ma_qaoa_ar_vec, ma_qaoa_parameters, ma_qaoa_partition, ma_qaoa_train_time = \
    ma_qaoa(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
        max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = 0,
        draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True,
        MaxCut = None, step_size = 0.99,
        diff_Rx = False)
    print(f"Partition using ma-QAOA: {ma_qaoa_partition}. Cut = {vqa_graph_instance.compute_cut(ma_qaoa_partition)}"); print()

    # Test the iQAQE function
    print("Testing the iQAQE function:")
    # Define the basis states lists (Chosen at random, here!)
    basis_states_lists = [['0001', '0011'], ['0101', '0111'], ['1001', '1011'], ['1101', '1111'], \
                          ['0000', '0010'], ['0100', '0110'], ['1000', '1010'], ['1100', '1110']]
    iqaqe_probabilities, iqaqe_nodes_probs, iqaqe_cost, iqaqe_cost_vec, iqaqe_ar_vec, iqaqe_parameters, iqaqe_partition, iqaqe_train_time = \
    iqaqe(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
        basis_states_lists = basis_states_lists, rs = None, non_deterministic_CNOT = False, B = None,
        max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = 0,
        draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True, draw_nodes_probs_plot = True,
        MaxCut = None, step_size = 0.99)
    print(f"Partition using iQAQE: {iqaqe_partition}. Cut = {vqa_graph_instance.compute_cut(iqaqe_partition)}"); print()
