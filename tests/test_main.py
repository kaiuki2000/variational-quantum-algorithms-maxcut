import sys; sys.path.append("..") # Adds higher directory to python modules path.

from vqas_maxcut.version import __version__
from vqas_maxcut.vqa_graph import *
from vqas_maxcut.vqas.base import qaoa, qemc, ma_qaoa, goemans_williamson
from vqas_maxcut.vqas.iqaqe import iqaqe, exp_iqaqe, smpi_iqaqe
from vqas_maxcut.utils import build_qaoa_bsl, build_parity_bsl, build_qemc_bsl, \
    polynomial_compression_n_qubits, generate_polynomial_encoding, number_of_parameters, avg_best_so_far

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
    
    # # Bigger 32-node graph, built using the Erdos-Renyi model
    # n, p = 32, 0.02
    # g = nx.erdos_renyi_graph(n, p, seed=0, directed=False)
    # vqa_graph_instance = vqa_graph(g.edges(), n)
    # # vqa_graph_instance.draw_graph()

    # # Test the QAOA function
    # print("Testing the QAOA function:")
    # qaoa_probabilities, qaoa_cost, qaoa_cost_vec, qaoa_ar_vec, qaoa_parameters, qaoa_partition, qaoa_train_time = \
    # qaoa(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
    #     max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = 0,
    #     draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True,
    #     MaxCut = None, step_size = 0.99)
    # print(f"Partition using QAOA: {qaoa_partition}. Cut = {vqa_graph_instance.compute_cut(qaoa_partition)}"); print()
    
    # Test the QEMC function (Testing the 'consecutive_count' parameter)
    print("Testing the QEMC function:")
    qemc_probabilities, qemc_cost, qemc_cost_vec, qemc_ar_vec, qemc_parameters, qemc_partition, qemc_train_time = \
    qemc(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
        B = 4,
        max_iter = 1000, rel_tol = 0, abs_tol = 0.0001, parameters = None, seed = 0,
        draw_circuit = True, draw_cost_plot = True, draw_nodes_probs_plot = True,
        MaxCut = None, step_size = 0.99)
    print(f"Partition using QEMC: {qemc_partition}. Cut = {vqa_graph_instance.compute_cut(qemc_partition)}"); print()

    # # Test the ma-QAOA function
    # print("Testing the ma-QAOA function:")
    # ma_qaoa_probabilities, ma_qaoa_cost, ma_qaoa_cost_vec, ma_qaoa_ar_vec, ma_qaoa_parameters, ma_qaoa_partition, ma_qaoa_train_time = \
    # ma_qaoa(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
    #     max_iter = 100, rel_tol = 0, abs_tol = 0, parameters = None, seed = 0,
    #     draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True,
    #     MaxCut = None, step_size = 0.99,
    #     diff_Rx = False)
    # print(f"Partition using ma-QAOA: {ma_qaoa_partition}. Cut = {vqa_graph_instance.compute_cut(ma_qaoa_partition)}"); print()

    # # Test the Goemans-Williamson function
    # print("Testing the Goemans-Williamson function:")
    # gw_partition, gw_cut, gw_approx_ratio, gw_run_time = goemans_williamson(vqa_graph_instance)
    # print(f"Partition using Goemans-Williamson: {gw_partition}. Cut = {gw_cut}"); print()

    # # Test the iQAQE function
    # print("Testing the iQAQE function:")
    # # Define the basis states lists (Chosen at random, here!)
    # basis_states_lists = [['0001', '0011'], ['0101', '0111'], ['1001', '1011'], ['1101', '1111'], \
    #                       ['0000', '0010'], ['0100', '0110'], ['1000', '1010'], ['1100', '1110']]
    # iqaqe_probabilities, iqaqe_nodes_probs, iqaqe_cost, iqaqe_cost_vec, iqaqe_ar_vec, iqaqe_parameters, iqaqe_partition, iqaqe_train_time = \
    # iqaqe(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
    #     basis_states_lists = basis_states_lists, rs = None, non_deterministic_CNOT = False, B = None,
    #     max_iter = 1000, rel_tol = 0.001, abs_tol = 0, consecutive_count = 5, parameters = None, seed = 0,
    #     draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True, draw_nodes_probs_plot = True,
    #     MaxCut = None, step_size = 0.99)
    # print(f"Partition using iQAQE: {iqaqe_partition}. Cut = {vqa_graph_instance.compute_cut(iqaqe_partition)}"); print()

    # # Test the Exponential iQAQE function
    # print("Testing the Exponential iQAQE function: (In the QEMC limit.)")
    # # Define the basis states lists (Chosen to be the 'QEMC basis states lists', here!)
    # basis_states_lists = build_qemc_bsl(n_qubits = math.ceil(np.log2(vqa_graph_instance.n_nodes)))
    # exp_iqaqe_probabilities, exp_iqaqe_nodes_probs, exp_iqaqe_cost, exp_iqaqe_cost_vec, exp_iqaqe_ar_vec, exp_iqaqe_parameters, exp_iqaqe_partition, exp_iqaqe_train_time = \
    # exp_iqaqe(vqa_graph_instance, n_layers = 2, shots = None, device = 'default.qubit',
    #     basis_states_lists = basis_states_lists, diff_Rx = False, B = None,
    #     max_iter = 1000, rel_tol = 0, abs_tol = 0.0001, parameters = None, seed = 0,
    #     draw_circuit = True, draw_cost_plot = True, draw_probs_plot = True, draw_nodes_probs_plot = True,
    #     MaxCut = None, step_size = 0.59)
    # print(f"Partition using Exponential iQAQE: {exp_iqaqe_partition}. Cut = {vqa_graph_instance.compute_cut(exp_iqaqe_partition)}"); print()

    # # Test the semi-problem-informed-ansatz iQAQE function
    # print("Testing the semi-problem-informed-ansatz iQAQE function:")
    # k = 2
    # n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k) # Not really needed here, but can be used for other purposes.
    # print(f"Number of qubits after polynomial compression (of order k = {k}): {n_qubits}") # Debugging.
    # encoding = generate_polynomial_encoding(vqa_graph_instance, k)
    # print(f"Encoding: {encoding}")
    
    # smpi_iqaqe_probabilities, smpi_iqaqe_nodes_probs, smpi_iqaqe_exp_vals, smpi_iqaqe_cost, smpi_iqaqe_cost_vec, smpi_iqaqe_ar_vec, smpi_iqaqe_parameters, smpi_iqaqe_partition, smpi_iqaqe_train_time = \
    # smpi_iqaqe(vqa_graph_instance, n_layers = 20, shots = 1024, device = 'lightning.qubit', # Why does this take so long, with fiinite shots? I know it's supposed to take longer, but I feel like it takes too long, especially in comparison to the previous implementation.
    #     ansatz_type = 'Parity', cost_type = 'Expectation', encoding = encoding, B = None, rs = None, diff_Rx=False, diff_Rz=False, # The huge number of gates is the only thing that comes to mind. Do we need to run the circuit every single shot? If that were the case, that'd explain the long runtime. I should try to check if this is true.
    #     max_iter = 100, rel_tol = 0.001, abs_tol = 0, seed = 0,
    #     draw_circuit = False, draw_cost_plot = True, draw_probs_plot = True, draw_nodes_probs_plot = True,
    #     MaxCut = None, step_size = 0.09, verbose = False)
    # print(f"Partition using semi-problem-informed-ansatz iQAQE: {smpi_iqaqe_partition}. Cut = {vqa_graph_instance.compute_cut(smpi_iqaqe_partition)}"); print()

    # # Test the 'number_of_parameters' function
    # print("Testing the 'number_of_parameters' function:")
    # print(f'Number of graph edges: {vqa_graph_instance.G.number_of_edges()}')
    # k = 3; n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = k)
    # print(f'Number of qubits after polynomial compression (of order k = {k}): {n_qubits}')
    # n_parameters_brickwork = number_of_parameters(vqa_graph_instance, ansatz_type='Brickwork', n_layers=2, k=k)
    # n_parameters_sel = number_of_parameters(vqa_graph_instance, ansatz_type='SEL', n_layers=2, k=k)
    # n_parameters_parity = number_of_parameters(vqa_graph_instance, ansatz_type='Parity', n_layers=2, k=k, diff_Rx=True)
    # print(f"Number of parameters for the Brickwork ansatz: {n_parameters_brickwork}")
    # print(f"Number of parameters for the SEL ansatz: {n_parameters_sel}")
    # print(f"Number of parameters for the Parity ansatz: {n_parameters_parity}"); print()

    # # Test the 'avg_best_so_far' function
    # print("Testing the 'avg_best_so_far' function:")

    # # qaoa
    # print("qaoa")
    # qaoa_ar_vec, qaoa_med_bsf_vec, qaoa_avg_bsf_vec, qaoa_bsf_avg_vec, qaoa_avg_vec, \
    #     qaoa_std_vec, qaoa_avg_cost_vec, qaoa_min_ar_vec, qaoa_max_ar_vec, qaoa_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'qaoa', n_qubits = vqa_graph_instance.n_nodes, n_layers = 2, device = 'default.qubit',
    #                    shots = None,
    #                    max_iter = 100, draw_plot = True,
    #                    repeat = 3,
    #                    MaxCut = 10, step_size = 0.99); print()
    # # qemc
    # print("qemc")
    # qemc_ar_vec, qemc_med_bsf_vec, qemc_avg_bsf_vec, qemc_bsf_avg_vec, qemc_avg_vec, \
    #     qemc_std_vec, qemc_avg_cost_vec, qemc_min_ar_vec, qemc_max_ar_vec, qemc_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'qemc', n_qubits = math.ceil(np.log2(vqa_graph_instance.n_nodes)), n_layers = 2, device = 'default.qubit',
    #                    shots = None,
    #                    max_iter = 100, draw_plot = True,
    #                    repeat = 3,
    #                    MaxCut = 10, step_size = 0.99); print()
    # # ma_qaoa
    # print("ma_qaoa")
    # ma_qaoa_ar_vec, ma_qaoa_med_bsf_vec, ma_qaoa_avg_bsf_vec, ma_qaoa_bsf_avg_vec, ma_qaoa_avg_vec, \
    #     ma_qaoa_std_vec, ma_qaoa_avg_cost_vec, ma_qaoa_min_ar_vec, ma_qaoa_max_ar_vec, ma_qaoa_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'ma_qaoa', n_qubits = vqa_graph_instance.n_nodes, n_layers = 2, device = 'default.qubit',
    #                     shots = None,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3, diff_Rx = False,
    #                     MaxCut = 10, step_size = 0.99); print()
    # # goemans_williamson
    # print("goemans_williamson")
    # gw_ar_vec, gw_med_bsf_vec, gw_avg_bsf_vec, gw_bsf_avg_vec, gw_avg_vec, \
    #     gw_std_vec, gw_avg_cost_vec, gw_min_ar_vec, gw_max_ar_vec, gw_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'gw', n_layers = 0, # n_layers is not used here.
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99) # No need to draw the plot here. (So, no 'print()'.)
    # # iqaqe
    # print("iqaqe")
    # iqaqe_ar_vec, iqaqe_med_bsf_vec, iqaqe_avg_bsf_vec, iqaqe_bsf_avg_vec, iqaqe_avg_vec, \
    #     iqaqe_std_vec, iqaqe_avg_cost_vec, iqaqe_min_ar_vec, iqaqe_max_ar_vec, iqaqe_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'iqaqe', n_qubits = len(basis_states_lists[-1][-1]), n_layers = 2, device = 'default.qubit',
    #                     shots = None, rs = None, B = None, basis_states_lists = basis_states_lists,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # # exp_iqaqe
    # print("exp_iqaqe")
    # exp_iqaqe_ar_vec, exp_iqaqe_med_bsf_vec, exp_iqaqe_avg_bsf_vec, exp_iqaqe_bsf_avg_vec, exp_iqaqe_avg_vec, \
    #     exp_iqaqe_std_vec, exp_iqaqe_avg_cost_vec, exp_iqaqe_min_ar_vec, exp_iqaqe_max_ar_vec, exp_iqaqe_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'exp_iqaqe', n_qubits = math.ceil(np.log2(vqa_graph_instance.n_nodes)), n_layers = 2, device = 'default.qubit',
    #                     shots = None, rs = None, B = None, basis_states_lists = build_qemc_bsl(n_qubits = math.ceil(np.log2(vqa_graph_instance.n_nodes))),
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # # smpi_iqaqe
    # print("smpi_iqaqe(sel_iqaqe)")
    # smpi_sel_iqaqe_ar_vec, smpi_sel_iqaqe_med_bsf_vec, smpi_sel_iqaqe_avg_bsf_vec, smpi_sel_iqaqe_bsf_avg_vec, smpi_sel_iqaqe_avg_vec, \
    #     smpi_sel_iqaqe_std_vec, smpi_sel_iqaqe_avg_cost_vec, smpi_sel_iqaqe_min_ar_vec, smpi_sel_iqaqe_max_ar_vec, smpi_sel_iqaqe_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = 2), n_layers = 2, device = 'default.qubit',
    #                     ansatz_type = 'SEL', cost_type = 'iQAQE', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 2), B = None, rs = None, diff_Rx = False,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # print("smpi_iqaqe(sel_expectation)")
    # smpi_sel_expectation_ar_vec, smpi_sel_expectation_med_bsf_vec, smpi_sel_expectation_avg_bsf_vec, smpi_sel_expectation_bsf_avg_vec, smpi_sel_expectation_avg_vec, \
    #     smpi_sel_expectation_std_vec, smpi_sel_expectation_avg_cost_vec, smpi_sel_expectation_min_ar_vec, smpi_sel_expectation_max_ar_vec, smpi_sel_expectation_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = 2), n_layers = 2, device = 'default.qubit',
    #                     ansatz_type = 'SEL', cost_type = 'Expectation', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 2), B = None, rs = None, diff_Rx = False,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # print("smpi_iqaqe(brickwork_iqaqe)")
    # smpi_brickwork_iqaqe_ar_vec, smpi_brickwork_iqaqe_med_bsf_vec, smpi_brickwork_iqaqe_avg_bsf_vec, smpi_brickwork_iqaqe_bsf_avg_vec, smpi_brickwork_iqaqe_avg_vec, \
    #     smpi_brickwork_iqaqe_std_vec, smpi_brickwork_iqaqe_avg_cost_vec, smpi_brickwork_iqaqe_min_ar_vec, smpi_brickwork_iqaqe_max_ar_vec, smpi_brickwork_iqaqe_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = 2), n_layers = 2, device = 'default.qubit',
    #                     ansatz_type = 'Brickwork', cost_type = 'iQAQE', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 2), B = None, rs = None, diff_Rx = False,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # print("smpi_iqaqe(brickwork_expectation)")
    # smpi_brickwork_expectation_ar_vec, smpi_brickwork_expectation_med_bsf_vec, smpi_brickwork_expectation_avg_bsf_vec, smpi_brickwork_expectation_bsf_avg_vec, smpi_brickwork_expectation_avg_vec, \
    #     smpi_brickwork_expectation_std_vec, smpi_brickwork_expectation_avg_cost_vec, smpi_brickwork_expectation_min_ar_vec, smpi_brickwork_expectation_max_ar_vec, smpi_brickwork_expectation_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = 2), n_layers = 2, device = 'default.qubit',
    #                     ansatz_type = 'Brickwork', cost_type = 'Expectation', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 2), B = None, rs = None, diff_Rx = False,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # print("smpi_iqaqe(parity_iqaqe)")
    # smpi_parity_iqaqe_ar_vec, smpi_parity_iqaqe_med_bsf_vec, smpi_parity_iqaqe_avg_bsf_vec, smpi_parity_iqaqe_bsf_avg_vec, smpi_parity_iqaqe_avg_vec, \
    #     smpi_parity_iqaqe_std_vec, smpi_parity_iqaqe_avg_cost_vec, smpi_parity_iqaqe_min_ar_vec, smpi_parity_iqaqe_max_ar_vec, smpi_parity_iqaqe_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = 2), n_layers = 2, device = 'default.qubit',
    #                     ansatz_type = 'Parity', cost_type = 'iQAQE', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 2), B = None, rs = None, diff_Rx = True,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    # print("smpi_iqaqe(parity_expectation)")
    # smpi_parity_expectation_ar_vec, smpi_parity_expectation_med_bsf_vec, smpi_parity_expectation_avg_bsf_vec, smpi_parity_expectation_bsf_avg_vec, smpi_parity_expectation_avg_vec, \
    #     smpi_parity_expectation_std_vec, smpi_parity_expectation_avg_cost_vec, smpi_parity_expectation_min_ar_vec, smpi_parity_expectation_max_ar_vec, smpi_parity_expectation_avg_train_time = \
    #     avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = 2), n_layers = 2, device = 'default.qubit',
    #                     ansatz_type = 'Parity', cost_type = 'Expectation', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 2), B = None, rs = None, diff_Rx = True,
    #                     max_iter = 100, draw_plot = True,
    #                     repeat = 3,
    #                     MaxCut = 10, step_size = 0.99); print()
    
