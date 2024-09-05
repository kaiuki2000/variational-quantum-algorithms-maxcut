# Finite 'shots' number.
import sys; sys.path.append("..") # Adds higher directory to python modules path.

from vqas_maxcut.version import __version__
from vqas_maxcut.vqa_graph import *
from vqas_maxcut.utils import build_qaoa_bsl, build_parity_bsl, build_qemc_bsl, \
    polynomial_compression_n_qubits, generate_polynomial_encoding, number_of_parameters, avg_best_so_far

import json

def print_version():
    print(f"\n*** Using 'vqas_maxcut' version: {__version__} ***\n")
    return None

if __name__ == "__main__":
    print_version()
    # Bigger 32-node graph, built using the Erdos-Renyi model
    n, p = 32, 0.2
    g = nx.erdos_renyi_graph(n, p, seed=0, directed=False)
    vqa_graph_instance = vqa_graph(g.edges(), n)
    # vqa_graph_instance.draw_graph()

    # number_of_parameters' function
    # We want an 'equal' number of parameters for each ansatz, so we set the number of layers accordingly. (Tuned manually.)
    print("'number_of_parameters' function:")

    # qemc: 3 * n_qubits * n_layers
    n_qubits_qemc = math.ceil(np.log2(n))
    n_layers_qemc = 7 # Verified manually.
    n_parameters_qemc = 3 * n_qubits_qemc * n_layers_qemc

    # smpi ansatz
    k = 3; n_qubits = polynomial_compression_n_qubits(vqa_graph_instance, k = k)
    print(f'Number of graph edges: {vqa_graph_instance.G.number_of_edges()}')
    print(f'Number of qubits after polynomial compression (of order k = {k}): {n_qubits}')
    n_parameters_sel       = number_of_parameters(vqa_graph_instance, ansatz_type='SEL', n_layers=5, k=k)
    n_parameters_brickwork = number_of_parameters(vqa_graph_instance, ansatz_type='Brickwork', n_layers=4, k=k)
    n_parameters_parity    = number_of_parameters(vqa_graph_instance, ansatz_type='Parity', n_layers=1, k=k, diff_Rx=True) # Note 'diff_Rx=True'.
    print(f"Number of parameters for the SEL ansatz (qemc): {n_parameters_qemc}")
    print(f"Number of parameters for the SEL ansatz (smpi): {n_parameters_sel}")
    print(f"Number of parameters for the Brickwork ansatz: {n_parameters_brickwork}")
    print(f"Number of parameters for the Parity ansatz: {n_parameters_parity}"); print()

    # (n_parameters, n_layers) for each ansatz
    d = {'SEL': (n_parameters_sel, 5),        \
    'Brickwork': (n_parameters_brickwork, 4), \
    'Parity': (n_parameters_parity, 1),        \
    'qemc': (n_parameters_qemc, n_layers_qemc)}

    print(f'Number of parameters & respective # of layers: {d}'); print()

    # Right now, this runs with a finite number of 'shots' (shots = 1024).

    # qemc
    print("qemc")
    qemc_ar_vec, qemc_med_bsf_vec, qemc_avg_bsf_vec, qemc_bsf_avg_vec, qemc_avg_vec, \
        qemc_std_vec, qemc_avg_cost_vec, qemc_min_ar_vec, qemc_max_ar_vec, qemc_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'qemc', n_qubits = n_qubits_qemc, n_layers = n_layers_qemc, device = 'lightning.qubit',
                        shots = 1024,
                        max_iter = 100, draw_plot = False,
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99); print()
    # smpi_iqaqe: infinite 'shots' number.
    print("smpi_iqaqe(sel_iqaqe)")
    smpi_sel_iqaqe_ar_vec, smpi_sel_iqaqe_med_bsf_vec, smpi_sel_iqaqe_avg_bsf_vec, smpi_sel_iqaqe_bsf_avg_vec, smpi_sel_iqaqe_avg_vec, \
        smpi_sel_iqaqe_std_vec, smpi_sel_iqaqe_avg_cost_vec, smpi_sel_iqaqe_min_ar_vec, smpi_sel_iqaqe_max_ar_vec, smpi_sel_iqaqe_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = n_qubits, n_layers = d['SEL'][1], device = 'lightning.qubit',
                        ansatz_type = 'SEL', cost_type = 'iQAQE', encoding = generate_polynomial_encoding(vqa_graph_instance, k = k), B = None, rs = None, diff_Rx = False,
                        max_iter = 100, draw_plot = False, shots = 1024,
                        repeat = 3, # Since we don't know the 'MaxCut' value, we set it to 1.
                        MaxCut = 1., step_size = 0.99); print()
    print("smpi_iqaqe(sel_expectation)")
    smpi_sel_expectation_ar_vec, smpi_sel_expectation_med_bsf_vec, smpi_sel_expectation_avg_bsf_vec, smpi_sel_expectation_bsf_avg_vec, smpi_sel_expectation_avg_vec, \
        smpi_sel_expectation_std_vec, smpi_sel_expectation_avg_cost_vec, smpi_sel_expectation_min_ar_vec, smpi_sel_expectation_max_ar_vec, smpi_sel_expectation_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = n_qubits, n_layers = d['SEL'][1], device = 'lightning.qubit',
                        ansatz_type = 'SEL', cost_type = 'Expectation', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 3), B = None, rs = None, diff_Rx = False,
                        max_iter = 100, draw_plot = False, shots = 1024,
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99); print()
    print("smpi_iqaqe(brickwork_iqaqe)")
    smpi_brickwork_iqaqe_ar_vec, smpi_brickwork_iqaqe_med_bsf_vec, smpi_brickwork_iqaqe_avg_bsf_vec, smpi_brickwork_iqaqe_bsf_avg_vec, smpi_brickwork_iqaqe_avg_vec, \
        smpi_brickwork_iqaqe_std_vec, smpi_brickwork_iqaqe_avg_cost_vec, smpi_brickwork_iqaqe_min_ar_vec, smpi_brickwork_iqaqe_max_ar_vec, smpi_brickwork_iqaqe_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = n_qubits, n_layers = d['Brickwork'][1], device = 'lightning.qubit',
                        ansatz_type = 'Brickwork', cost_type = 'iQAQE', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 3), B = None, rs = None, diff_Rx = False,
                        max_iter = 100, draw_plot = False, shots = 1024,
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99); print()
    print("smpi_iqaqe(brickwork_expectation)")
    smpi_brickwork_expectation_ar_vec, smpi_brickwork_expectation_med_bsf_vec, smpi_brickwork_expectation_avg_bsf_vec, smpi_brickwork_expectation_bsf_avg_vec, smpi_brickwork_expectation_avg_vec, \
        smpi_brickwork_expectation_std_vec, smpi_brickwork_expectation_avg_cost_vec, smpi_brickwork_expectation_min_ar_vec, smpi_brickwork_expectation_max_ar_vec, smpi_brickwork_expectation_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = n_qubits, n_layers = d['Brickwork'][1], device = 'lightning.qubit',
                        ansatz_type = 'Brickwork', cost_type = 'Expectation', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 3), B = None, rs = None, diff_Rx = False,
                        max_iter = 100, draw_plot = False, shots = 1024,
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99); print()
    print("smpi_iqaqe(parity_iqaqe)")
    smpi_parity_iqaqe_ar_vec, smpi_parity_iqaqe_med_bsf_vec, smpi_parity_iqaqe_avg_bsf_vec, smpi_parity_iqaqe_bsf_avg_vec, smpi_parity_iqaqe_avg_vec, \
        smpi_parity_iqaqe_std_vec, smpi_parity_iqaqe_avg_cost_vec, smpi_parity_iqaqe_min_ar_vec, smpi_parity_iqaqe_max_ar_vec, smpi_parity_iqaqe_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = n_qubits, n_layers = d['Parity'][1], device = 'lightning.qubit',
                        ansatz_type = 'Parity', cost_type = 'iQAQE', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 3), B = None, rs = None, diff_Rx = True,
                        max_iter = 100, draw_plot = False, shots = 1024,
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99); print()
    print("smpi_iqaqe(parity_expectation)")
    smpi_parity_expectation_ar_vec, smpi_parity_expectation_med_bsf_vec, smpi_parity_expectation_avg_bsf_vec, smpi_parity_expectation_bsf_avg_vec, smpi_parity_expectation_avg_vec, \
        smpi_parity_expectation_std_vec, smpi_parity_expectation_avg_cost_vec, smpi_parity_expectation_min_ar_vec, smpi_parity_expectation_max_ar_vec, smpi_parity_expectation_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'smpi_iqaqe', n_qubits = n_qubits, n_layers = d['Parity'][1], device = 'lightning.qubit',
                        ansatz_type = 'Parity', cost_type = 'Expectation', encoding = generate_polynomial_encoding(vqa_graph_instance, k = 3), B = None, rs = None, diff_Rx = True,
                        max_iter = 100, draw_plot = False, shots = 1024,
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99); print()
    # goemans_williamson
    print("goemans_williamson")
    gw_ar_vec, gw_med_bsf_vec, gw_avg_bsf_vec, gw_bsf_avg_vec, gw_avg_vec, \
        gw_std_vec, gw_avg_cost_vec, gw_min_ar_vec, gw_max_ar_vec, gw_avg_train_time = \
        avg_best_so_far(vqa_graph_instance, vqa = 'gw', n_layers = 0, # n_layers is not used here.
                        repeat = 3,
                        MaxCut = 1., step_size = 0.99) # No need to draw the plot here. (So, no 'print()'.)
    
    # Define the x-axis values (iterations)
    iterations = range(len(smpi_parity_expectation_med_bsf_vec))

    # Define the colors
    colors = ['#ff4538', '#ffda38', '#8fff38', '#38ff77', '#38f2ff', '#385dff', '#a838ff', '#ff38c0']

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the results
    ax.plot(iterations, smpi_sel_expectation_med_bsf_vec, label=f"c: Expectation, a: SEL ({d['SEL'][0]}, {d['SEL'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[0])
    ax.plot(iterations, smpi_brickwork_expectation_med_bsf_vec, label=f"c: Expectation, a: Brickwork ({d['Brickwork'][0]}, {d['Brickwork'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[1])
    ax.plot(iterations, smpi_parity_expectation_med_bsf_vec, label=f"c: Expectation, a: Parity ({d['Parity'][0]}, {d['Parity'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[2])
    ax.plot(iterations, smpi_sel_iqaqe_med_bsf_vec, label=f"c: iqaqe, a: SEL ({d['SEL'][0]}, {d['SEL'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[3])
    ax.plot(iterations, smpi_brickwork_iqaqe_med_bsf_vec, label=f"c: iQAQE, a: Brickwork ({d['Brickwork'][0]}, {d['Brickwork'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[4])
    ax.plot(iterations, smpi_parity_iqaqe_med_bsf_vec, label=f"c: iqaqe, a: Parity ({d['Parity'][0]}, {d['Parity'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[5])
    ax.plot(iterations, qemc_med_bsf_vec, label=f"qemc ({d['qemc'][0]}, {d['qemc'][1]})", linestyle = 'dashed', alpha = 0.85, color = colors[6])

    # Plot the 'gw' med_bsf with error bars
    ax.hlines(np.median(gw_ar_vec), 0, len(smpi_parity_expectation_med_bsf_vec), label = 'gw (Median. + Min./Max.)', color = colors[7], linestyle = 'dashed')
    ax.fill_between(iterations, gw_min_ar_vec, gw_max_ar_vec, alpha=0.3, color = colors[7]) # Min./Max. error bars.

    # Add labels and title
    ax.set_xlabel('# Iterations')
    ax.set_ylabel(r'Median $BSF$ Cut Value ($3$ repetitions)')
    # ax.set_ylim([0.4, 1.05]); ax.set_yticks(np.arange(0.4, 1.05, 0.1))
    ax.set_title(r'smpi: Cost, ansätze variations (shots = 1024, $32$-node graph, k = 3):'
                '\n'
                '(len(params), n_layers) = (x, y)')

    # Add legend
    ax.legend(loc = 'lower right')

    # Save the plot: 32-node graph, k = 3, repeat = 10, median BSF values.
    plt.savefig('smpi_32-node_graph_k3_r10_med_bsf_finite_shots.png', dpi = 600, bbox_inches = 'tight')
    print("[Info.] The plot has been saved as 'smpi_32-node_graph_k3_r10_med_bsf_finite_shots.png'.")

    # Show the plot
    # plt.show()

    # Now, save the numerical values.
    # The 'smpi' results
    smpi_results = {
        'smpi_sel_iqaqe': {'ar_vec': smpi_sel_iqaqe_ar_vec, 'med_bsf_vec': smpi_sel_iqaqe_med_bsf_vec, 'avg_bsf_vec': smpi_sel_iqaqe_avg_bsf_vec, \
        'bsf_avg_vec': smpi_sel_iqaqe_bsf_avg_vec, 'avg_vec': smpi_sel_iqaqe_avg_vec, 'std_vec': smpi_sel_iqaqe_std_vec, 'avg_cost_vec': smpi_sel_iqaqe_avg_cost_vec, \
        'min_ar_vec': smpi_sel_iqaqe_min_ar_vec, 'max_ar_vec': smpi_sel_iqaqe_max_ar_vec, 'avg_train_time': smpi_sel_iqaqe_avg_train_time},

        'smpi_sel_expectation': {'ar_vec': smpi_sel_expectation_ar_vec, 'med_bsf_vec': smpi_sel_expectation_med_bsf_vec, 'avg_bsf_vec': smpi_sel_expectation_avg_bsf_vec, \
        'bsf_avg_vec': smpi_sel_expectation_bsf_avg_vec, 'avg_vec': smpi_sel_expectation_avg_vec, 'std_vec': smpi_sel_expectation_std_vec, 'avg_cost_vec': smpi_sel_expectation_avg_cost_vec, \
        'min_ar_vec': smpi_sel_expectation_min_ar_vec, 'max_ar_vec': smpi_sel_expectation_max_ar_vec, 'avg_train_time': smpi_sel_expectation_avg_train_time},

        'smpi_brickwork_iqaqe': {'ar_vec': smpi_brickwork_iqaqe_ar_vec, 'med_bsf_vec': smpi_brickwork_iqaqe_med_bsf_vec, 'avg_bsf_vec': smpi_brickwork_iqaqe_avg_bsf_vec, \
        'bsf_avg_vec': smpi_brickwork_iqaqe_bsf_avg_vec, 'avg_vec': smpi_brickwork_iqaqe_avg_vec, 'std_vec': smpi_brickwork_iqaqe_std_vec, 'avg_cost_vec': smpi_brickwork_iqaqe_avg_cost_vec, \
        'min_ar_vec': smpi_brickwork_iqaqe_min_ar_vec, 'max_ar_vec': smpi_brickwork_iqaqe_max_ar_vec, 'avg_train_time': smpi_brickwork_iqaqe_avg_train_time},

        'smpi_brickwork_expectation': {'ar_vec': smpi_brickwork_expectation_ar_vec, 'med_bsf_vec': smpi_brickwork_expectation_med_bsf_vec, 'avg_bsf_vec': smpi_brickwork_expectation_avg_bsf_vec, \
        'bsf_avg_vec': smpi_brickwork_expectation_bsf_avg_vec, 'avg_vec': smpi_brickwork_expectation_avg_vec, 'std_vec': smpi_brickwork_expectation_std_vec, 'avg_cost_vec': smpi_brickwork_expectation_avg_cost_vec, \
        'min_ar_vec': smpi_brickwork_expectation_min_ar_vec, 'max_ar_vec': smpi_brickwork_expectation_max_ar_vec, 'avg_train_time': smpi_brickwork_expectation_avg_train_time},

        'smpi_parity_iqaqe': {'ar_vec': smpi_parity_iqaqe_ar_vec, 'med_bsf_vec': smpi_parity_iqaqe_med_bsf_vec, 'avg_bsf_vec': smpi_parity_iqaqe_avg_bsf_vec, \
        'bsf_avg_vec': smpi_parity_iqaqe_bsf_avg_vec, 'avg_vec': smpi_parity_iqaqe_avg_vec, 'std_vec': smpi_parity_iqaqe_std_vec, 'avg_cost_vec': smpi_parity_iqaqe_avg_cost_vec, \
        'min_ar_vec': smpi_parity_iqaqe_min_ar_vec, 'max_ar_vec': smpi_parity_iqaqe_max_ar_vec, 'avg_train_time': smpi_parity_iqaqe_avg_train_time},

        'smpi_parity_expectation': {'ar_vec': smpi_parity_expectation_ar_vec, 'med_bsf_vec': smpi_parity_expectation_med_bsf_vec, 'avg_bsf_vec': smpi_parity_expectation_avg_bsf_vec, \
        'bsf_avg_vec': smpi_parity_expectation_bsf_avg_vec, 'avg_vec': smpi_parity_expectation_avg_vec, 'std_vec': smpi_parity_expectation_std_vec, 'avg_cost_vec': smpi_parity_expectation_avg_cost_vec, \
        'min_ar_vec': smpi_parity_expectation_min_ar_vec, 'max_ar_vec': smpi_parity_expectation_max_ar_vec, 'avg_train_time': smpi_parity_expectation_avg_train_time},

        'qemc': {'ar_vec': qemc_ar_vec, 'med_bsf_vec': qemc_med_bsf_vec, 'avg_bsf_vec': qemc_avg_bsf_vec, \
        'bsf_avg_vec': qemc_bsf_avg_vec, 'avg_vec': qemc_avg_vec, 'std_vec': qemc_std_vec, 'avg_cost_vec': qemc_avg_cost_vec, \
        'min_ar_vec': qemc_min_ar_vec, 'max_ar_vec': qemc_max_ar_vec, 'avg_train_time': qemc_avg_train_time},

        'gw': {'ar_vec': gw_ar_vec, 'med_bsf_vec': gw_med_bsf_vec, 'avg_bsf_vec': gw_avg_bsf_vec, \
        'bsf_avg_vec': gw_bsf_avg_vec, 'avg_vec': gw_avg_vec, 'std_vec': gw_std_vec, 'avg_cost_vec': gw_avg_cost_vec, \
        'min_ar_vec': gw_min_ar_vec, 'max_ar_vec': gw_max_ar_vec, 'avg_train_time': gw_avg_train_time}
        }
    
    # Save the 'smpi' results
    np.save('smpi_32-node_graph_k3_r10_results_finite_shots.npy', smpi_results)
    print("[Info.] The 'smpi' results have been saved as 'smpi_32-node_graph_k3_r10_results_finite_shots.npy'.")

    # Read the 'smpi' results
    smpi_results = np.load('smpi_32-node_graph_k3_r10_results_finite_shots.npy', allow_pickle=True).item()
    print("[Info.] The 'smpi' results have been read from 'smpi_32-node_graph_k3_r10_results_finite_shots.npy'.")
    print("[Info.] The 'smpi' results are:")
    print(f"smpi_results: {smpi_results}."); print()
