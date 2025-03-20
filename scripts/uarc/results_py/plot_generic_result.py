import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from get_configuration import get_configuration

def plot_generic_result(row_offset, column_offset, y_label, app_type, calculate_percentage):
    """
    Plot generic simulation results

    Args:
        row_offset (int): Row offset for reading data
        column_offset (int): Column offset for reading data
        y_label (str or list): Label for y-axis
        app_type (str): Type of application
        calculate_percentage (str): Method to calculate percentage
    """
    folder_path = get_configuration(1)
    num_of_simulations = get_configuration(3)
    step_of_x_axis = get_configuration(4)
    scenario_type = get_configuration(5)
    legends = get_configuration(6)
    start_of_mobile_device_loop = get_configuration(10)
    step_of_mobile_device_loop = get_configuration(11)
    end_of_mobile_device_loop = get_configuration(12)
    num_of_mobile_devices = int((end_of_mobile_device_loop - start_of_mobile_device_loop) / step_of_mobile_device_loop + 1)
    x_tick_label_coefficient = get_configuration(17)

    all_results = np.zeros((num_of_simulations, len(scenario_type), num_of_mobile_devices))
    min_results = np.zeros((len(scenario_type), num_of_mobile_devices))
    max_results = np.zeros((len(scenario_type), num_of_mobile_devices))

    for s in range(1, num_of_simulations + 1):
        for i in range(len(scenario_type)):
            for j in range(1, num_of_mobile_devices + 1):
                try:
                    mobile_device_number = start_of_mobile_device_loop + step_of_mobile_device_loop * (j - 1)
                    file_path = os.path.join(folder_path, f'ite{s}',
                                        f'SIMRESULT_TWO_TIER_WITH_EO_{scenario_type[i]}_{mobile_device_number}DEVICES_{app_type}_GENERIC.log')

                    # In Python, we'll use numpy's loadtxt instead of MATLAB's dlmread
                    # Note: row_offset and column_offset are adjusted for zero-indexing in Python
                    read_data = np.loadtxt(file_path, delimiter=';', skiprows=row_offset,usecols=True)
                    value = read_data[0, column_offset]

                    if calculate_percentage == 'percentage_for_all':
                        read_data = np.loadtxt(file_path, delimiter=';', skiprows=1)
                        total_task = read_data[0, 0] + read_data[0, 1]
                        value = (100 * value) / total_task
                    elif calculate_percentage == 'percentage_for_completed':
                        read_data = np.loadtxt(file_path, delimiter=';', skiprows=1)
                        total_task = read_data[0, 0]
                        value = (100 * value) / total_task
                    elif calculate_percentage == 'percentage_for_failed':
                        read_data = np.loadtxt(file_path, delimiter=';', skiprows=1)
                        total_task = read_data[0, 1]
                        value = (100 * value) / total_task

                    all_results[s-1, i, j-1] = value
                except Exception as err:
                    raise Exception(f"Error processing file: {err}")

    if num_of_simulations == 1:
        results = all_results
    else:
        results = np.mean(all_results, axis=0)  # Mean across simulations

    # No need for squeeze in Python as numpy handles dimensions differently

    for i in range(len(scenario_type)):
        for j in range(num_of_mobile_devices):
            x = all_results[:, i, j]  # Create Data
            SEM = np.std(x, ddof=1) / np.sqrt(len(x))  # Standard Error
            # t-score for 95% confidence interval
            ts = stats.t.ppf([0.05, 0.95], len(x) - 1)  # T-Score
            CI = np.mean(x) + ts * SEM  # Confidence Intervals

            if CI[0] < 0:
                CI[0] = 0

            if CI[1] < 0:
                CI[1] = 0

            min_results[i, j] = results[i, j] - CI[0]
            max_results[i, j] = CI[1] - results[i, j]

    types = np.zeros(num_of_mobile_devices)
    for i in range(num_of_mobile_devices):
        types[i] = start_of_mobile_device_loop + (i * step_of_mobile_device_loop)

    fig = plt.figure()
    pos = get_configuration(7)
    # Set figure size in inches (convert from cm)
    fig.set_size_inches(pos[2]/2.54, pos[3]/2.54)

    # Set default font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['text.fontsize'] = 12

    if get_configuration(20) == 1:  # If colorful plot
        for i in range(step_of_x_axis-1, num_of_mobile_devices, step_of_x_axis):
            x_index = start_of_mobile_device_loop + (i * step_of_mobile_device_loop)

            markers = get_configuration(50)
            for j in range(len(scenario_type)):
                plt.plot(x_index, results[j, i], markers[j], markerfacecolor=get_configuration(20+j+1),
                        color=get_configuration(20+j+1))

        for j in range(len(scenario_type)):
            if get_configuration(19) == 1:
                plt.errorbar(types, results[j, :], yerr=[min_results[j, :], max_results[j, :]],
                            fmt=':k', color=get_configuration(20+j+1), linewidth=1.5)
            else:
                plt.plot(types, results[j, :], ':k', color=get_configuration(20+j+1), linewidth=1.5)

        # Make plot background transparent
        ax = plt.gca()
        ax.set_facecolor('none')
    else:
        markers = get_configuration(40)
        for j in range(len(scenario_type)):
            if get_configuration(19) == 1:
                plt.errorbar(types, results[j, :], yerr=[min_results[j, :], max_results[j, :]],
                            fmt=markers[j], markerfacecolor='w', linewidth=1.2)
            else:
                plt.plot(types, results[j, :], markers[j], markerfacecolor='w', linewidth=1.2)

    lgnd = plt.legend(legends, loc='northwest')
    if get_configuration(20) == 1:
        lgnd.get_frame().set_facecolor('none')

    plt.axis('square')
    plt.xlabel(get_configuration(9))

    # Set x-ticks
    x_ticks = np.arange(
        start_of_mobile_device_loop * x_tick_label_coefficient,
        end_of_mobile_device_loop + 1,
        step_of_x_axis * step_of_mobile_device_loop * x_tick_label_coefficient
    )

    plt.xticks(x_ticks, x_ticks)

    # Set y-label (might be a list for multi-line label)
    if isinstance(y_label, list):
        plt.ylabel('\n'.join(y_label))
    else:
        plt.ylabel(y_label)

    plt.xlim([start_of_mobile_device_loop - 5, end_of_mobile_device_loop + 5])

    # Set font sizes
    plt.xlabel(plt.gca().get_xlabel(), fontsize=12)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=12)
    lgnd.set_fontsize(11)

    if get_configuration(18) == 1:
        # Save as PDF
        filename = os.path.join(folder_path, f'{row_offset}_{column_offset}_{app_type}.pdf')
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()