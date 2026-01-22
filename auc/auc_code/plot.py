import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_comparison(data_name, eta):
    tex_fonts = {
        "text.usetex": False,
        "font.family": "serif",
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    
    # Define file information: filename, label, color, line style
    file_info = [
       (f'{data_name}lr{eta}SGDA.npy', 'SGDA', 'navy', '--o'),
       (f'{data_name}lr{eta}Clip.npy', 'Clipped SGDA', 'maroon', '-s'),
       (f'{data_name}lr{eta}Zero.npy', 'Zeroth-order SGDA', 'orange', '-.*')
    ]
    print(file_info)
    
    #print(file_info)
    
    # Iterate through each file to plot the curve
    for filename, label, color, line_style in file_info:
        path = os.path.join('res', filename)
        if os.path.isfile(path):
            # Load data
            data = np.load(path, allow_pickle=True).item()
            n_tr = data['n_tr']
            # Calculate the number of iterations (passes)
            res_idx = np.array(data['res_idx']) / n_tr
            # Get distance difference data (mean and standard deviation)
            #print(data.keys())
            gen_diff = data['dist_diff']
            
            # Filter data for passes no more than 6
            mask = res_idx <= 6 # Only keep data points with pass number ≤ 6
            res_idx = res_idx[mask]
            mean_vals = gen_diff['mean'][mask]
            std_vals = gen_diff['std'][mask]

            # Plot error bar chart (using filtered data)
            plt.errorbar(
                res_idx, 
                mean_vals, 
                yerr=std_vals * 0.3,  # Error bar length
                color=color, 
                linewidth=1.5,
                fmt=line_style, 
                capsize=3, 
                elinewidth=1, 
                markeredgewidth=1, 
                markersize=5,
                label=label
            )
    
    # Set axis labels and legend
    plt.xlabel('Number of Passes', fontsize=18)
    plt.ylabel(r'Euclidean Distance', fontsize=18)
    plt.legend(loc='lower right', frameon=False, fontsize=19)
    plt.grid(linestyle='--', alpha=0.5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # Set x-axis range
    plt.xlim(0,6.1)  

    plt.semilogy()
    # Save as PNG format
    save_path = os.path.join('res', f'comparsion_dis_K=d_{data_name}_eta={eta:.1e}_mu=1e-4.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability measured by Euclidean distance')
    parser.add_argument('-d', '--data', default='svmguide3', help='name of the dataset (default: svmguide3)')
    parser.add_argument('-eta', '--eta', default=0.01, type=float, help='step size (default: 1e-2)')

    args = parser.parse_args()

    plot_comparison(args.data, args.eta)