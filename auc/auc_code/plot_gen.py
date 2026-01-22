import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


# log(x+1) transformation (to handle data containing 0 values)
def log1p_transform(x):
    return np.log1p(x)  # Use np.log1p, which is equivalent to log(x+1)

def expm1_transform(x):
    return np.expm1(x)  # Use np.expm1, which is equivalent to exp(x)-1

def plot_comparison(data_name, eta):
    tex_fonts = {
        "text.usetex": False,
        "font.family": "Arial", 
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    
    # Define file information: filename, label, color, line style
    file_info = [
       (f'{data_name}lr{eta}SGDA.npy', 'SGDA', 'navy', '--o'),
       (f'{data_name}lr{eta}Clip.npy', 'Clipped SGDA', 'maroon', '-s'),
       (f'{data_name}lr{eta}Zero.npy', 'Zeroth-order SGDA', 'orange', '-.*')
    ]
    
    #print(file_info)
    
    # Iterate through each file to plot the curves
    for filename, label, color, line_style in file_info:
        path = os.path.join('res', filename)
        if os.path.isfile(path):
            #print(path)
            # Load data
            data = np.load(path, allow_pickle=True).item()
            n_tr = data['n_tr']
            # Calculate the number of iterations (passes)
            res_idx = np.array(data['res_idx']) / n_tr
            # Get distance difference data (mean and standard deviation)
            gen_diff = data['gen_diff']
            
            # Filter data for the first 12 passes
            mask = res_idx <= 12  # Only keep data points with pass number not exceeding 12
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
    plt.ylabel('Test AUC', fontsize=18)  # Remove LaTeX syntax, use plain text directly
    plt.legend(loc='lower right', frameon=False, fontsize=19)
    plt.grid(linestyle='--', alpha=0.5)

    # Apply the corrected logarithmic transformation for the x-axis
    plt.xscale('function', functions=(log1p_transform, expm1_transform))
    plt.xticks([0.1, 1, 2, 5],  ['0.1', '1', '2', '5'], fontsize=15)
    plt.yticks(fontsize=15)
    
    # Save and display the figure
    save_path = os.path.join('res', f'comparsion_gen_K=d_{data_name}_eta={eta:.1e}_mu=1e-4.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability measured by Euclidean distance')
    parser.add_argument('-d', '--data', default='svmguide3', help='name of the dataset (default: svmguide3)')
    parser.add_argument('-eta', '--eta', default=0.01, type=float, help='step size (default: 1e-2)')

    args = parser.parse_args()

    plot_comparison(args.data, args.eta)