import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_comparison(data_name, eta):
    # 关闭LaTeX渲染，改用matplotlib默认字体（核心修改）
    tex_fonts = {
        "text.usetex": False,  # 从True改为False
        "font.family": "serif",
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    
    # 定义文件信息：文件名、标签、颜色、线型
    file_info = [
       (f'{data_name}{eta*100:.1f}SGDAmu1e-4.npy', 'SGDA', 'navy', '--o'),
       (f'{data_name}{eta*100:.1f}Clipmu1e-4.npy', 'Clipped SGDA', 'maroon', '-s'),
       (f'{data_name}{eta*100:.1f}Zeromu1e-4.npy', 'Zeroth-order SGDA', 'orange', '-.*')
    ]
    
    #print(file_info)
    
    # 遍历每个文件绘制曲线
    for filename, label, color, line_style in file_info:
        path = os.path.join('res', filename)
        if os.path.isfile(path):
            # 加载数据
            data = np.load(path, allow_pickle=True).item()
            n_tr = data['n_tr']
            # 计算迭代次数（passes）
            res_idx = np.array(data['res_idx']) / n_tr
            # 获取距离差异数据（均值和标准差）
            #print(data.keys())
            gen_diff = data['dist_diff']
            
            # 筛选前3个pass的数据
            mask = res_idx <= 6 # 只保留pass数不超过3的数据点
            res_idx = res_idx[mask]
            mean_vals = gen_diff['mean'][mask]
            std_vals = gen_diff['std'][mask]

            # 绘制误差线图（使用筛选后的数据）
            plt.errorbar(
                res_idx, 
                mean_vals, 
                yerr=std_vals * 0.3,  # 误差线长度
                color=color, 
                linewidth=1.5,
                fmt=line_style, 
                capsize=3, 
                elinewidth=1, 
                markeredgewidth=1, 
                markersize=5,
                label=label
            )
    
    # 设置坐标轴标签和图例
    plt.xlabel('Number of Passes', fontsize=18)
    plt.ylabel(r'Euclidean Distance', fontsize=18)
    plt.legend(loc='lower right', frameon=False, fontsize=19)
    plt.grid(linestyle='--', alpha=0.5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # 设置x轴范围
    plt.xlim(0,6.1)  

    plt.semilogy()
    # 保存为PNG格式
    save_path = os.path.join('res', f'comparsion_dis_K=d_{data_name}_eta={eta:.1e}_mu=1e-4.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability measured by Euclidean distance')
    parser.add_argument('-d', '--data', default='svmguide3', help='name of the dataset (default: svmguide3)')
    parser.add_argument('-eta', '--eta', default=0.01, type=float, help='step size (default: 1e-2)')

    args = parser.parse_args()

    plot_comparison(args.data, args.eta)