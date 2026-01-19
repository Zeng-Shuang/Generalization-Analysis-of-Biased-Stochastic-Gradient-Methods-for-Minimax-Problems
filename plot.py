# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison():
    # 设置LaTeX字体和绘图风格
    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.tight_layout()
    
    # 定义文件信息：文件名、标签、颜色、线型
    file_info = [
        ('dna0.01SGDAcpu_testK=5e8lr1e-4_npasses=6.npy', 'SGDA', 'navy', '--o'),
        ('dna0.01Clipcpu_testK=5e8lr1e-4_npasses=6.npy', 'Clipped SGDA', 'maroon', '-s'),
        ('dna0.01Zerocpu_testK=5e8lr1e-4_npasses=6.npy', 'Zeroth-order SGDA', 'orange', '-.*')
    ]
    
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
            dist_diff = data['dist_diff']
            
            # 筛选前3个pass的数据
            mask = res_idx <= 3  # 只保留pass数不超过3的数据点
            res_idx = res_idx[mask]
            mean_vals = dist_diff['mean'][mask]
            std_vals = dist_diff['std'][mask]
            
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
    plt.xlabel('Number of Passes')
    plt.ylabel('Euclidean Distance')
    plt.legend(loc='upper left')
    #plt.semilogy()
    
    # 可以手动设置x轴范围，确保只显示到3
    plt.xlim(0, 3.1)  # 稍微超过3，让显示更美观
    
    # 保存并显示图形
    save_path = os.path.join('res', 'comparsion_dna_K=5e8_lr=1e-4.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == '__main__':
    plot_comparison()