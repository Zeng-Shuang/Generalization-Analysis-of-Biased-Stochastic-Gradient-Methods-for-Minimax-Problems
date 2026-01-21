import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


# 示例1：log(x+1) 变换（处理含0的数据）
def log1p_transform(x):
    return np.log1p(x)  # 修正：使用np.log1p，等价于log(x+1)

def expm1_transform(x):
    return np.expm1(x)  # 修正：使用np.expm1，等价于exp(x)-1

def plot_comparison(data_name, eta):
    # 设置绘图风格（关闭LaTeX，使用系统字体）
    tex_fonts = {
        "text.usetex": False,  # 关键修改：关闭LaTeX渲染
        "font.family": "Arial",  # 使用系统自带的Arial字体
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
       (f'{data_name}{eta*100:.1f}SGDAmu1e-4_version_1222.npy', 'SGDA', 'navy', '--o'),
       (f'{data_name}{eta*100:.1f}Clipmu1e-4_version_1222.npy', 'Clipped SGDA', 'maroon', '-s'),
       (f'{data_name}{eta*100:.1f}Zeromu1e-4_version_1222.npy', 'Zeroth-order SGDA', 'orange', '-.*')
    ]
    
    #print(file_info)
    
    # 遍历每个文件绘制曲线
    for filename, label, color, line_style in file_info:
        path = os.path.join('res', filename)
        if os.path.isfile(path):
            #print(path)
            # 加载数据
            data = np.load(path, allow_pickle=True).item()
            n_tr = data['n_tr']
            # 计算迭代次数（passes）
            res_idx = np.array(data['res_idx']) / n_tr
            # 获取距离差异数据（均值和标准差）
            gen_diff = data['gen_diff']
            
            # 筛选前12个pass的数据
            mask = res_idx <= 12  # 只保留pass数不超过12的数据点
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
    plt.ylabel('Test AUC', fontsize=18)  # 移除LaTeX语法，直接使用普通文本
    plt.legend(loc='lower right', frameon=False, fontsize=19)
    plt.grid(linestyle='--', alpha=0.5)

    # 应用修正后的x轴对数变换
    plt.xscale('function', functions=(log1p_transform, expm1_transform))
    plt.xticks([0.1, 1, 2, 5],  ['0.1', '1', '2', '5'], fontsize=15)
    plt.yticks(fontsize=15)
    
    # 保存并显示图形
    save_path = os.path.join('res', f'comparsion_gen_K=d_{data_name}_eta={eta:.1e}_mu=1e-4.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability measured by Euclidean distance')
    parser.add_argument('-d', '--data', default='svmguide3', help='name of the dataset (default: svmguide3)')
    parser.add_argument('-eta', '--eta', default=0.01, type=float, help='step size (default: 1e-2)')

    args = parser.parse_args()

    plot_comparison(args.data, args.eta)