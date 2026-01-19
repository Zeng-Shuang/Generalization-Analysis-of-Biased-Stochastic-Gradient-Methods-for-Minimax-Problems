# -*- coding: utf-8 -*-
"""
GPU加速+分块版本说明：
1. 保持原GPU加速逻辑，新增分块策略处理大K场景
2. 将K拆分为多块小批量计算，避免单次显存溢出
3. 分块后梯度估计仍为无偏估计，不改变原算法收敛性
4. 支持通过block_size参数灵活控制单批次显存占用
"""
import torch
import numpy as np
from sklearn import metrics
from scipy.sparse import isspmatrix

# 确保使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def auc_solam_zero_order(x_tr, y_tr, x_te, y_te, options):
    # 参数解析（新增block_size：每块随机样本数，默认1e5，可根据GPU显存调整）
    ids = options['ids']
    beta = options['beta']
    etas = options['etas']
    res_idx = options['res_idx']   
    n_tr, dim = x_tr.shape
    
    # 零阶方法参数（新增block_size，默认1e5）
    mu1 = options.get('mu1', 1/(dim*dim*10000))
    mu2 = options.get('mu2', 1/(dim*dim*10000))
    K = options.get('K', int(1e8))
    block_size = options.get('block_size', int(3e6))  # 单块大小，可调整
    
    # 将数据转换为PyTorch张量并移至GPU
    if isspmatrix(x_tr):
        x_tr = x_tr.toarray()  # 转为稠密矩阵
    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    
    if isspmatrix(x_te):
        x_te = x_te.toarray()
    x_te = torch.tensor(x_te, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_te, dtype=torch.float32, device=device)
    
    # 初始化参数（在GPU上）
    v = torch.zeros(dim + 2, device=device)  # [w(dim), a(1), b(1)]
    alpha = torch.tensor(0.0, device=device)  # 对偶变量
    sp = 0     # 正样本计数
    t = 0      # 迭代次数  
    
    # 结果存储（在CPU上，用于最终输出）
    n_idx = len(res_idx)
    ws = np.zeros((n_idx, dim + 3))
    gens = np.zeros(n_idx)
    i_res = 0
    
    # 向量化目标函数：支持批量输入
    def hat_F_t(v, alpha, x, y, hat_p):
        """
        向量化实现的hat_F_t函数
        - 当v为(B, dim_v)时，返回(B,)的函数值数组（B为block_size）
        - 当alpha为(B,)时，返回(B,)的函数值数组
        - 否则返回标量
        """
        # 提取w, a, b（处理批量v的情况）
        if v.dim() == 2:  # 批量v输入 (B, dim_v)
            w = v[:, :-2]  # (B, dim)
            a = v[:, -2]   # (B,)
            b = v[:, -1]   # (B,)
            w_dot_x = torch.matmul(w, x)  # (B,) 内积批量计算
        else:  # 单个v输入 (dim_v,)
            w = v[:-2]  # (dim,)
            a = v[-2]   # 标量
            b = v[-1]   # 标量
            w_dot_x = torch.dot(w, x)  # 标量
        
        # 计算term1 (y=1时有效)
        term1 = (1 - hat_p) * (w_dot_x - a)**2 if y == 1 else torch.tensor(0.0, device=device)
        
        # 计算term2 (y=-1时有效)
        term2 = hat_p * (w_dot_x - b)**2 if y == -1 else torch.tensor(0.0, device=device)
        
        # 计算term3 (处理批量alpha的情况)
        if y == -1:
            coeff = hat_p * w_dot_x
        else:
            coeff = -(1 - hat_p) * w_dot_x
            
        if alpha.dim() == 1:  # 批量alpha
            term3 = 2 * (1 + alpha) * coeff
        else:  # 单个alpha
            term3 = 2 * (1 + alpha) * coeff
        
        # 计算term4
        if alpha.dim() == 1:  # 批量alpha
            term4 = -hat_p * (1 - hat_p) * (alpha**2)
        else:  # 单个alpha
            term4 = -hat_p * (1 - hat_p) * (alpha**2)
        
        return term1 + term2 + term3 + term4
    
    # 分块实现v的梯度估计（核心修改）
    def estimate_gradient_v(v, alpha, x, y, hat_p, mu1, K, block_size):
        dim_v = v.numel()
        total_gradient = torch.zeros(dim_v, device=device)  # 梯度累加器
        num_blocks = (K + block_size - 1) // block_size  # 向上取整计算总块数
        
        for block_idx in range(num_blocks):
            # 计算当前块的实际大小（最后一块可能小于block_size）
            current_block_size = min(block_size, K - block_idx * block_size)
            
            # 批量生成当前块的v副本 (current_block_size, dim_v)
            v_batch = v.repeat(current_block_size).view(current_block_size, dim_v)
            # 批量生成当前块的随机向量 (current_block_size, dim_v)
            U = torch.randn(current_block_size, dim_v, device=device)
            # 批量计算扰动后的v
            V_plus = v_batch + mu1 * U
            
            # 计算函数值
            f_orig = hat_F_t(v, alpha, x, y, hat_p)  # 标量
            f_plus = hat_F_t(V_plus, alpha, x, y, hat_p)  # (current_block_size,)
            
            # 计算当前块的梯度贡献并累加
            block_terms = ((f_plus - f_orig) / mu1).unsqueeze(1) * U  # (current_block_size, dim_v)
            total_gradient += block_terms.sum(dim=0)
        
        # 除以总样本数K，得到平均梯度（保持无偏性）
        return total_gradient / K
    
    # 分块实现alpha的梯度估计（核心修改）
    def estimate_gradient_alpha(v, alpha, x, y, hat_p, mu2, K, block_size):
        total_gradient = torch.tensor(0.0, device=device)  # 梯度累加器
        num_blocks = (K + block_size - 1) // block_size  # 向上取整计算总块数
        
        for block_idx in range(num_blocks):
            # 计算当前块的实际大小
            current_block_size = min(block_size, K - block_idx * block_size)
            
            # 批量生成当前块的随机标量 (current_block_size,)
            U = torch.randn(current_block_size, device=device)
            # 批量计算扰动后的alpha
            Alpha_plus = alpha + mu2 * U
            
            # 计算函数值
            f_orig = hat_F_t(v, alpha, x, y, hat_p)  # 标量
            f_plus = hat_F_t(v, Alpha_plus, x, y, hat_p)  # (current_block_size,)
            
            # 计算当前块的梯度贡献并累加
            block_terms = ((f_plus - f_orig) / mu2) * U  # (current_block_size,)
            total_gradient += block_terms.sum()
        
        # 除以总样本数K，得到平均梯度
        return total_gradient / K
    
    # 主迭代过程
    while t < len(ids):
        # 获取当前样本（在GPU上）
        print(f'{t}/{len(ids)}', end='\r')
        x_t = x_tr[ids[t], :]
        y_t = y_tr[ids[t]]
        eta = etas[t]
        t += 1
        
        # 更新正样本概率估计
        if y_t == 1:
            sp += 1
        hat_p = sp / t
        
        # 分块版零阶梯度估计（传入block_size参数）
        gd = estimate_gradient_v(v, alpha, x_t, y_t, hat_p, mu1, K, block_size)
        gd_alpha = estimate_gradient_alpha(v, alpha, x_t, y_t, hat_p, mu2, K, block_size)
        
        # 参数更新
        v = v - eta * gd
        alpha = alpha + eta * gd_alpha        
        
        # L2正则化
        v[:dim] = 1 / (1 + beta * eta) * v[:dim]
        w_ = v[:dim]
        
        # 结果存储
        if i_res < n_idx and res_idx[i_res] == t:                    
            # 检查是否有非有限值
            if not torch.all(torch.isfinite(w_)):
                gens[i_res:] = gens[i_res - 1] if i_res > 0 else 0.0
                ws[i_res:, :] = v.cpu().numpy()
                break
            
            # 计算测试集AUC（需要转移到CPU）
            pred_te = torch.matmul(x_te, w_).cpu().numpy().ravel()
            fpr_te, tpr_te, _ = metrics.roc_curve(y_te.cpu().numpy(), pred_te, pos_label=1)               
            test_auc = metrics.auc(fpr_te, tpr_te)
            
            # 计算训练集AUC
            pred_tr = torch.matmul(x_tr, w_).cpu().numpy().ravel()
            fpr_tr, tpr_tr, _ = metrics.roc_curve(y_tr.cpu().numpy(), pred_tr, pos_label=1)               
            train_auc = metrics.auc(fpr_tr, tpr_tr)   
            
            # 存储结果（转移到CPU）
            gens[i_res] = test_auc - train_auc
            ws[i_res, :dim + 2] = v.cpu().numpy()
            ws[i_res, dim + 2] = alpha.cpu().numpy()
            i_res += 1
    
    return ws, gens