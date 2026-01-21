# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from scipy.sparse import isspmatrix


def auc_solam_zero_order(x_tr, y_tr, x_te, y_te, options):
    ids = options['ids']
    beta = options['beta']
    etas = options['etas']
    res_idx = options['res_idx']   
    n_tr, dim = x_tr.shape
    v = np.zeros(dim + 2)  # [w(dim), a(1), b(1)]
    alpha = 0  # Dual variable
    sp = 0     # Positive sample count
    t = 0      # Iteration number  
    
    # Zero-order method parameters
    mu1 = 1./(dim * dim * 1e4)
    mu2 = 1./(dim * dim * 1e4)
    K = options.get('K', int(dim))
    
    # Result storage
    n_idx = len(res_idx)
    ws = np.zeros((n_idx, dim + 3))
    gens = np.zeros(n_idx)
    i_res = 0
    gd = np.zeros(dim + 2)
    
    # Convert sparse matrix to dense matrix
    if isspmatrix(x_tr):
        x_tr = x_tr.toarray()  # Convert to dense matrix to speed up calculation

    # ===================== Objective Function =====================
    def hat_F_t(v, alpha, x, y, hat_p):
        if v.ndim == 2:  # Batch v input (K, dim_v)
            K_v = v.shape[0]
            w = v[:, :-2]  # (K_v, dim)
            a = v[:, -2]   # (K_v,)
            b = v[:, -1]   # (K_v,)
            w_dot_x = w.dot(x)  # (K_v,) Batch inner product calculation
        else:  # Single v input (dim_v,)
            w = v[:-2]  # (dim,)
            a = v[-2]   # Scalar
            b = v[-1]   # Scalar
            w_dot_x = np.inner(w, x)  # Scalar
        
        term1 = (1 - hat_p) * (w_dot_x - a)**2 if y == 1 else 0.0
        term2 = hat_p * (w_dot_x - b)** 2 if y == -1 else 0.0
        
        if y == -1:
            coeff = hat_p * w_dot_x
        else:
            coeff = -(1 - hat_p) * w_dot_x
            
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            term3 = 2 * (1 + alpha) * coeff
        else:
            term3 = 2 * (1 + alpha) * coeff
        
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            term4 = -hat_p * (1 - hat_p) * (alpha **2)
        else:
            term4 = -hat_p * (1 - hat_p) * (alpha** 2)
        
        return term1 + term2 + term3 + term4
    
    # ===================== v Gradient Estimation =====================
    def estimate_gradient_v(v, alpha, x, y, hat_p, mu1, K):
        dim_v = len(v)
        
        # Generate normalized random vectors
        U = np.random.normal(loc=0.0, scale=1.0, size=(K, dim_v))
        U_norm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-10  # Avoid division by zero
        U = U / U_norm
        
        v_tile = np.broadcast_to(v, (K, dim_v))
        
        # Perturbation (one-sided difference)
        V_plus = v_tile + mu1 * U
        V_minus = v_tile #- mu1 * U
        
        # Calculate function values
        f_plus = hat_F_t(V_plus, alpha, x, y, hat_p)  # (K,)
        f_minus = hat_F_t(V_minus, alpha, x, y, hat_p)   # (K,)
        
        # Gradient estimation
        terms = ((f_plus - f_minus) / ( mu1))[:, np.newaxis] * U * dim_v   # (K, dim_v)
        
        grad = np.mean(terms, axis=0)
        return grad
    
    # ===================== alpha Gradient Estimation =====================
    def estimate_gradient_alpha(v, alpha, x, y, hat_p, mu2, K):
        # Generate normalized random scalars
        U = np.random.normal(loc=0.0, scale=1.0, size=K)
        U = U / (np.linalg.norm(U) + 1e-10)  # Normalization 
        
        # Perturbation (one-sided difference)
        Alpha_plus = alpha + mu2 * U
        Alpha_minus = alpha #- mu2 * U
        
        # Batch calculate function values
        f_plus = hat_F_t(v, Alpha_plus, x, y, hat_p)  # (K,)
        f_minus = hat_F_t(v, Alpha_minus, x, y, hat_p)  # (K,)
        
        # One-sided difference gradient estimation
        terms = ((f_plus - f_minus ) / (mu2)) * U  # (K,)
        
        grad = np.mean(terms)
        return grad
    
    # ===================== Main Iteration Logic =====================
    while t < len(ids):
        x_t = x_tr[ids[t], :]
        y_t = y_tr[ids[t]]
        eta = etas[t]
        t += 1
        
        # Update positive sample probability estimation
        if y_t == 1:
            sp += 1
        hat_p = sp / t
        
        # Call optimized gradient estimation functions
        gd = estimate_gradient_v(v, alpha, x_t, y_t, hat_p, mu1, K)
        gd_alpha = estimate_gradient_alpha(v, alpha, x_t, y_t, hat_p, mu2, K)

        # Original parameter update logic
        v = v - eta * gd
        alpha = alpha + eta * gd_alpha        
        
        # Original L2 regularization logic
        v[:dim] = 1 / (1 + beta * eta) * v[:dim]
        w_ = v[:dim]
        
        # Original result storage logic
        if i_res < n_idx and res_idx[i_res] == t:                    
            if not np.all(np.isfinite(w_)):
                gens[i_res:] = gens[i_res - 1]    
                ws[i_res:, :] = v
                break
            # Calculate test set AUC
            pred_te = (x_te.dot(w_.T)).ravel()
            fpr_te, tpr_te, _ = metrics.roc_curve(y_te, pred_te, pos_label=1)               
            test_auc = metrics.auc(fpr_te, tpr_te)
            # Calculate training set AUC
            pred_tr = (x_tr.dot(w_.T)).ravel()
            fpr_tr, tpr_tr, _ = metrics.roc_curve(y_tr, pred_tr, pos_label=1)               
            train_auc = metrics.auc(fpr_tr, tpr_tr)   
            # Store results
            gens[i_res] = test_auc
            ws[i_res, :dim + 2] = v
            ws[i_res, dim + 2] = alpha
            i_res += 1
    
    return ws, gens