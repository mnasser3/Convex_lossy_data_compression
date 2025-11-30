import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



#---------------------------------------------------
# core convex optimization routines

def convex_solve_rd(D_target,n,m,pX,Dmat):
    Q = cp.Variable((n, m), nonneg=True)     
    pY = cp.Variable(m, nonneg=True)
    constraints = [cp.sum(Q, axis=1) == 1]
    constraints += [pY == (pX@Q)]

    avg_distortion = cp.sum(cp.multiply(pX,cp.sum(cp.multiply(Q, Dmat), axis=1)))
    constraints += [avg_distortion <= D_target]

    I_terms = cp.rel_entr(Q, cp.vstack([pY] * n))
    mutual_info = cp.sum(cp.multiply(pX, cp.sum(I_terms, axis=1)))

    problem = cp.Problem(cp.Minimize(mutual_info), constraints)
    problem.solve()  

    return {
        "status": problem.status,
        "I_opt": mutual_info.value,
        "D_achieved": avg_distortion.value,
        "Q": Q.value,
        "pY": pY.value,
    }

def convex_solve_rd_with_fidelity(D_target,n,m,pX,Dmat,Dprime=None, subset_idx=None):
    Q = cp.Variable((n, m), nonneg=True)
    pY = cp.Variable(m, nonneg=True)
    constraints = [cp.sum(Q, axis=1) == 1]
    constraints += [pY == (pX@Q)]

    avg_distortion = cp.sum(cp.multiply(pX,cp.sum(cp.multiply(Q, Dmat), axis=1)))
    constraints += [avg_distortion <= D_target]
    
    if subset_idx is not None and Dprime is not None:
        subset_idx = np.array(subset_idx, dtype=int)
        px_subset = pX[subset_idx]
        Q_subset = Q[subset_idx, :]
        Dmat_subset = Dmat[subset_idx, :]
        avg_distortion_subset = cp.sum(cp.multiply(px_subset,cp.sum(cp.multiply(Q_subset, Dmat_subset), axis=1)))
        constraints += [avg_distortion_subset <= Dprime]

    I_terms = cp.rel_entr(Q, cp.vstack([pY] * n))
    mutual_info = cp.sum(cp.multiply(pX, cp.sum(I_terms, axis=1)))

    problem = cp.Problem(cp.Minimize(mutual_info), constraints)
    problem.solve()  

    res= {
        "status": problem.status,
        "I_opt": mutual_info.value,
        "D_achieved": avg_distortion.value,
        "Q": Q.value,
        "pY": pY.value,
        "dual_D_global": constraints[2].dual_value
    }
    if subset_idx is not None and Dprime is not None:
        res["dual_D_subset"] = constraints[-1].dual_value
    else:
        res["dual_D_subset"] = None
        
    return res


def convex_solve_rd_with_fidelity_tau(D_target, n, m, pX, Dmat,
                                      Dprime=None, subset_idx=None,
                                      tau=0.0):

    Q = cp.Variable((n, m), nonneg=True)
    pY = cp.Variable(m, nonneg=True)

    constraints = [cp.sum(Q, axis=1) == 1]          
    constraints += [pY == (pX @ Q)]                

    avg_distortion = cp.sum(cp.multiply(pX,
                                        cp.sum(cp.multiply(Q, Dmat), axis=1)))
    constr_D_global = (avg_distortion <= D_target)
    constraints += [constr_D_global]

    constr_D_subset = None
    if subset_idx is not None and Dprime is not None:
        subset_idx = np.array(subset_idx, dtype=int)
        px_subset = pX[subset_idx]
        Q_subset = Q[subset_idx, :]
        Dmat_subset = Dmat[subset_idx, :]

        avg_distortion_subset = cp.sum(
            cp.multiply(px_subset,
                        cp.sum(cp.multiply(Q_subset, Dmat_subset), axis=1))
        )
        constr_D_subset = (avg_distortion_subset <= Dprime)
        constraints += [constr_D_subset]

    I_terms = cp.rel_entr(Q, cp.vstack([pY] * n))
    mutual_info = cp.sum(cp.multiply(pX, cp.sum(I_terms, axis=1)))

    H_cond = cp.sum(cp.multiply(pX[:, None], cp.entr(Q)))


    objective = mutual_info - tau * H_cond
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    res = {
        "status": problem.status,
        "I_opt": mutual_info.value,          
        "D_achieved": avg_distortion.value,
        "Q": Q.value,
        "pY": pY.value,
        "dual_D_global": constr_D_global.dual_value,
        "H_cond": H_cond.value,
        "tau": tau,
    }
    if constr_D_subset is not None:
        res["dual_D_subset"] = constr_D_subset.dual_value
    else:
        res["dual_D_subset"] = None

    return res

#---------------------------------------------------
#updating codebook

def update_codebook_centroids(X, pX, Q, mass_tol=1e-12):
    if X.ndim == 1:
        X = X[:, None]   

    n, d = X.shape
    nQ, m = Q.shape
    assert n == nQ

    w = pX[:, None] * Q
    pY = w.sum(axis=0)         

    Y_new = np.zeros((m, d))
    for j in range(m):
        if pY[j] > mass_tol:
            Y_new[j] = (w[:, j][:, None] * X).sum(axis=0) / pY[j]
        else:
            Y_new[j] = 0.0

    return Y_new, pY

#merging
def merge_small_codewords(Y, pY, mass_threshold=1e-3, dist_threshold=1e-3):
    Y = np.array(Y, copy=True)
    pY = np.array(pY, copy=True)
    m = len(pY)
    keep = np.ones(m, dtype=bool)

    for j in range(m):
        if not keep[j]:
            continue
        
        if pY[j] < mass_threshold:
            cond = True
        else:
            cond = False

        if not cond:

            candidates = np.where(keep & (np.arange(m) != j))[0]
            if len(candidates) == 0:
                continue
            dists = np.linalg.norm(Y[candidates] - Y[j], axis=1)
            k = candidates[np.argmin(dists)]
            if dists.min() < dist_threshold:
                cond = True

        if cond:
            candidates = np.where(keep & (np.arange(m) != j))[0]
            if len(candidates) == 0:
                break
            dists = np.linalg.norm(Y[candidates] - Y[j], axis=1)
            k = candidates[np.argmin(dists)]
            total_mass = pY[k] + pY[j]
            if total_mass > 0:
                Y[k] = (pY[k]*Y[k] + pY[j]*Y[j]) / total_mass
                pY[k] = total_mass
            keep[j] = False

    return Y[keep], pY[keep]



#---------------------------------------------------
# main

def init_Q_nearest(xs, ys):
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)

    if xs_arr.ndim == 1:
        xs_arr = xs_arr[:, None]
    if ys_arr.ndim == 1:
        ys_arr = ys_arr[:, None]

    n = xs_arr.shape[0]
    m = ys_arr.shape[0]

    Dmat = squared_euclidean_Dmat(xs_arr, ys_arr)  
    nn_idx = np.argmin(Dmat, axis=1)               
    Q = np.zeros((n, m))
    Q[np.arange(n), nn_idx] = 1.0
    return Q, Dmat


def compute_metrics_from_Q(Q, pX, Dmat, eps=1e-12):
    Q_clipped = np.clip(Q, eps, 1.0)
    pY = pX @ Q_clipped

    pY_clipped = np.clip(pY, eps, 1.0)


    H_Y = -np.sum(pY_clipped * np.log(pY_clipped))
    H_Y_given_X = -np.sum(pX[:, None] * Q_clipped * np.log(Q_clipped))

    I_emp = H_Y - H_Y_given_X

    avg_distortion = np.sum(pX * np.sum(Q * Dmat, axis=1))

    return H_Y, H_Y_given_X, I_emp, avg_distortion

def run_main(ys, xs, pX, num_iters=10, tol=1e-6,
             Dprime=0.6, D_target=0.6, subset_idx=None):

    X = np.asarray(xs)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape

    Y_current = np.asarray(ys)
    if Y_current.ndim == 1:
        Y_current = Y_current[:, None]

    I_list = []
    D_list = []
    H_Y_list = []
    H_YX_list = []
    m_list = []

    Q0, Dmat0 = init_Q_nearest(X, Y_current)
    H_Y0, H_YX0, I0, D0 = compute_metrics_from_Q(Q0, pX, Dmat0)

    I_list.append(I0)
    D_list.append(D0)
    H_Y_list.append(H_Y0)
    H_YX_list.append(H_YX0)
    m_list.append(Y_current.shape[0])

    for it in range(num_iters):
        Dmat = squared_euclidean_Dmat(X, Y_current)

        res = convex_solve_rd_with_fidelity(
            D_target,
            n,
            Y_current.shape[0],
            pX,
            Dmat,
            Dprime=Dprime,
            subset_idx=subset_idx,
        )

        if res["status"] not in ["optimal", "optimal_inaccurate"]:
            break

        ents = compute_entropies_from_solution(res, pX)
        I_list.append(res["I_opt"])
        D_list.append(res["D_achieved"])
        H_Y_list.append(ents["H_Y"])
        H_YX_list.append(ents["H_Y_given_X"])

        Y_new, pY = update_codebook_centroids(X, pX, res["Q"])
        if Y_new.size == 0:
            break

        Y_new = np.asarray(Y_new)
        if Y_new.ndim == 1:
            Y_new = Y_new[:, None]

        pY = np.atleast_1d(pY.squeeze())

        Y_merged, pY_merged = merge_small_codewords(
            Y_new, pY, mass_threshold=1e-3, dist_threshold=1e-3
        )
        if Y_merged.size == 0:
            break

        Y_merged = np.asarray(Y_merged)
        if Y_merged.ndim == 1:
            Y_merged = Y_merged[:, None]

        m_prev = Y_current.shape[0]
        m_new = Y_merged.shape[0]
        m_min = min(m_prev, m_new)

        if m_min > 0:
            moves = np.linalg.norm(
                Y_merged[:m_min, :] - Y_current[:m_min, :],
                axis=1,
            )
            max_move = np.max(moves)
        else:
            max_move = 0.0

        Y_current = Y_merged
        m_list.append(Y_current.shape[0])

        if max_move < tol:
            break

    if d == 1:
        ys_final = Y_current.squeeze()
    else:
        ys_final = Y_current

    return ys_final, I_list, D_list, H_Y_list, H_YX_list, m_list

#---------------------------------------------------
# helpers
def squared_euclidean_Dmat(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    diff = X[:, None, :] - Y[None, :, :]
    return np.sum(diff**2, axis=2)

def mahalanobis_Dmat(X, Y, W):
    diff = X[:, None, :] - Y[None, :, :]
    tmp = diff @ W
    return np.sum(tmp * diff, axis=2)

def compute_entropies_from_solution(res, pX, eps=1e-12):
    Q = res["Q"]
    pY = res["pY"]
    Q_clipped = np.clip(Q, eps, 1.0)
    pY_clipped = np.clip(pY, eps, 1.0)
    H_Y = -np.sum(pY_clipped * np.log(pY_clipped))
    H_Y_given_X = -np.sum(pX[:, None] * Q_clipped * np.log(Q_clipped))
    I_emp = H_Y - H_Y_given_X

    return {
        "H_Y": H_Y,
        "H_Y_given_X": H_Y_given_X,
        "I_emp": I_emp,
    }


#---------------------------------------------------
# RD curve routines

def run_rd_curve(pX, Dmat, D_targets):
    n, m = Dmat.shape
    I_vals = []
    D_vals = []

    for D_target in D_targets:
        res = convex_solve_rd(D_target, n, m, pX, Dmat)
        if res["status"] not in ["optimal", "optimal_inaccurate"]:
            I_vals.append(np.nan)
            D_vals.append(np.nan)
        else:
            I_vals.append(res["I_opt"])
            D_vals.append(res["D_achieved"])

    return np.array(D_vals), np.array(I_vals)

def run_rd_curve_with_fidelity(pX, Dmat, D_targets, Dprime, subset_idx):
    n, m = Dmat.shape
    D_vals = []
    I_vals = []
    dual_global_vals = []
    dual_subset_vals = []

    for D_target in D_targets:
        res = convex_solve_rd_with_fidelity(D_target, n, m, pX, Dmat,
                                             subset_idx=subset_idx,
                                             Dprime=Dprime)
        if res["status"] not in ["optimal", "optimal_inaccurate"]:
            D_vals.append(np.nan)
            I_vals.append(np.nan)
            dual_global_vals.append(np.nan)
            dual_subset_vals.append(np.nan)
            continue

        D_vals.append(res["D_achieved"])
        I_vals.append(res["I_opt"])
        dual_global_vals.append(res["dual_D_global"])
        dual_subset_vals.append(res["dual_D_subset"])

    return (np.array(D_vals),
            np.array(I_vals),
            np.array(dual_global_vals),
            np.array(dual_subset_vals))
    
def run_tau_sensitivity(pX, Dmat, D_target, Dprime, subset_idx, taus):
    H_Y_list = []
    H_YX_list = []
    I_list = []
    n, m = Dmat.shape

    for tau in taus:
        res_tau = convex_solve_rd_with_fidelity_tau(
            D_target, n, m, pX, Dmat,
            Dprime=Dprime,
            subset_idx=subset_idx,
            tau=tau
        )
        ents = compute_entropies_from_solution(res_tau, pX)
        H_Y_list.append(ents["H_Y"])
        H_YX_list.append(ents["H_Y_given_X"])
        I_list.append(ents["I_emp"])
        
    return I_list, H_Y_list, H_YX_list
        
        
    

#---------------------------------------------------
# plotting routines

def plot_rd_curve(D_vals, I_vals, title="Rate–Distortion Curve"):
    R_bits = I_vals / np.log(2)
    plt.figure(figsize=(4, 3))
    plt.plot(D_vals, R_bits, marker='.', color='purple')
    plt.xlabel("Average distortion")
    plt.ylabel("Rate R(D) [bits/symbol]")
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def plot_rd_curve_with_fidelity(D_vals, I_vals, title="Rate–Distortion Curve with Fidelity Constraint"):
    R_bits = I_vals / np.log(2)
    plt.figure(figsize=(4, 3))
    plt.plot(D_vals, R_bits, marker='.', color='green')
    plt.xlabel("Average distortion")
    plt.ylabel("Rate R(D) [bits/symbol]")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_dual_sensitivity(D_vals, dual_global, dual_subset,
                          title="Dual sensitivity vs distortion"):
    plt.figure(figsize=(4, 3))
    plt.plot(D_vals, dual_global, marker='o', label=r"$\lambda_{\mathrm{global}}$")
    plt.plot(D_vals, dual_subset, marker='s', label=r"$\lambda_{\mathrm{subset}}$")
    plt.xlabel("Achieved average distortion")
    plt.ylabel("Dual value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_mains(I_list, D_list, H_Y_list, H_YX_list, m_list, D_target):
    iters = np.arange(len(I_list))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(iters, np.array(I_list) / np.log(2), label="I(X;Y) [bits]")
    axes[0].plot(iters, np.array(H_Y_list) / np.log(2), label="H(Y) [bits]")
    axes[0].plot(iters, np.array(H_YX_list) / np.log(2), label="H(Y|X) [bits]")
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel("Bits")
    axes[0].set_title("Info quantities vs iteration")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(iters, D_list,)
    axes[1].set_ylim([0, D_target+0.1*D_target])
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("Average distortion")
    axes[1].set_title("Distortion vs iteration")
    axes[1].grid(True)

    axes[2].step(iters, m_list, where='post')
    axes[2].set_xlabel("Outer iteration")
    axes[2].set_ylabel("Number of codewords m")
    axes[2].set_title("Codebook size vs iteration")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    
    
