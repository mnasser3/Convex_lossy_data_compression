import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

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
