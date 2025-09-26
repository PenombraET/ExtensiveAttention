import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sys import argv


def _flat_len_to_T(L):
    T = int((np.sqrt(1 + 8*L) - 1) // 2)
    return T

def sym(A_flat):

    if isinstance(A_flat, torch.Tensor):
        L = A_flat.numel()
        T = _flat_len_to_T(L)
        device = A_flat.device
        dtype = A_flat.dtype
        S = torch.zeros((T, T), device=device, dtype=dtype)
        iu = torch.triu_indices(T, T, offset=0, device=device)
        S[iu[0], iu[1]] = A_flat
        S[iu[1], iu[0]] = A_flat  
        return S
    else:
        L = len(A_flat)
        T = _flat_len_to_T(L)
        R = np.zeros((T, T), dtype=A_flat.dtype)
        c = 0
        for i in range(T):
            for j in range(i, T):
                v = A_flat[c]
                R[i, j] = v
                R[j, i] = v
                c += 1
        return R

def attention_from_scaled_S(x, S_scaled, beta=1.0, delta_in=0.0):

    N, L, D = x.shape
    logits = torch.einsum('nap,pq,nbq->nab', x, S_scaled, x) /np.sqrt(D)  
    trace_part = torch.diagonal(S_scaled).sum() /np.sqrt(D)         
    logits = logits - trace_part * torch.eye(L, device=x.device, dtype=x.dtype)

    if delta_in > 0.0:
        M = torch.full((L, L), 1.0/np.sqrt(2), device=x.device, dtype=x.dtype)
        M.diagonal().fill_(1)
        eps = torch.normal(0.0, 1.0, logits.shape, device=x.device, dtype=x.dtype)
        i, j = torch.triu_indices(row=L, col=L, offset=1, device=eps.device)
        eps[..., j, i] = eps[..., i, j]
        logits = logits + np.sqrt(delta_in) * eps * M

    return nn.Softmax(dim=-1)(beta * logits)


class NetS(nn.Module):
    def __init__(self, D, L, R_for_scale, beta=1.0, init_std=1.0):

        super().__init__()
        self.beta = beta
        self.D = D
        self.L = L
        self.R_for_scale = R_for_scale
        self.M = D*(D+1)//2
        self.s = nn.Parameter(torch.empty(self.M))
        nn.init.normal_(self.s, mean=0.0, std=init_std)

    def forward(self, x, delta_in):
        tilde_S = sym(self.s) 
        S_scaled = tilde_S / np.sqrt(self.R_for_scale * (self.D)) 
        return attention_from_scaled_S(x, S_scaled, beta=self.beta, delta_in=delta_in)


def make_teacher_S_scaled(D, R_star):

    B = torch.normal(0.0, 1.0, (D, R_star)) 
    tilde_S = B @ B.T                                                   
    S_scaled = tilde_S / np.sqrt(R_star * D)                             
    with torch.no_grad():
        iu = torch.triu_indices(D, D, offset=0)
        s_teacher_vec = tilde_S[iu[0], iu[1]].detach().cpu().numpy()
    return S_scaled, s_teacher_vec


def S_scaled_from_vec(s_vec, D, R_for_scale, device, dtype):
    s_t = torch.tensor(s_vec, device=device, dtype=dtype)
    tilde_S = sym(s_t)
    return tilde_S / np.sqrt(R_for_scale * D)

def S_MSE_scaled_from_vecs(s_student_vec, s_teacher_vec, D, R, R_star):

    S_student = sym(s_student_vec) / np.sqrt(R * D)
    S_teacher = sym(s_teacher_vec) / np.sqrt(R_star * D)
    return float(((S_student - S_teacher) ** 2).sum() / D)


def train_student_on_data_S(
    D, L, R, beta, lam, x_train, y_train,
    T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8
):

    student = NetS(D, L, R_for_scale=R, beta=beta, init_std=norm_init)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    prev_total_loss = None
    for t in range(T):
        optimizer.zero_grad()
        y_pred = student(x_train, delta_in=0.0)                
        data_loss = torch.sum((y_pred - y_train)**2)         
        tilde_S = sym(student.s)
        reg_loss = lam * (torch.sum(tilde_S**2) / np.sqrt(R * D))  
        total_loss = data_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        cur = float(total_loss.item())
        if prev_total_loss is not None and abs(cur - prev_total_loss) < tol and t > 100:
            break
        prev_total_loss = cur

    with torch.no_grad():
        y_pred_final    = student(x_train, delta_in=0.0)
        data_loss_final = torch.sum((y_pred_final - y_train)**2).item()
        tilde_S_final   = sym(student.s)
        reg_loss_final  = (lam * (torch.sum(tilde_S_final**2) / np.sqrt(R * D))).item()
        s_student_vec   = student.s.detach().cpu().numpy()

    return s_student_vec, data_loss_final, reg_loss_final


if __name__ == "__main__":
    D = 100
    L = 2

    rho      = 1.0    
    rho_star = 0.05  
    R        = int(rho * D)
    R_star   = int(rho_star * D)

    beta = beta_star = 1.0

    lam_list   = np.logspace(-6,-1,100)
    Delta_list = [0.0]
    Delta_in   = 0.05

    alpha_list = np.linspace(0.05, 0.5 , 64)
    task_id = int(argv[1]) if len(argv) > 1 else 0
    alpha = float(alpha_list[task_id])

    learning_rate = 0.1
    norm_init     = 1.0
    samples       = 8
    T             = 1000000
    tol           = 1e-7
    N_test        = 2000
    
    base_dir = "..." # Specify your base_dir
    os.makedirs(base_dir, exist_ok=True)


    for lam_cur in lam_list:
        for Delta_cur in Delta_list:

            sub_dir = os.path.join(base_dir, f"lam_{lam_cur:.6f}_Delta_in{Delta_in:.2f}_D{D}")
            os.makedirs(sub_dir, exist_ok=True)

            N = int(alpha * D**2)

            with torch.no_grad():
                S_teacher_scaled, s_teacher_vec = make_teacher_S_scaled(D, R_star)
            def teacher_forward(x, delta_in):
                return attention_from_scaled_S(x, S_teacher_scaled, beta=beta_star, delta_in=delta_in)

            MSE_runs = []
            label_err_runs = []
            train_data_runs = []
            train_reg_runs  = []
            total_loss_runs = []
            s_runs = []  

            for i in range(samples):
                x_train = torch.normal(0, 1, (N, L, D))
                with torch.no_grad():
                    y_train = teacher_forward(x_train, delta_in=Delta_in) 

                s_last, data_loss_i, reg_loss_i = train_student_on_data_S(
                    D, L, R, beta, lam_cur,
                    x_train, y_train,
                    T=T, learning_rate=learning_rate, norm_init=norm_init, tol=tol
                )
                s_runs.append(s_last)

                mse_i = S_MSE_scaled_from_vecs(s_last, s_teacher_vec, D, R, R_star)
                MSE_runs.append(mse_i)

                x_test = torch.normal(0, 1, (N_test, L, D))
                with torch.no_grad():
                    y_test_teacher = teacher_forward(x_test, delta_in=Delta_in)

                student_eval = NetS(D, L, R_for_scale=R, beta=beta, init_std=0.0)
                with torch.no_grad():
                    s_tensor = torch.tensor(s_last)
                    student_eval.s.copy_(s_tensor)
                    y_test_student = student_eval(x_test, delta_in=0.0)
                    label_err_i = torch.sum((y_test_student - y_test_teacher) ** 2).item()

                label_err_runs.append(label_err_i)
                train_data_runs.append(data_loss_i)
                train_reg_runs.append(reg_loss_i)
                total_loss_runs.append(data_loss_i + reg_loss_i)

            MSE_mean = float(np.mean(MSE_runs))
            print(MSE_mean)
            MSE_std  = float(np.std(MSE_runs, ddof=1)) if len(MSE_runs) > 1 else 0.0

            label_err_mean = float(np.mean(label_err_runs))
            label_err_std  = float(np.std(label_err_runs, ddof=1)) if len(label_err_runs) > 1 else 0.0

            train_data_mean = float(np.mean(train_data_runs))
            train_data_std  = float(np.std(train_data_runs, ddof=1)) if len(train_data_runs) > 1 else 0.0
            train_reg_mean  = float(np.mean(train_reg_runs))
            train_reg_std   = float(np.std(train_reg_runs, ddof=1)) if len(train_reg_runs) > 1 else 0.0

            train_total_mean = float(np.mean(total_loss_runs))
            train_total_std  = float(np.std(total_loss_runs, ddof=1)) if len(total_loss_runs) > 1 else 0.0

            fname = f"erm_alpha{alpha:.6f}_rho{rho:.2f}_Delta{Delta_cur:.2f}_lam{lam_cur:.6f}.csv"
            out_path = os.path.join(sub_dir, fname)

            row = {
                "alpha": alpha,
                "D": D,
                "L": L,
                "rho": rho,
                "rho_star": rho_star,
                "lambda": lam_cur,
                "Delta": Delta_in,
                "MSE_mean": MSE_mean,
                "MSE_std": MSE_std,
                "test_err_mean": label_err_mean,
                "test_err_std": label_err_std,
                "train_data_loss_mean": train_data_mean,
                "train_reg_loss_mean":  train_reg_mean,
                "train_total_loss_mean": train_total_mean,
                "train_total_loss_std": train_total_std,
            }

            M = D*(D+1)//2
            for run_idx, s_vec in enumerate(s_runs):
                for j in range(M):
                    row[f"s_run{run_idx}_{j}"] = float(s_vec[j])

            for j in range(M):
                row[f"s_teacher_{j}"] = float(s_teacher_vec[j])

            pd.DataFrame([row]).to_csv(out_path, index=False)
