import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sys import argv

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()
        self.beta = beta
        self.D = input_dim
        self.L = number_tokens
        self.R = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(0, norm)

    def forward(self, x, delta_in):
        x = self.fc1(x) / np.sqrt(self.D)                  
        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R) 
        trace_part = torch.norm(self.fc1.weight)**2 / np.sqrt(self.R * self.D**2)
        x = attention_matrix - trace_part * torch.eye(self.L, device=attention_matrix.device)
        if delta_in > 0.0:

            M = torch.full((self.L, self.L), 1.0/np.sqrt(2),
                       device=x.device, dtype=x.dtype) 
            M.diagonal().fill_(1)
            eps = torch.normal(0.0, 1.0, x.shape, device=x.device, dtype=x.dtype)
            i, j = torch.triu_indices(row=self.L, col=self.L, offset=1, device=eps.device)
            eps[..., j, i] = eps[..., i, j]

            x = x + np.sqrt(delta_in) * eps * M 

        x = nn.Softmax(dim=-1)(self.beta * x)
        return x


def train_student_on_data(
    D, L, R, beta, lam, x_train, y_train,
    T=1000, learning_rate=0.02, norm_init=1.0, tol=1e-8
):
    student = Net(D, R, L, norm=norm_init, beta=beta)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    prev_total_loss = None

    for t in range(T):
        optimizer.zero_grad()
        y_pred = student(x_train, delta_in=0.0)                     
        data_loss = torch.sum((y_pred - y_train)**2)       
        reg_loss = lam * torch.sum(student.fc1.weight ** 2) 
        total_loss = data_loss + reg_loss

        total_loss.backward()
        optimizer.step()

        cur = float(total_loss.item())
        if prev_total_loss is not None and abs(cur - prev_total_loss) < tol and t >100:
            break
        prev_total_loss = cur
    
    with torch.no_grad():
        y_pred_final      = student(x_train, delta_in=0.0)
        data_loss_final   = torch.sum((y_pred_final - y_train)**2).item()
        reg_loss_final    = (lam * torch.sum(student.fc1.weight ** 2)).item() 

    W_student = student.fc1.weight.detach().cpu().numpy() 
    return W_student, data_loss_final, reg_loss_final


def compute_S_from_W(W, R, D):
    # W: (R, D)
    return (W.T @ W) / np.sqrt(R * D) 

def S_MSE(W_student, W_teacher, R, R_star, D):
    S_stud   = compute_S_from_W(W_student, R, D)
    S_teach  = compute_S_from_W(W_teacher, R_star, D)
    return float(((S_stud - S_teach) ** 2).sum() / D)


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
    Delta_in = 0.05

    alpha_list =np.linspace(0.05, 0.5 , 64)
    task_id = int(argv[1]) if len(argv) > 1 else 0
    alpha = float(alpha_list[task_id])

    learning_rate = 0.1
    norm_init     = 1.0
    samples       = 8
    T             = 1000000
    tol=1e-7
    N_test=2000

    base_dir = "..." # Specity your base_dir


    os.makedirs(base_dir, exist_ok=True)

    for lam_cur in lam_list:
        for Delta_cur in Delta_list:

            sub_dir = os.path.join(base_dir, f"lam_{lam_cur:.6f}_Delta_in{Delta_in:.2f}_D{D}")
            os.makedirs(sub_dir, exist_ok=True)


            N = int(alpha * D**2)

            with torch.no_grad():
                teacher = Net(D, R_star, L, norm=1.0, beta=beta_star)

            W_teacher = teacher.fc1.weight.detach().cpu().numpy()  


            MSE_runs = []
            label_err_runs = [] 
            train_data_runs = []
            train_reg_runs = []
            total_loss_runs = []
            W_runs = []          

            for i in range(samples):
                x_train = torch.normal(0, 1, (N, L, D))
                with torch.no_grad():
                    y_train = teacher(x_train, delta_in=Delta_in) 

                W_last, data_loss_i, reg_loss_i = train_student_on_data(
                    D, L, R, beta, lam_cur,
                    x_train, y_train,
                    T=T, learning_rate=learning_rate, norm_init=norm_init, tol=tol
                )
                W_runs.append(W_last)

                mse_i = S_MSE(W_last, W_teacher, R, R_star, D)
                MSE_runs.append(mse_i)

                x_test = torch.normal(0, 1, (N_test, L, D))
                with torch.no_grad():
                    y_test_teacher = teacher(x_test, delta_in=Delta_in)  

                student_eval = Net(D, R, L, norm=0.0, beta=beta) 
                with torch.no_grad():
                    student_eval.fc1.weight.copy_(torch.tensor(W_last, dtype=student_eval.fc1.weight.dtype))
                    y_test_student = student_eval(x_test,delta_in=0.0)
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
            train_data_std = float(np.std(train_data_runs, ddof=1)) if len(train_data_runs) > 1 else 0.0
            train_reg_mean  = float(np.mean(train_reg_runs))
            train_data_std = float(np.std(train_data_runs, ddof=1)) if len(train_data_runs) > 1 else 0.0

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

            for run_idx, W in enumerate(W_runs):
                flat = W.T.reshape(-1)  # (D*R,)
                for j, v in enumerate(flat):
                    row[f"w_run{run_idx}_{j}"] = float(v)

            pd.DataFrame([row]).to_csv(out_path, index=False)

