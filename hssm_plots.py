import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def plot_quants(p, df):
    p_1111 = df[df['participant_id'] == p]
    quantiles = np.linspace(0.1, 0.9, 9)
    rt_q = p_1111['rt'].quantile(quantiles)
    plt.plot(quantiles, rt_q.values, 'o-')
    plt.xlabel("Quantile"); plt.ylabel("RT")
    plt.title(f"Participant {p} – RT quantiles")
    plt.show()

def plot_rt_distributions(data, responses=[0, 1], quantiles=[0.25, 0.5, 0.75], title="RT Distributions"):
    rt_correct = data.loc[data["response"] == responses[1], "rt"].values
    rt_error   = data.loc[data["response"] == responses[0], "rt"].values
    x = np.linspace(0, rt_correct.max(), 500)
    yc, ye = gaussian_kde(rt_correct)(x), gaussian_kde(rt_error)(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(x, 0, yc, color="0.7"); ax.plot(x, yc, color="black")
    ax.fill_between(x, 0, -ye, color="0.7"); ax.plot(x, -ye, color="black")
    ax.axhline(0, color="black", lw=0.8)
    
    h = 0.25 * max(yc.max(), ye.max())
    for q in np.quantile(rt_correct, quantiles): ax.plot([q, q], [0, h], color="black", lw=1)
    for q in np.quantile(rt_error, quantiles): ax.plot([q, q], [0, -h], color="black", lw=1)
    plt.title(title); plt.tight_layout(); plt.show()

def plot_rt_distributions_ppc(data, sim_data, responses=((0, 1), (-1, 1)), quantiles=[0.25, 0.5, 0.75], title="RT Distributions (PPC)"):
    (err_d, cor_d), (err_s, cor_s) = responses
    rt_c, rt_e = data.loc[data["response"] == cor_d, "rt"].values, data.loc[data["response"] == err_d, "rt"].values
    rt_c_s, rt_e_s = sim_data.loc[sim_data["response"] == cor_s, "rt"].values, sim_data.loc[sim_data["response"] == err_s, "rt"].values
    
    x = np.linspace(0, max(rt_c.max(), rt_c_s.max()), 500)
    yc, ye = gaussian_kde(rt_c)(x), gaussian_kde(rt_e)(x)
    yc_s, ye_s = gaussian_kde(rt_c_s)(x), gaussian_kde(rt_e_s)(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, yc, label="Correct (data)"); ax.plot(x, yc_s, ls="--", label="Correct (sim)")
    ax.plot(x, -ye, label="Error (data)"); ax.plot(x, -ye_s, ls="--", label="Error (sim)")
    ax.legend(); plt.title(title); plt.show()

def plot_rt_quantiles_participant(participant_id, real_df, sim_df, rt_col="rt", pid_col="participant_id", quantiles=np.linspace(0.1, 0.9, 9)):
    participant_id = str(participant_id)
    real_p = real_df[real_df[pid_col].astype(str) == participant_id]
    sim_p = sim_df[sim_df[pid_col].astype(str) == participant_id]
    
    plt.figure(figsize=(5, 4))
    plt.plot(quantiles, real_p[rt_col].quantile(quantiles), 'o-', label="Data")
    plt.plot(quantiles, sim_p[rt_col].quantile(quantiles), 's--', label="Sim")
    plt.title(f"Participant {participant_id}"); plt.legend(); plt.show()

def plot_rt_quantiles_group(real_df, sim_df, rt_col="rt", pid_col="participant_id", quantiles=np.linspace(0.1, 0.9, 9), show_sem=True, title="Group RT quantiles"):
    def get_q(df): return np.vstack(df.groupby(pid_col).apply(lambda x: x[rt_col].quantile(quantiles).values).to_list())
    
    real_Q, sim_Q = get_q(real_df), get_q(sim_df)
    r_mean, r_sem = real_Q.mean(axis=0), real_Q.std(axis=0) / np.sqrt(real_Q.shape[0])
    
    plt.plot(quantiles, r_mean, 'o-', label="Data")
    if show_sem: plt.fill_between(quantiles, r_mean-r_sem, r_mean+r_sem, alpha=0.3)
    plt.plot(quantiles, sim_Q.mean(axis=0), 's--', label="Sim")
    plt.legend(); plt.title(title); plt.show()

def plot_parameter_recovery(recovery_df, param_name, param_label):
    data = recovery_df[recovery_df['param'] == param_name]
    corr = data['true_mean'].corr(data['recovered_mean'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(data['true_mean'], data['recovered_mean'], edgecolors='black')
    sns.kdeplot(data=data['true_mean'], ax=ax2, label='True', fill=True)
    sns.kdeplot(data=data['recovered_mean'], ax=ax2, label='Recovered', fill=True)
    ax1.set_title(f"Recovery {param_label} (r={corr:.2f})")
    plt.tight_layout(); plt.show()