import os
import glob
import datetime

import jax
import numpyro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hssm
import arviz as az
import seaborn as sns

import sqlite3

jax.config.update('jax_platform_name', 'cpu')
hssm.set_floatX("float32")
numpyro.set_host_device_count(14)

def subsitute_values_sequential(data,varname,new_values):
  unique_values = sorted(data[varname].unique())
  substitutions = {val: new_val for val, new_val in zip(unique_values,new_values)}
  return  data[varname].replace(substitutions)

def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / (mad if mad>0 else np.finfo(float).eps)

def fit_hssm_participant(df, participant_column):
    all_summaries = []
    all_inferences = {}   # <- store InferData here

    for nsub, isub in enumerate(df[participant_column].unique()):
        print(f"___Participant {isub}, {nsub+1}/{df[participant_column].nunique()}___")

        df_sub = df[df[participant_column] == isub].drop(columns=[participant_column])

        print("Median RT =", np.median(df_sub['rt']))
        print("N trials =", len(df_sub))

        model = hssm.HSSM(
            model="ddm",
            data=df_sub,
        )

        infer_data_sub = model.sample(
            cores=3,
            chains=3,
            draws=300,
            tune=1000,
            idata_kwargs=dict(log_likelihood=True),
            progressbar=True,
            target_accept=0.99,
        )

        all_inferences[isub] = infer_data_sub

        summary_df = (
            az.summary(infer_data_sub)
              .reset_index()
              .rename(columns={'index': 'param'})
        )
        summary_df['participant_id'] = isub
        all_summaries.append(summary_df)

    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    return all_summaries_df, all_inferences

def write_to_sql(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()

def write_summary_to_sql(df, db_path, table_name):
    df = df.copy()
    df["timestamp"] = datetime.now().isoformat()

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="append", index=False)



def get_fitted_participants(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        try:
            q = f"SELECT DISTINCT participant_id FROM {table_name}"
            return set(pd.read_sql(q, conn)['participant_id'])
        except Exception:
            return set()
        
def run_sequential_fits(
    df_hssm,
    participant_column,
    db_path,
    predictor='ab_nominal',
    use_log=False,
    max_participants=10,
    model_name=None,
):
    participants = df_hssm[participant_column].unique()

    models = {
        "ddm_mod_th":   fit_hssm_mod_th_single,
        "ddm_mod_v":    fit_hssm_mod_v_single,
    }

    # If model_name specified, only run that model
    if model_name:
        models = {model_name: models[model_name]}

    for table_name, fit_func in models.items():
        print(f"\n===== Running model: {table_name} =====")

        fitted = get_fitted_participants(db_path, table_name)
        remaining = [p for p in participants if p not in fitted]

        print(f"{len(remaining)} participants remaining")

        fitted_count = 0

        for i, pid in enumerate(remaining, 1):
            if fitted_count >= max_participants:
                print(
                    f"\n⏸️  Reached {max_participants} participants "
                    f"for model {table_name}. Re-run to continue."
                )
                break

            print(f"\n--- {table_name}: participant {pid} ({i}/{len(remaining)}) ---")

            try:
                summary_df = fit_func(
                    df=df_hssm,
                    participant_id=pid,
                    participant_column=participant_column,
                    predictor=predictor,
                    use_log=use_log,
                )

                write_summary_to_sql(
                    summary_df,
                    db_path=db_path,
                    table_name=table_name,
                )

                fitted_count += 1

            except Exception as e:
                print(f"❌ Failed participant {pid}: {e}")
                continue

def run_ddm_mod_v_sequential(
    df_hssm,
    participant_column,
    db_path,
    predictor="ab_nominal",
    use_log=False,
):
    participants = np.sort(df_hssm[participant_column].unique())
    fitted = set(get_fitted_participants(db_path, "ddm_mod_v"))
    remaining = [p for p in participants if p not in fitted]

    print(f"{len(remaining)} participants remaining")

    for i, pid in enumerate(remaining, 1):
        print(f"\n--- ddm_mod_v: participant {pid} ({i}/{len(remaining)}) ---")

        try:
            summary_df = fit_hssm_mod_v_single(
                df=df_hssm,
                participant_id=pid,
                participant_column=participant_column,
                predictor=predictor,
                use_log=use_log,
            )

            write_summary_to_sql(
                summary_df,
                db_path=db_path,
                table_name="ddm_mod_v",
            )

        except Exception as e:
            print(f"❌ Failed participant {pid}: {e}")

def get_fitted_parameters(df, participant_id, model):
    subset = df[df['participant_id'] == participant_id]
    z = subset[subset['param'] == 'z']['mean'].values[0]
    t = subset[subset['param'] == 't']['mean'].values[0]
    if model == 'pure':
        v = subset[subset['param'] == 'v']['mean'].values[0]
        a = subset[subset['param'] == 'a']['mean'].values[0]
        return v, a, z, t
    if model == 'th':
        v_int = subset[subset['param'] == 'v_Intercept']['mean'].values[0]
        a_int = subset[subset['param'] == 'a']['mean'].values[0]
        a_x = subset[subset['param'] == 'a_X']['mean'].values[0]
        return v_int, a_int, a_x, z, t
    if model == 'v':
        v_int = subset[subset['param'] == 'v_Intercept']['mean'].values[0]
        v_x = subset[subset['param'] == 'v_X']['mean'].values[0]
        a = subset[subset['param'] == 'a']['mean'].values[0]
        return v_int, v_x, a, z, t
    

def simulate_participant_ddm(participant_id, df, model='pure', size=300, bonus_prob=0.5):
    """
    Simulate DDM trials for a single participant.
    
    Parameters
    ----------
    participant_id : int or str
        Participant identifier.
    df : pd.DataFrame
        Fitted parameters dataframe (output of HSSM summary).
    model : str
        One of 'pure', 'v', 'th'.
    size : int
        Number of trials to simulate.
    bonus_prob : float
        Probability that a trial has the bonus (X=1). Only used for 'v' or 'th'.
        
    Returns
    -------
    pd.DataFrame
        Simulated dataset with RT, response, X, and participant_id.
    """

    # --- Get fixed parameters ---
    subset = df[df['participant_id'] == participant_id]
    z = subset[subset['param'] == 'z']['mean'].values[0]
    t = subset[subset['param'] == 't']['mean'].values[0]

    # --- Generate trial-level bonus coding ---
    X = np.random.binomial(1, bonus_prob, size) if model in ['v', 'th'] else np.zeros(size)

    # --- Assign v and a depending on model ---
    if model == 'pure':
        v = np.repeat(subset[subset['param'] == 'v']['mean'].values[0], size)
        a = subset[subset['param'] == 'a']['mean'].values[0]
    elif model == 'v':
        v_int = subset[subset['param'] == 'v_Intercept']['mean'].values[0]
        v_x = subset[subset['param'] == 'v_X']['mean'].values[0]
        v = v_int + X * v_x
        a = subset[subset['param'] == 'a']['mean'].values[0]
    elif model == 'th':
        v = subset[subset['param'] == 'v_Intercept']['mean'].values[0]
        a_int = subset[subset['param'] == 'a_Intercept']['mean'].values[0]
        a_x = subset[subset['param'] == 'a_X']['mean'].values[0]
        a = a_int + X * a_x
        print(a)
    else:
        raise ValueError("model must be one of 'pure', 'v', 'th'")

    # --- Stack parameters for HSSM ---
    true_values = np.column_stack([v, np.repeat([[a, z, t]], size, axis=0)])

    # --- Simulate data ---
    dataset = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # 1 trial per row
    )

    dataset["participant_id"] = str(participant_id)
    dataset["X"] = X  # optional, keeps track of bonus
    return dataset


def as_trialwise(x, size):
    return x if np.ndim(x) else np.full(size, x)

def simulate_participant_ddm(participant_id, df, model='pure', size=300, bonus_prob=0.5):

    subset = df[df['participant_id'] == participant_id]

    z = subset[subset['param'] == 'z']['mean'].values[0]
    t = subset[subset['param'] == 't']['mean'].values[0]

    # --- Generate trial-level bonus coding ---
    X = np.random.binomial(1, bonus_prob, size) if model in ['v', 'th'] else np.zeros(size)

    # --- Assign v and a depending on model ---
    if model == 'pure':
        v = subset[subset['param'] == 'v']['mean'].values[0]
        a = subset[subset['param'] == 'a']['mean'].values[0]

    elif model == 'v':
        v_int = subset[subset['param'] == 'v_Intercept']['mean'].values[0]
        v_x = subset[subset['param'] == 'v_X']['mean'].values[0]
        v = v_int + X * v_x
        a = subset[subset['param'] == 'a']['mean'].values[0]

    elif model == 'th':
        v = subset[subset['param'] == 'v_Intercept']['mean'].values[0]
        a_int = subset[subset['param'] == 'a_Intercept']['mean'].values[0]
        a_x = subset[subset['param'] == 'a_X']['mean'].values[0]
        a = a_int + X * a_x

    else:
        raise ValueError("model must be one of 'pure', 'v', 'th'")

    # --- Normalize to trial-wise (KEY FIX) ---
    v_vec = as_trialwise(v, size)
    a_vec = as_trialwise(a, size)
    z_vec = np.full(size, z)
    t_vec = np.full(size, t)

    # --- Stack parameters for HSSM ---
    true_values = np.column_stack([v_vec, a_vec, z_vec, t_vec])

    # --- Safety check (strongly recommended) ---
    assert true_values.shape == (size, 4)

    # --- Simulate data ---
    dataset = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,
    )

    dataset["participant_id"] = str(participant_id)
    dataset["X"] = X

    return dataset

def get_fitted_parameters(df, participant_id):
    subset = df[df['participant_id'] == participant_id]
    v = subset[subset['param'] == 'v']['mean'].values[0]
    a = subset[subset['param'] == 'a']['mean'].values[0]
    z = subset[subset['param'] == 'z']['mean'].values[0]
    t = subset[subset['param'] == 't']['mean'].values[0]
    return v, a, z, t

def simulate_participant(participant_id, df, size=300):
    v, a, z, t = get_fitted_parameters(df, participant_id)
    v = np.repeat(v, size)          # drift rate
    a = a                           # boundary
    z = z                           # starting point
    t = t                           # non-decision time
    true_values = np.column_stack([v, np.repeat([[a, z, t]], size, axis=0)])

    dataset = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,
    )

    dataset["participant_id"] = str(participant_id)
    return dataset

def fit_hssm_participant(df, participant_column):
    all_summaries = []
    all_inferences = {}   # <- store InferData here

    for nsub, isub in enumerate(df[participant_column].unique()):
        print(f"___Participant {isub}, {nsub+1}/{df[participant_column].nunique()}___")

        df_sub = df[df[participant_column] == isub].drop(columns=[participant_column])

        print("Median RT =", np.median(df_sub['rt']))
        print("N trials =", len(df_sub))

        model = hssm.HSSM(
            model="ddm",
            data=df_sub,
        )

        infer_data_sub = model.sample(
            cores=3,
            chains=3,
            draws=300,
            tune=1000,
            idata_kwargs=dict(log_likelihood=True),
            progressbar=True,
            target_accept=0.99,
        )

        all_inferences[isub] = infer_data_sub

        summary_df = (
            az.summary(infer_data_sub)
              .reset_index()
              .rename(columns={'index': 'param'})
        )
        summary_df['participant_id'] = isub
        all_summaries.append(summary_df)

    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    return all_summaries_df, all_inferences       




##### PLOTS ######
def plot_quants(p, df):
    p_1111 = df[df['participant_id'] == p]

    quantiles = np.linspace(0.1, 0.9, 9)
    rt_q = p_1111['rt'].quantile(quantiles)

    plt.plot(quantiles, rt_q.values, 'o-')
    plt.xlabel("Quantile")
    plt.ylabel("RT")
    plt.title(f"Participant {p} – RT quantiles")
    plt.show()


def plot_rt_distributions(data, responses=[0, 1], quantiles=[0.25, 0.5, 0.75], title="RT Distributions"):
    rt_correct = data.loc[data["response"] == responses[1], "rt"].values
    rt_error   = data.loc[data["response"] == responses[0], "rt"].values

    x = np.linspace(0, rt_correct.max(), 500)

    kde_c = gaussian_kde(rt_correct)
    kde_e = gaussian_kde(rt_error)

    yc = kde_c(x)
    ye = kde_e(x)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Correct (top)
    ax.fill_between(x, 0, yc, color="0.7")
    ax.plot(x, yc, color="black")

    # Error (bottom, mirrored)
    ax.fill_between(x, 0, -ye, color="0.7")
    ax.plot(x, -ye, color="black")

    # Zero line
    ax.axhline(0, color="black", lw=0.8)

    # --- Quantile ticks ---
    q_correct = np.quantile(rt_correct, quantiles)
    q_error   = np.quantile(rt_error, quantiles)

    tick_height = 0.25 * max(yc.max(), ye.max())

    for q in q_correct:
        ax.plot([q, q], [0, tick_height], color="black", lw=1)

    for q in q_error:
        ax.plot([q, q], [0, -tick_height], color="black", lw=1)
    # ----------------------

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density")

    ax.set_yticks([yc.max(), -ye.max()])
    ax.set_yticklabels(["Correct RTs", "Error RTs"])

    plt.title(title)

    plt.tight_layout()
    plt.show()

    print("Accuracy: ", len(rt_correct) / (len(rt_correct) + len(rt_error)))

def plot_rt_distributions_ppc(
    data, sim_data,
    responses=((0, 1), (-1, 1)),
    quantiles=[0.25, 0.5, 0.75],
    title="RT Distributions (PPC)"
):
    (err_d, cor_d), (err_s, cor_s) = responses

    # --- empirical ---
    rt_c = data.loc[data["response"] == cor_d, "rt"].values
    rt_e = data.loc[data["response"] == err_d, "rt"].values

    # --- simulated ---
    rt_c_sim = sim_data.loc[sim_data["response"] == cor_s, "rt"].values
    rt_e_sim = sim_data.loc[sim_data["response"] == err_s, "rt"].values

    x = np.linspace(
        0,
        max(rt_c.max(), rt_c_sim.max()),
        500
    )

    yc = gaussian_kde(rt_c)(x)
    ye = gaussian_kde(rt_e)(x)
    yc_sim = gaussian_kde(rt_c_sim)(x)
    ye_sim = gaussian_kde(rt_e_sim)(x)

    fig, ax = plt.subplots(figsize=(6, 4))

    # data
    ax.fill_between(x, 0, yc, alpha=0.3)
    ax.plot(x, yc, lw=2, label="Correct (data)")

    ax.fill_between(x, 0, -ye, alpha=0.3)
    ax.plot(x, -ye, lw=2, label="Error (data)")

    # sim
    ax.plot(x, yc_sim, lw=2, ls="--", label="Correct (sim)")
    ax.plot(x, -ye_sim, lw=2, ls="--", label="Error (sim)")

    ax.axhline(0, color="black", lw=0.8)

    # quartiles
    h = 0.25 * max(yc.max(), ye.max())

    for q in np.quantile(rt_c, quantiles):
        ax.plot([q, q], [0, h], lw=1)

    for q in np.quantile(rt_c_sim, quantiles):
        ax.plot([q, q], [0, h], lw=1, ls="--")

    for q in np.quantile(rt_e, quantiles):
        ax.plot([q, q], [0, -h], lw=1)

    for q in np.quantile(rt_e_sim, quantiles):
        ax.plot([q, q], [0, -h], lw=1, ls="--")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Density")
    ax.set_yticks([yc.max(), -ye.max()])
    ax.set_yticklabels(["Correct RTs", "Error RTs"])

    ax.legend(frameon=False)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    print("Accuracy (data):", len(rt_c) / (len(rt_c) + len(rt_e)))
    print("Accuracy (sim): ", len(rt_c_sim) / (len(rt_c_sim) + len(rt_e_sim)))


def plot_rt_quantiles_participant(
    participant_id,
    real_df,
    sim_df,
    rt_col="rt",
    pid_col="participant_id",
    quantiles=np.linspace(0.1, 0.9, 9)
):
    """
    Plot RT quantile curves for one participant:
    real data vs simulated data.
    """

    # --- standardize dtypes ---
    participant_id = str(participant_id)
    real_df = real_df.copy()
    sim_df = sim_df.copy()

    real_df[pid_col] = real_df[pid_col].astype(str)
    sim_df[pid_col] = sim_df[pid_col].astype(str)

    # --- subset participant ---
    real_p = real_df[real_df[pid_col] == participant_id]
    sim_p  = sim_df[sim_df[pid_col] == participant_id]

    if real_p.empty or sim_p.empty:
        print(f"Participant {participant_id} not found in one of the datasets")
        pass
        #raise ValueError(f"Participant {participant_id} not found in both datasets")

    # --- compute quantiles ---
    real_q = real_p[rt_col].quantile(quantiles).values
    sim_q  = sim_p[rt_col].quantile(quantiles).values

    # --- plot ---
    plt.figure(figsize=(5, 4))
    plt.plot(quantiles, real_q, 'o-', label="Experimental Data")
    plt.plot(quantiles, sim_q,  's--', label="Simulation")

    plt.xlabel("Quantile")
    plt.ylabel("Reaction time (sec)")
    plt.title(f"Participant {participant_id} – RT quantiles")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rt_quantiles_group(
    real_df,
    sim_df,
    rt_col="rt",
    pid_col="participant_id",
    quantiles=np.linspace(0.1, 0.9, 9),
    show_sem=True,
    title="Group-average RT quantiles"
):
    """
    Plot group-average RT quantile curves:
    real data vs simulated data.
    """

    # --- standardize dtypes ---
    real_df = real_df.copy()
    sim_df  = sim_df.copy()

    real_df[pid_col] = real_df[pid_col].astype(str)
    sim_df[pid_col]  = sim_df[pid_col].astype(str)

    # --- helper: participant-level quantiles ---
    def participant_quantiles(df):
        return df[rt_col].quantile(quantiles).values

    # --- real data ---
    real_Q = (
        real_df
        .groupby(pid_col)
        .apply(participant_quantiles)
        .to_list()
    )
    real_Q = np.vstack(real_Q)

    real_mean = real_Q.mean(axis=0)
    real_sem  = real_Q.std(axis=0) / np.sqrt(real_Q.shape[0])

    # --- simulated data ---
    sim_Q = (
        sim_df
        .groupby(pid_col)
        .apply(participant_quantiles)
        .to_list()
    )
    sim_Q = np.vstack(sim_Q)

    sim_mean = sim_Q.mean(axis=0)

    # --- plot ---
    plt.figure(figsize=(5, 4))

    plt.plot(quantiles, real_mean, 'o-', label="Data")
    if show_sem:
        plt.fill_between(
            quantiles,
            real_mean - real_sem,
            real_mean + real_sem,
            alpha=0.3
        )

    plt.plot(quantiles, sim_mean, 's--', label="DDM simulation")

    plt.errorbar(
    quantiles,
    real_mean,
    yerr=real_sem,
    fmt='none',
    capsize=3,
    alpha=0.7
)


    plt.xlabel("Quantile")
    plt.ylabel("Reaction time")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_parameter_recovery(recovery_df, param_name, param_label):
    """Plot parameter recovery for a specific parameter"""
    param_data = recovery_df[recovery_df['param'] == param_name].copy()

    if len(param_data) == 0:
        print(f"No data for parameter {param_name}")
        return

    # Calculate statistics
    corr = param_data['true_mean'].corr(param_data['recovered_mean'])
    rmse = np.sqrt(np.mean((param_data['true_mean'] - param_data['recovered_mean'])**2))
    bias = np.mean(param_data['recovered_mean'] - param_data['true_mean'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1.scatter(param_data['true_mean'], param_data['recovered_mean'],
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add identity line
    min_val = min(param_data['true_mean'].min(), param_data['recovered_mean'].min())
    max_val = max(param_data['true_mean'].max(), param_data['recovered_mean'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Identity line')

    # Add regression line
    z = np.polyfit(param_data['true_mean'], param_data['recovered_mean'], 1)
    p = np.poly1d(z)
    ax1.plot(param_data['true_mean'], p(param_data['true_mean']), 'b-', alpha=0.7, label='Regression line')

    ax1.set_xlabel(f'True {param_label}')
    ax1.set_ylabel(f'Recovered {param_label}')
    ax1.set_title(f'Parameter Recovery: {param_label}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density plots
    sns.kdeplot(data=param_data['true_mean'], ax=ax2, label='True', fill=True, alpha=0.5)
    sns.kdeplot(data=param_data['recovered_mean'], ax=ax2, label='Recovered', fill=True, alpha=0.5)
    ax2.set_xlabel(param_label)
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution Comparison: {param_label}')
    ax2.legend()

    plt.tight_layout()

    # Print statistics
    print(f"\n{param_label} Recovery Statistics:")
    print(f"  Correlation: {corr:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Bias: {bias:.3f}")
    print(f"  N: {len(param_data)}")

    plt.show()

# Plot recovery for each parameter
param_labels = {
    'v': 'Drift Rate (v)',
    'a': 'Boundary Separation (a)',
    'z': 'Starting Point (z)',
    't': 'Non-decision Time (t)'
}

for param, label in param_labels.items():
    plot_parameter_recovery(recovery_df, param, label)


##### MODELS ######
def fit_hssm_mod_th_v_single(
    df, participant_id, participant_column,
    predictor='ab_nominal', use_log=False
):
    df = df.copy()

    df['X'] = (df[predictor] == 10).astype(int)

    df_sub = (
        df[df[participant_column] == participant_id]
        .drop(columns=[participant_column])
    )

    print(f"___Participant {participant_id} | TH + V ___")
    print("Median RT =", np.median(df_sub['rt']))
    print("N trials =", len(df_sub))

    a_prior = {
        "Intercept": {"name": "Normal", "mu": 1.35, "sigma": 0.35},
        "X": {"name": "Normal", "mu": 0.0, "sigma": 0.25},
    }
    v_prior = {
        "Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22},
        "X": {"name": "Normal", "mu": 0.0, "sigma": 0.15},
    }

    model = hssm.HSSM(
        data=df_sub,
        model="ddm",
        include=[
            {"name": "a", "formula": "a ~ 1 + X", "prior": a_prior},
            {"name": "v", "formula": "v ~ 1 + X", "prior": v_prior},
        ],
    )

    idata = model.sample(
        cores=3,
        chains=3,
        draws=300,
        tune=1000,
        progressbar=True,
        target_accept=0.99,
    )

    summary_df = (
        az.summary(idata)
        .reset_index()
        .rename(columns={"index": "param"})
    )
    summary_df["participant_id"] = participant_id

    return summary_df

def fit_hssm_mod_th_single(
    df, participant_id, participant_column,
    predictor='ab_nominal', use_log=False
):
    df = df.copy()

    df['X'] = (df[predictor] == 10).astype(int)

    df_sub = (
        df[df[participant_column] == participant_id]
        .drop(columns=[participant_column])
    )

    print(f"___Participant {participant_id} | TH only ___")
    print("Median RT =", np.median(df_sub['rt']))
    print("N trials =", len(df_sub))

    a_prior = {
        "Intercept": {"name": "Normal", "mu": 1.35, "sigma": 0.35},
        "X": {"name": "Normal", "mu": 0.0, "sigma": 0.25},
    }
    v_prior = {
        "Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22},
    }

    model = hssm.HSSM(
        data=df_sub,
        model="ddm",
        include=[
            {"name": "a", "formula": "a ~ 1 + X", "prior": a_prior},
            {"name": "v", "formula": "v ~ 1", "prior": v_prior},
        ],
    )

    idata = model.sample(
        cores=3,
        chains=3,
        draws=300,
        tune=1000,
        progressbar=True,
        target_accept=0.99,
    )

    summary_df = (
        az.summary(idata)
        .reset_index()
        .rename(columns={"index": "param"})
    )
    summary_df["participant_id"] = participant_id

    return summary_df

def fit_hssm_mod_v_single(
    df, participant_id, participant_column,
    predictor='ab_nominal', use_log=False
):
    df = df.copy()

    df['X'] = (df[predictor] == 10).astype("float64")

    df_sub = (
        df[df[participant_column] == participant_id]
        .drop(columns=[participant_column])
    )

    print(f"___Participant {participant_id} | V only ___")
    print("Median RT =", np.median(df_sub['rt']))
    print("N trials =", len(df_sub))

    a_prior = {
        "Intercept": {"name": "Normal", "mu": 1.35, "sigma": 0.35},
    }
    v_prior = {
        "Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22},
        "X": {"name": "Normal", "mu": 0.0, "sigma": 0.15},
    }

    model = hssm.HSSM(
        data=df_sub,
        model="ddm",
        include=[
            {"name": "a", "formula": "a ~ 1", "prior": a_prior},
            {"name": "v", "formula": "v ~ 1 + X", "prior": v_prior},
        ],
    )

    idata = model.sample(
        cores=3,
        chains=3,
        draws=300,
        tune=1000,
        progressbar=True,
        target_accept=0.99,
    )

    summary_df = (
        az.summary(idata)
        .reset_index()
        .rename(columns={"index": "param"})
    )
    summary_df["participant_id"] = participant_id

    return summary_df

def fit_hssm_mod_v(
    df, participant_id, participant_column
):
    df = df.copy()

    df['X'] = df['ab_nominal_binary'].astype("float64")

    df = df.astype(np.float32)

    df_sub = (
        df[df[participant_column] == participant_id]
        .drop(columns=[participant_column])
    )

    print(f"___Participant {participant_id} | V only ___")
    print("Median RT =", np.median(df_sub['rt']))
    print("N trials =", len(df_sub))

    a_prior = {
        "Intercept": {"name": "Normal", "mu": 1.35, "sigma": 0.35},
    }
    v_prior = {
        "Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22},
        "X": {"name": "Normal", "mu": 0.0, "sigma": 0.15},
    }

    model = hssm.HSSM(
        data=df_sub,
        model="ddm",
        include=[
            {"name": "a", "formula": "a ~ 1", "prior": a_prior},
            {"name": "v", "formula": "v ~ 1 + X", "prior": v_prior},
        ],
    )

    idata = model.sample(
        cores=3,
        chains=3,
        draws=300,
        tune=1000,
        progressbar=True,
        target_accept=0.99,
    )

    summary_df = (
        az.summary(idata)
        .reset_index()
        .rename(columns={"index": "param"})
    )
    summary_df["participant_id"] = participant_id

    return summary_df


def fit_hssm_mod_v_single(
    df, participant_id, participant_column,
    predictor='ab_nominal', use_log=False
):
    df = df.copy()

    df['X'] = (df[predictor] == 10).astype("float64")

    df_sub = (
        df[df[participant_column] == participant_id]
        .drop(columns=[participant_column])
    )

    print(f"___Participant {participant_id} | V only ___")
    print("Median RT =", np.median(df_sub['rt']))
    print("N trials =", len(df_sub))
    v_prior = {
        "Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22},
        "X": {"name": "Normal", "mu": 0.0, "sigma": 0.15},
    }

    model = hssm.HSSM(
        data=df_sub,
        model="ddm",
        include=[
            {"name": "v", "formula": "v ~ 1 + X", "prior": v_prior},
        ],
    )

    idata = model.sample(
        cores=3,
        chains=3,
        draws=300,
        tune=1000,
        progressbar=True,
        target_accept=0.99,
    )

    summary_df = (
        az.summary(idata)
        .reset_index()
        .rename(columns={"index": "param"})
    )
    summary_df["participant_id"] = participant_id

    return summary_df #this is the one that worked for v