import jax
import numpyro
import numpy as np
import pandas as pd
import hssm
import arviz as az
from utils import get_fitted_participants, write_summary_to_sql, as_trialwise

jax.config.update('jax_platform_name', 'cpu')
hssm.set_floatX("float32")
numpyro.set_host_device_count(14)

def fit_hssm_participant(df, participant_column):
    all_summaries = []
    all_inferences = {}
    for nsub, isub in enumerate(df[participant_column].unique()):
        print(f"___Participant {isub}, {nsub+1}/{df[participant_column].nunique()}___")
        df_sub = df[df[participant_column] == isub].drop(columns=[participant_column])
        
        model = hssm.HSSM(model="ddm", data=df_sub)
        infer_data_sub = model.sample(
            cores=3, chains=3, draws=300, tune=1000,
            idata_kwargs=dict(log_likelihood=True),
            progressbar=True, target_accept=0.99,
        )
        all_inferences[isub] = infer_data_sub
        summary_df = az.summary(infer_data_sub).reset_index().rename(columns={'index': 'param'})
        summary_df['participant_id'] = isub
        all_summaries.append(summary_df)
    return pd.concat(all_summaries, ignore_index=True), all_inferences

def fit_hssm_mod_th_v_single(df, participant_id, participant_column, predictor='ab_nominal', use_log=False):
    df = df.copy()
    df['X'] = (df[predictor] == 10).astype(int)
    df_sub = df[df[participant_column] == participant_id].drop(columns=[participant_column])
    
    a_prior = {"Intercept": {"name": "Normal", "mu": 1.35, "sigma": 0.35}, "X": {"name": "Normal", "mu": 0.0, "sigma": 0.25}}
    v_prior = {"Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22}, "X": {"name": "Normal", "mu": 0.0, "sigma": 0.15}}
    
    model = hssm.HSSM(data=df_sub, model="ddm", include=[
        {"name": "a", "formula": "a ~ 1 + X", "prior": a_prior},
        {"name": "v", "formula": "v ~ 1 + X", "prior": v_prior},
    ])
    idata = model.sample(cores=3, chains=3, draws=300, tune=1000, progressbar=True, target_accept=0.99)
    summary_df = az.summary(idata).reset_index().rename(columns={"index": "param"})
    summary_df["participant_id"] = participant_id
    return summary_df

def fit_hssm_mod_th_single(df, participant_id, participant_column, predictor='ab_nominal', use_log=False):
    df = df.copy()
    df['X'] = (df[predictor] == 10).astype(int)
    df_sub = df[df[participant_column] == participant_id].drop(columns=[participant_column])
    
    a_prior = {"Intercept": {"name": "Normal", "mu": 1.35, "sigma": 0.35}, "X": {"name": "Normal", "mu": 0.0, "sigma": 0.25}}
    v_prior = {"Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22}}
    
    model = hssm.HSSM(data=df_sub, model="ddm", include=[
        {"name": "a", "formula": "a ~ 1 + X", "prior": a_prior},
        {"name": "v", "formula": "v ~ 1", "prior": v_prior},
    ])
    idata = model.sample(cores=3, chains=3, draws=300, tune=1000, progressbar=True, target_accept=0.99)
    summary_df = az.summary(idata).reset_index().rename(columns={"index": "param"})
    summary_df["participant_id"] = participant_id
    return summary_df

def fit_hssm_mod_v_single(df, participant_id, participant_column, predictor='ab_nominal', use_log=False):
    df = df.copy()
    df['X'] = (df[predictor] == 10).astype("float64")
    df_sub = df[df[participant_column] == participant_id].drop(columns=[participant_column])
    v_prior = {"Intercept": {"name": "Normal", "mu": 0.45, "sigma": 0.22}, "X": {"name": "Normal", "mu": 0.0, "sigma": 0.15}}
    model = hssm.HSSM(data=df_sub, model="ddm", include=[{"name": "v", "formula": "v ~ 1 + X", "prior": v_prior}])
    idata = model.sample(cores=3, chains=3, draws=300, tune=1000, progressbar=True, target_accept=0.99)
    summary_df = az.summary(idata).reset_index().rename(columns={"index": "param"})
    summary_df["participant_id"] = participant_id
    return summary_df

def run_sequential_fits(df_hssm, participant_column, db_path, predictor='ab_nominal', use_log=False, max_participants=10, model_name=None):
    participants = df_hssm[participant_column].unique()
    models = {"ddm_mod_th": fit_hssm_mod_th_single, "ddm_mod_v": fit_hssm_mod_v_single}
    if model_name:
        models = {model_name: models[model_name]}

    for table_name, fit_func in models.items():
        fitted = get_fitted_participants(db_path, table_name)
        remaining = [p for p in participants if p not in fitted]
        fitted_count = 0
        for i, pid in enumerate(remaining, 1):
            if fitted_count >= max_participants: break
            try:
                summary_df = fit_func(df=df_hssm, participant_id=pid, participant_column=participant_column, predictor=predictor, use_log=use_log)
                write_summary_to_sql(summary_df, db_path=db_path, table_name=table_name)
                fitted_count += 1
            except Exception as e:
                print(f"❌ Failed participant {pid}: {e}")

def simulate_participant_ddm(participant_id, df, model='pure', size=300, bonus_prob=0.5):
    subset = df[df['participant_id'] == participant_id]
    z = subset[subset['param'] == 'z']['mean'].values[0]
    t = subset[subset['param'] == 't']['mean'].values[0]
    X = np.random.binomial(1, bonus_prob, size) if model in ['v', 'th'] else np.zeros(size)

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

    true_values = np.column_stack([as_trialwise(v, size), as_trialwise(a, size), np.full(size, z), np.full(size, t)])
    dataset = hssm.simulate_data(model="ddm", theta=true_values, size=1)
    dataset["participant_id"], dataset["X"] = str(participant_id), X
    return dataset

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