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