import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

def subsitute_values_sequential(data, varname, new_values):
    unique_values = sorted(data[varname].unique())
    substitutions = {val: new_val for val, new_val in zip(unique_values, new_values)}
    return data[varname].replace(substitutions)

def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / (mad if mad > 0 else np.finfo(float).eps)

def write_to_sql(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()

def write_summary_to_sql(df, db_path, table_name):
    df = df.copy()
    print(df)
    df["timestamp"] = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        print("writing")
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()

def get_fitted_participants(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        try:
            q = f"SELECT DISTINCT participant_id FROM {table_name}"
            return set(pd.read_sql(q, conn)['participant_id'])
        except Exception:
            return set()

def as_trialwise(x, size):
    return x if np.ndim(x) else np.full(size, x)