import pandas as pd
import numpy as np
import json
import random


def get_a_funnyName():
    f = open("datasets/funnynames.txt", "r")
    nomes = f.readlines()
    return random.choice(nomes).rstrip('\n')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def describe(df):
    return pd.concat([df.describe(include="all", percentiles=[.1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9]).T,
                      df.dtypes.rename('dtypes'),
                      #df.mad().rename('mad'),
                      #df.skew().rename('skew'),
                      #df.kurt().rename('kurt'),
                      df.nunique().rename('nunique'),
                     ], axis=1)


def describe_cols(df, df_summary=None):
    if df_summary is None:
        df_summary = describe(df)
    ret = {}
    for i, row in df_summary.iterrows():
        if row["nunique"] <= 100:
            ret[i] = {"classes": json.loads(df.groupby([i]).size().to_json())}
        elif np.issubdtype(row["dtypes"], np.number):
            _h = np.histogram(df[i])
            ret[i] = {"hist": list(_h[0]),
                      "bin_edges": list(_h[1])}
        else:
            ret[i] = {}
    return ret

