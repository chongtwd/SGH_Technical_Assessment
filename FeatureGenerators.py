import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import PolynomialFeatures

methods = ["simple", "polynomial"]

def mt20000(x):
    if pd.isna(x):
        return x
    matches = re.findall("[0-9]+", x)
    if len(matches) > 1:
        return int(matches[0]) > 20000
    else:
        return x.find(">") > -1
    
def binarise_income(x):
    return x.apply(mt20000)

def rationalise_income(x):
    return x.replace({"> 20000" : "[20000,25000)", "< 20000" : "[15000,20000)"})

def process_categorical(x):
    df = x[x.columns[x.dtypes != "float64"]]
    df.loc[:, "sex"] = df["sex"] == "male" # Convert strings into objects
    if "income" in x.columns:
        df["income_binary"] = binarise_income(df["income"])
        df.loc[:, "income"] = rationalise_income(df["income"])
    return pd.get_dummies(df)

def get_polynomial_features(x, d=2):
    poly = PolynomialFeatures(d)
    poly.fit(x)
    out = pd.DataFrame(poly.transform(x), columns=poly.get_feature_names_out())
    return out.rename({"1" : "bias"}, axis=1)

def get_ratios(x):
    out = {}
    if "armc" in x.columns and "arml" in x.columns:
        out["armc_by_l"] = x["armc"] / x["arml"]
    if "wasit" in x.columns and "ht" in x.columns:
        out["waist_by_ht"] = x["waist"] / x["ht"]
    if "bun" in x.columns and "SCr" in x.columns:
        out["bun_by_SCr"] = x["bun"] / x["SCr"]
    out = pd.DataFrame(out) if len(out) > 0 else None
    return out

def get_features(x, method):
    bias = pd.DataFrame(np.ones(x.shape[0]), columns=["bias"])
    bias.index = x.index
    categorical = process_categorical(x)
    continuous = x[x.columns[x.dtypes == "float64"]]
    ratios = get_ratios(x)
    if ratios is not None:
        ratios.index = x.index
    if method.find("simple") > -1:
        return pd.concat((bias, categorical, continuous, ratios), axis=1)
    elif method == "polynomial":
        poly = get_polynomial_features(continuous)
        poly.index = x.index
        return pd.concat((categorical, poly, ratios), axis=1)
    else:
        return pd.concat((bias, categorical, continuous, ratios), axis=1)
