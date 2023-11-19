import numpy as np
from scipy.stats import norm, multivariate_normal

# Define methods for NA removal
methods = ["rows", "columns", "mode", "custom"]

def drop_rows(x_train, x_test, y_train, y_test):
    out_y_train = y_train.loc[x_train.isnull().sum(axis=1) == 0]
    out_y_test = y_test.loc[x_test.isnull().sum(axis=1) == 0]
    return x_train.dropna(), x_test.dropna(), out_y_train, out_y_test

def drop_columns(x_train, x_test, y_train, y_test):
    cols = x_train.columns[(x_train.isnull().sum(axis=0) == 0) & (x_test.isnull().sum(axis=0) == 0)]
    return x_train[cols], x_test[cols], y_train, y_test

def mode_imputation(x_train, x_test, y_train, y_test):
    modes = x_train.apply(lambda x : x.value_counts().index[0] if x.dtype != "float64" else x.mean())
    train_out = x_train
    test_out = x_test
    for i, c in enumerate(x_train.columns):
        train_out.loc[train_out[c].isnull(), c] = modes.iat[i]
        test_out.loc[test_out[c].isnull(), c] = modes.iat[i]
    return train_out, test_out, y_train, y_test

def custom_imputation(x_train, x_test, y_train, y_test):
    # Prepare cleaned inputs
    out_train = x_train
    out_test = x_test
    # Impute weight correlated variables
    weight_related = ["armc","waist","tri","sub"]
    for var_name in weight_related:
        # Compute linear model based on clean train data
        clean = x_train.dropna(subset=var_name)
        mu, std = lm(clean["wt"], clean[var_name])
        # Impute training data
        train_nulls = out_train.loc[x_train[var_name].isnull(), "wt"]
        out_train.loc[x_train[var_name].isnull(), var_name] = norm.rvs(loc=mu(train_nulls), scale=std)
        # Impute test data based on training data
        test_nulls = out_test.loc[x_test[var_name].isnull(), var_name]
        out_test.loc[x_test[var_name].isnull(), var_name] = norm.rvs(loc=mu(test_nulls), scale=std)
    # Impute height correlated variables
    height_related = ["leg", "arml"]
    for var_name in height_related:
        # Compute linear model based on clean train data
        clean = x_train.dropna(subset=var_name)
        mu, std = lm(clean["ht"], clean[var_name])
        # Impute training data
        train_nulls = out_train.loc[x_train[var_name].isnull(), "ht"]
        out_train.loc[x_train[var_name].isnull(), var_name] = norm.rvs(loc=mu(train_nulls), scale=std)
        # Impute test data based on training data
        test_nulls = out_test.loc[x_test[var_name].isnull(),var_name]
        out_test.loc[x_test[var_name].isnull(), var_name] = norm.rvs(loc=mu(test_nulls), scale=std)
    # Impute remaining NA values independently as no strong correlations
    # Impute income
    clean = x_train.dropna(subset="income")
    income_probs = (clean["income"].value_counts() / clean.shape[0]).cumsum() # Calculate empirical probabilites
    gen = np.random.rand(out_train["income"].isnull().sum()) # Generate samples
    gen = income_probs.index[(income_probs.to_numpy().reshape(-1, 1) <= gen).argmin(axis=0)] # Sample from income categories
    out_train.loc[out_train["income"].isnull(), "income"] = gen
    gen = np.random.rand(out_test["income"].isnull().sum()) # Generate samples
    gen = income_probs.index[(income_probs.to_numpy().reshape(-1, 1) <= gen).argmin(axis=0)] # Sample from income categories
    out_test.loc[out_test["income"].isnull(), "income"] = gen
    # Impute albumin
    clean = x_train.dropna(subset="albumin")
    nulls = out_train["albumin"].isnull()
    mu = clean["albumin"].mean()
    std = clean["albumin"].std()
    out_train.loc[nulls, "albumin"] = norm.rvs(size=nulls.sum(), loc=mu, scale=std) # Fill in train
    nulls = out_test["albumin"].isnull()
    out_test.loc[nulls, "albumin"] = norm.rvs(size=nulls.sum(), loc=mu, scale=std) # Fill in test
    # Impute BUN and SCr
    # Impute BUN and SCr together as they are strongly correlated
    # In the data all the BUN and SCr nulls are missing together, so we can't use one to impute the other
    clean = x_train.dropna(subset=["SCr", "bun"])
    # fit the log of these variables as they are right skewed and positive
    logged = np.log(clean[["SCr","bun"]])
    mv_gen = multivariate_normal(mean=logged.mean(), cov=logged.cov())
    # Impute data for train
    nulls = x_train["SCr"].isnull()
    out_train.loc[nulls, ["SCr", "bun"]] = np.exp(mv_gen.rvs(nulls.sum())) # Convert back to original scale
    # Impute data for test
    nulls = x_test["SCr"].isnull()
    out_test.loc[nulls, ["SCr", "bun"]] = np.exp(mv_gen.rvs(nulls.sum()))  # Convert back to original scale
    return out_train, out_test, y_train, y_test    
    
def remove_NAs(x_train, x_test, y_train, y_test, method):
    if method == "rows":
        return drop_rows(x_train, x_test, y_train, y_test)
    elif method == "columns":
        return drop_columns(x_train, x_test, y_train, y_test)
    elif method == "mode":
        return mode_imputation(x_train, x_test, y_train, y_test)
    elif method == "custom":
        return custom_imputation(x_train, x_test, y_train, y_test)
    else:
        return x_train, x_test, y_train, y_test
    
def lm(x,y):
    xs = np.vstack((np.ones_like(x), x)).T
    w = np.linalg.pinv(xs) @ y
    preds = xs @ w
    mse = np.sqrt(np.square(y - preds).sum() / (len(x) - 1))
    return lambda x : w[1] * x + w[0], mse


    