import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score
from sklearn.pipeline import Pipeline
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
models = [
    ("Ordinary Least Squares (OLS)", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.01)),
    ("Ridge Regression", Ridge(alpha=1.0)),
    ("Gradient Descent Regression", SGDRegressor(max_iter=1000, tol=1e-3)),
    ("Polynomial Regression", Pipeline([("poly", PolynomialFeatures(degree=2)), ("linear", LinearRegression())])),
]
import pandas as pd
result = []
result1 = []
for model_name, model in models:
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    explained_var_train = explained_variance_score(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    explained_var_test = explained_variance_score(y_test, y_test_pred)
    result.append({
        "Model": model_name,
        "MSE (Training)": mse_train,
        "RMSE (Training)": rmse_train,
        "MAE (Training)": mae_train,
        "R2 (Training)": r2_train,
        "Explained Variance (Training)": explained_var_train,
    })
    result1.append({
        "Model": model_name,
        "MSE (Test)": mse_test,
        "RMSE (Test)": rmse_test,
        "MAE (Test)": mae_test,
        "R2 (Test)": r2_test,
        "Explained Variance (Test)": explained_var_test
    })
result_df = pd.DataFrame(result)
result1_df = pd.DataFrame(result1)
print("Training Metrics:")
print(result_df)
print("\nTest Metrics:")
print(result1_df)
