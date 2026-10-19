"""Train the model and export it as docs/model.json so the web page can run it in the browser."""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def export_model(model, X, mae, r2):
    """Turn a fitted GradientBoostingRegressor into plain JSON."""
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "f": t.feature.tolist(),
            "t": t.threshold.tolist(),
            "v": [round(float(v[0][0]), 6) for v in t.value],
        })
    return {
        "features": list(X.columns),
        "init": float(np.ravel(model.init_.constant_)[0]),
        "lr": model.learning_rate,
        "trees": trees,
        "meta": {
            "mae": float(mae),
            "r2": float(r2),
            "features": {
                c: {"min": float(X[c].quantile(0.01)),
                    "max": float(X[c].quantile(0.99)),
                    "default": float(X[c].median())}
                for c in X.columns
            },
        },
    }


def main():
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)  # y is in $100k
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline = LinearRegression().fit(X_tr, y_tr)
    print(f"Linear Regression  R2 = {r2_score(y_te, baseline.predict(X_te)):.3f}")

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    r2, mae = r2_score(y_te, pred), mean_absolute_error(y_te, pred) * 100_000
    print(f"Gradient Boosting  R2 = {r2:.3f} | MAE = ${mae:,.0f}")

    out = Path("docs/model.json")
    out.write_text(json.dumps(export_model(model, X, mae, r2)))
    print(f"Saved {out} ({out.stat().st_size / 1024:.0f} KB). Commit this file.")


if __name__ == "__main__":
    main()
