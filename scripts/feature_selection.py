"""Correlation-based feature selection for the DengAI dengue forecasting data.

For each city (San Juan, Iquitos) the script:
  1. keeps only the climate/vegetation columns as candidate features,
  2. measures each feature's Pearson correlation with weekly dengue cases on the
     TRAINING data only (so the test data never influences the selection),
  3. drops features that are negatively correlated or weakly correlated
     (r < MIN_CORR),
  4. compares test MAE for KNN, Random Forest, Gradient Boosting and SVR trained
     on all features versus on the selected features.

Two evaluation setups are reported:
  random        random 70/30 split (as in notebooks/thesis2_SJ.ipynb), with year
                dummies always kept as extra features.
  chronological first 70% of weeks for training, last 30% for testing, with no
                year dummies. This is closer to real forecasting.

Run from anywhere:  python scripts/feature_selection.py
Writes results/feature_selection_results.csv and images/feature_correlations.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
MIN_CORR = 0.10  # keep a feature only if its correlation with cases is >= this
ID_COLS = ["city", "year", "weekofyear", "week_start_date"]
CITIES = {"sj": "San Juan", "iq": "Iquitos"}
MODELS = ["KNN", "RF", "GBR", "SVR"]


def load():
    features = pd.read_csv(ROOT / "data" / "dengue_features_train_1457.csv")
    labels = pd.read_csv(ROOT / "data" / "dengue_labels_train_1457.csv")
    return features.merge(labels, on=["city", "year", "weekofyear"], validate="one_to_one")


def split(df, mode):
    if mode == "random":
        train, test = train_test_split(df, test_size=0.3, random_state=SEED)
    else:
        df = df.sort_values("week_start_date")
        cut = int(len(df) * 0.7)
        train, test = df.iloc[:cut], df.iloc[cut:]
    return train.copy(), test.copy()


def build_matrices(train, test, cols, mode):
    """Impute with training means, add year dummies (random mode only), standardize."""
    means = train[cols].mean()
    x_train, x_test = train[cols].fillna(means), test[cols].fillna(means)
    if mode == "random":
        for year in sorted(train["year"].unique())[1:]:  # drop first year: dummy trap
            x_train[f"year_{year}"] = (train["year"] == year).astype(int)
            x_test[f"year_{year}"] = (test["year"] == year).astype(int)
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)


def make_models():
    return {
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RF": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=SEED),
        "GBR": GradientBoostingRegressor(random_state=SEED),
        "SVR": SVR(),
    }


def select_features(train, cols):
    corr = train[cols].fillna(train[cols].mean()).corrwith(train["total_cases"])
    kept = corr[corr >= MIN_CORR].index.tolist()
    return corr, kept


def main():
    data = load()
    rows, corr_for_plot = [], {}
    for mode in ("random", "chronological"):
        for code, name in CITIES.items():
            city = data[data["city"] == code]
            train, test = split(city, mode)
            candidates = [c for c in city.columns if c not in ID_COLS + ["total_cases"]]
            corr, kept = select_features(train, candidates)
            if mode == "random":
                corr_for_plot[name] = (corr, kept)
            for label, cols in (("all features", candidates), ("selected features", kept)):
                x_train, x_test = build_matrices(train, test, cols, mode)
                for model_name, model in make_models().items():
                    model.fit(x_train, train["total_cases"])
                    mae = mean_absolute_error(test["total_cases"], model.predict(x_test))
                    rows.append(
                        dict(split=mode, city=name, features=label, n_climate_features=len(cols),
                             model=model_name, mae=round(mae, 2))
                    )

    results = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "feature_selection_results.csv", index=False)

    # ---- console summary ----
    for mode in ("random", "chronological"):
        print(f"\n=== {mode} split ===")
        for name in CITIES.values():
            sub = results[(results.split == mode) & (results.city == name)]
            counts = sub.groupby("features").n_climate_features.first()
            print(f"{name}: {counts['all features']} candidate features -> "
                  f"{counts['selected features']} selected")
            table = sub.pivot(index="model", columns="features", values="mae").loc[MODELS]
            table["change"] = (table["selected features"] - table["all features"]).round(2)
            print(table.to_string())

    # ---- figure: correlations with cases, colored by kept/dropped ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True)
    for ax, (name, (corr, kept)) in zip(axes, corr_for_plot.items()):
        corr = corr.sort_values()
        colors = ["#2a9d8f" if f in kept else "#c8c8c8" for f in corr.index]
        ax.barh(corr.index, corr.values, color=colors)
        ax.axvline(MIN_CORR, color="#e76f51", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{name}: {len(kept)} of {len(corr)} features kept")
        ax.set_xlabel("Correlation with weekly dengue cases (training data)")
    fig.suptitle(f"Correlation-based feature selection (keep r ≥ {MIN_CORR})", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "feature_correlations.png", dpi=150)


if __name__ == "__main__":
    main()
