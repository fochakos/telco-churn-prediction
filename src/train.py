"""Train, compare and evaluate churn models; save the chosen model for the app.

Usage (from the project root):
    python src/train.py

Steps
1. Stratified 80/20 split. The test set is only used once, at the end.
2. Compare a baseline, logistic regression, random forest and gradient boosting
   with 5-fold cross-validation on the training set (ROC-AUC, PR-AUC).
3. Choose a decision threshold that maximises the expected profit of a retention
   campaign, using out-of-fold predictions on the training set.
4. Evaluate the chosen model and threshold once on the test set.
5. Save the model, metrics and charts.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET, load_clean  # noqa: E402

SEED = 42
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"

# Retention campaign assumptions (illustrative; change them to match a real business).
OFFER_COST = 60.0         # cost of one retention offer, about one month's bill
SAVE_RATE = 0.30          # share of would-be churners the offer keeps
MONTHS_OF_VALUE = 12      # revenue kept per saved customer = 12 x monthly charge


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])


def candidate_models() -> dict:
    return {
        "Baseline (always 'stay')": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "Random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=SEED),
    }


def campaign_profit(y_true, proba, monthly_charges, threshold) -> float:
    """Expected profit of offering a discount to everyone scored at or above the threshold."""
    target = proba >= threshold
    value_saved = SAVE_RATE * MONTHS_OF_VALUE * monthly_charges * y_true
    return float(np.sum((value_saved - OFFER_COST) * target))


def profit_curve(y_true, proba, monthly_charges):
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = [campaign_profit(y_true, proba, monthly_charges, t) for t in thresholds]
    return thresholds, np.array(profits)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Compare models with cross-validation (training set only)
    cv_rows, oof = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        cv_rows.append({
            "model": name,
            "cv_roc_auc": round(roc_auc_score(y_train, proba), 3),
            "cv_pr_auc": round(average_precision_score(y_train, proba), 3),
        })
    cv_table = pd.DataFrame(cv_rows)
    print(cv_table.to_string(index=False))

    # Pick the best real model on CV PR-AUC; prefer logistic regression when within 0.01
    # because its drivers are easy to explain to a retention team.
    real = cv_table[~cv_table["model"].str.startswith("Baseline")].set_index("model")
    best = real["cv_pr_auc"].idxmax()
    if real.loc["Logistic regression", "cv_pr_auc"] >= real.loc[best, "cv_pr_auc"] - 0.01:
        best = "Logistic regression"
    print(f"\nSelected: {best}")

    # 2. Threshold from out-of-fold predictions on the training set
    mc_train = X_train["MonthlyCharges"].to_numpy()
    thresholds, profits = profit_curve(y_train.to_numpy(), oof[best], mc_train)
    threshold = float(thresholds[profits.argmax()])
    print(f"Profit-maximising threshold (train, out-of-fold): {threshold}")

    # 3. Fit on the full training set and evaluate once on the test set
    final = Pipeline([("prep", preprocessor()), ("model", clone(candidate_models()[best]))])
    final.fit(X_train, y_train)
    test_proba = final.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    mc_test = X_test["MonthlyCharges"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    profit_model = campaign_profit(y_test.to_numpy(), test_proba, mc_test, threshold)
    profit_all = campaign_profit(y_test.to_numpy(), np.ones_like(test_proba), mc_test, 0.5)
    test_metrics = {
        "model": best,
        "threshold": threshold,
        "roc_auc": round(roc_auc_score(y_test, test_proba), 3),
        "pr_auc": round(average_precision_score(y_test, test_proba), 3),
        "pr_auc_chance": round(float(y_test.mean()), 3),
        "recall": round(recall_score(y_test, test_pred), 3),
        "precision": round(precision_score(y_test, test_pred), 3),
        "f1": round(f1_score(y_test, test_pred), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "customers_targeted_pct": round(100 * test_pred.mean(), 1),
        "campaign_profit_model": round(profit_model),
        "campaign_profit_target_everyone": round(profit_all),
        "test_customers": int(len(y_test)),
        "test_churners": int(y_test.sum()),
    }
    print(json.dumps(test_metrics, indent=2))

    # 4. What drives churn? Permutation importance on the test set (in PR-AUC)
    perm = permutation_importance(final, X_test, y_test, scoring="average_precision",
                                  n_repeats=20, random_state=SEED, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean,
                                "std": perm.importances_std})
                  .sort_values("importance", ascending=False))

    # Odds ratios when the model is logistic regression
    odds = None
    if best == "Logistic regression":
        names = final.named_steps["prep"].get_feature_names_out()
        coefs = final.named_steps["model"].coef_[0]
        odds = (pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
                .sort_values("coef", ascending=False))

    # 5. Save everything the app and the README need
    joblib.dump({
        "pipeline": final,
        "threshold": threshold,
        "features": FEATURES,
        "assumptions": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE,
                        "months_of_value": MONTHS_OF_VALUE},
        "test_metrics": test_metrics,
    }, MODELS_DIR / "churn_model.joblib")

    cv_table.to_csv(REPORTS / "cv_results.csv", index=False)
    importance.to_csv(REPORTS / "permutation_importance.csv", index=False)
    if odds is not None:
        odds.to_csv(REPORTS / "odds_ratios.csv", index=False)
    pd.DataFrame({"threshold": thresholds, "profit_train_oof": profits}).to_csv(
        REPORTS / "profit_curve.csv", index=False)
    pd.DataFrame({"y_true": y_test.to_numpy(), "proba": test_proba,
                  "monthly_charges": mc_test}).to_csv(REPORTS / "test_predictions.csv", index=False)
    with open(REPORTS / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved model to {MODELS_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
