"""Customer churn risk app.

Run locally (from the project root):
    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
FIG = ROOT / "reports" / "figures"

YES_NO = ["No", "Yes"]
INTERNET_ADDON = ["No", "Yes", "No internet service"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}
LABELS = {
    "SeniorCitizen": "Senior citizen", "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges", "tenure": "Tenure",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def expected_value(proba, monthly, a):
    """Expected profit of sending one retention offer to this customer."""
    return proba * a["save_rate"] * a["months_of_value"] * monthly - a["offer_cost"]


def drivers(pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """Per-field contribution to the log-odds for a logistic regression pipeline.

    Numeric fields are standardised, so their contribution is measured against an
    average customer. One-hot fields contribute their coefficient when present.
    """
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    x = prep.transform(row)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    names = prep.get_feature_names_out()
    contrib = pd.DataFrame({"term": names, "value": x * model.coef_[0]})
    contrib = contrib[contrib["value"].abs() > 1e-9]
    contrib["field"] = contrib["term"].str.replace(r"^(num|cat)__", "", regex=True).str.split("_").str[0]
    out = contrib.groupby("field", as_index=False)["value"].sum()
    out["label"] = out["field"].map(lambda f: LABELS.get(f, f))
    return out.sort_values("value", key=np.abs, ascending=False)


def customer_form() -> pd.DataFrame:
    st.sidebar.header("Customer details")
    st.sidebar.caption("Defaults describe a typical high-risk customer. Change any field.")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 4)
    monthly = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 85.0, step=0.5)
    total = st.sidebar.number_input("Total charges to date ($)", min_value=0.0,
                                    value=float(round(tenure * monthly, 2)), step=10.0)
    defaults = {"Contract": "Month-to-month", "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check", "PaperlessBilling": "Yes",
                "OnlineSecurity": "No", "TechSupport": "No", "PhoneService": "Yes"}
    values = {"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total}
    for field, opts in OPTIONS.items():
        default = defaults.get(field, opts[0])
        values[field] = st.sidebar.selectbox(LABELS.get(field, field.title()), opts,
                                             index=opts.index(default))
    return pd.DataFrame([values])


def score_one(bundle, row):
    pipe, threshold, a = bundle["pipeline"], bundle["threshold"], bundle["assumptions"]
    proba = float(pipe.predict_proba(row[bundle["features"]])[:, 1][0])
    ev = expected_value(proba, float(row["MonthlyCharges"].iloc[0]), a)
    contact = proba >= threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn risk", f"{proba:.0%}")
    c2.metric("Recommendation", "Contact" if contact else "No action")
    c3.metric("Expected value of an offer", f"${ev:,.0f}")
    if contact:
        st.warning(f"Risk is above the campaign threshold of {threshold:.0%}: include this customer "
                   "in the retention campaign.", icon="⚠️")
    else:
        st.success(f"Risk is below the campaign threshold of {threshold:.0%}: no offer needed.", icon="✅")

    st.subheader("What's driving this score")
    d = drivers(pipe, row[bundle["features"]]).head(6)
    up, down = d[d["value"] > 0], d[d["value"] < 0]
    col_up, col_down = st.columns(2)
    with col_up:
        st.markdown("**Raises risk**")
        for _, r in up.iterrows():
            st.markdown(f"- {r['label']}: **{row[r['field']].iloc[0]}**")
        if up.empty:
            st.caption("Nothing notable.")
    with col_down:
        st.markdown("**Lowers risk**")
        for _, r in down.iterrows():
            st.markdown(f"- {r['label']}: **{row[r['field']].iloc[0]}**")
        if down.empty:
            st.caption("Nothing notable.")
    st.caption("Based on the logistic regression's coefficients for this customer, largest effects first. "
               "Tenure, monthly and total charges overlap, so read them together.")


def score_batch(bundle):
    st.write("Upload a CSV with the same columns as the Telco dataset (customerID optional) "
             "to score many customers at once.")
    template = pd.DataFrame([{
        "customerID": "0001-EXAMPLE", "tenure": 4, "MonthlyCharges": 85.0, "TotalCharges": 340.0,
        **{f: OPTIONS[f][0] for f in OPTIONS},
    }])
    st.download_button("Download a template CSV", template.to_csv(index=False), "churn_template.csv",
                       "text/csv")
    file = st.file_uploader("Customer file", type="csv")
    if file is None:
        return
    df = pd.read_csv(file)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    a = bundle["assumptions"]
    df["churn_risk"] = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1].round(3)
    df["contact"] = np.where(df["churn_risk"] >= bundle["threshold"], "Yes", "No")
    df["expected_value"] = expected_value(df["churn_risk"], df["MonthlyCharges"], a).round(0)
    df = df.sort_values("churn_risk", ascending=False)
    st.metric("Customers to contact", f"{(df['contact'] == 'Yes').sum():,} of {len(df):,}")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download scored file", df.to_csv(index=False), "scored_customers.csv", "text/csv")


def model_info(bundle):
    m, a = bundle["test_metrics"], bundle["assumptions"]
    st.write(f"**Model:** {m['model']}, evaluated once on {m['test_customers']:,} held-out customers.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}", help=f"Chance level is {m['pr_auc_chance']:.3f}")
    c3.metric("Churners caught", f"{m['recall']:.0%}")
    c4.metric("Contacts who would churn", f"{m['precision']:.0%}")
    st.write(f"At the {m['threshold']:.0%} threshold the campaign contacts {m['customers_targeted_pct']}% "
             f"of customers. Expected profit on the test set: **\\${m['campaign_profit_model']:,}**, against "
             f"\\${m['campaign_profit_target_everyone']:,} for offering everyone a discount.")
    st.caption(f"Assumptions: an offer costs \\${a['offer_cost']:.0f}, keeps {a['save_rate']:.0%} of "
               f"would-be churners, and a kept customer is worth {a['months_of_value']} months of revenue.")
    for name in ["01_churn_drivers.png", "02_profit_curve.png", "03_feature_importance.png"]:
        path = FIG / name
        if path.exists():
            st.image(str(path), width="stretch")


def main():
    st.set_page_config(page_title="Customer Churn Risk", page_icon="📉", layout="wide")
    st.title("Customer churn risk")
    st.caption("Predicts which telecom customers are likely to leave and whether a retention offer "
               "pays off. Model trained on the IBM Telco Customer Churn dataset (7,043 customers).")
    bundle = load_model()
    row = customer_form()
    tab1, tab2, tab3 = st.tabs(["Score a customer", "Score a list", "How the model works"])
    with tab1:
        score_one(bundle, row)
    with tab2:
        score_batch(bundle)
    with tab3:
        model_info(bundle)


main()
