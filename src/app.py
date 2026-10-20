"""CareConnect Stakeholder Dashboard - Module 5 (BAN 6800).

Every number on every page is read from artifacts produced by the Module 3/4 code
(artifacts/analysis.json, mitigation.json, model_results.json) or computed live by the
trained XGBoost model. Nothing is hard-coded and there is no demo mode.

Run:  streamlit run src/app.py
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")
ART = Path(__file__).resolve().parent.parent / "artifacts"

TEAL, CORAL, AMBER, SLATE, GREEN, RED, LIGHT = "#0F4C5C", "#E36414", "#E9A03B", "#5C6B73", "#2A9D8F", "#C0392B", "#EEF4F5"
st.set_page_config(page_title="CareConnect Dashboard", page_icon="🏥", layout="wide")
st.markdown(f"""<style>
h1,h2,h3 {{color:{TEAL};}}
.card {{background:{LIGHT};border-radius:10px;padding:14px 18px;}}
.big {{font-size:2rem;font-weight:700;color:{TEAL};line-height:1.1;}}
.small {{font-size:0.85rem;color:{SLATE};}}
.warn {{background:#FFF4E5;border-radius:10px;padding:12px 16px;border:1px solid {AMBER};}}
</style>""", unsafe_allow_html=True)


def _opt(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


@st.cache_data
def load():
    A = json.load(open(ART / "analysis.json"))
    M = json.load(open(ART / "mitigation.json"))
    H = json.load(open(ART / "neighbourhoods.json"))
    return (A, M, H, _opt(ART / "fairness_policy.json"), _opt(ART / "explanations.json"), _opt(ART / "monitoring" / "monitoring.json"),
            _opt(ART / "promotion_decision.json"), _opt(ART / "policy_comparison.json"), _opt(ART / "model_results.json"))


@st.cache_resource
def load_model():
    return joblib.load(ART / "xgboost.joblib")


A, MIT, HOODS, POLICY, EXPL, MON, GATE, PCOMP, RES = load()
model = load_model()
T = A["threshold"]
HIGH = json.load(open(ART / "operating_point.json"))["high_threshold"]
OP = A["operating"]["xgboost"]["at_recall_target"]
ADOPTED = MIT.get("configured", MIT.get("adopted", "B_equalized_odds"))   # configured, not approved
MB = MIT[ADOPTED]
VERSION = A.get("model_version", "1.x")
import hashlib  # noqa: E402
MODEL_SHA = hashlib.sha256((ART / "xgboost.joblib").read_bytes()).hexdigest()
BUILD_SHA = (RES or {}).get("git_sha", "unknown")
FIGS = ART / "figures"


def policy_prob(p, group):
    if not POLICY:
        return float(p >= T)
    r = POLICY["groups"].get(group)
    op = lambda s, o: (s > o[1]) if o[0] == ">" else (s < o[1])
    inner = r["p0"] * op(p, r["op0"]) + r["p1"] * op(p, r["op1"])
    return float(r.get("p_ignore", 0) * r.get("prediction_constant", 0) + (1 - r.get("p_ignore", 0)) * inner)
PLAIN = {"lead_time_days": "Lead time (days from booking to visit)", "Age": "Patient age",
         "prior_noShow_rate": "Share of past visits missed", "prior_noShows": "Number of past missed visits",
         "prior_appointments": "Number of past visits", "SMS_received": "Reminder SMS sent",
         "Neighbourhood": "Neighbourhood", "age_band": "Age group", "Scholarship": "Welfare scholarship",
         "appointment_month": "Month of visit", "appointment_dow": "Day of week", "Gender": "Gender",
         "Alcoholism": "Alcoholism flag", "Diabetes": "Diabetes flag", "Hipertension": "Hypertension flag",
         "Handcap": "Disability level"}
BANDS = [("child", 0, 12), ("teen", 13, 18), ("young_adult", 19, 35), ("adult", 36, 55), ("senior", 56, 75), ("elderly", 76, 100)]


def band(age):
    return next(b for b, lo, hi in BANDS if lo <= age <= hi)


def tier(p):
    return "High" if p >= HIGH else ("Medium" if p >= T else "Low")


def tier_color(t):
    return {"High": RED, "Medium": AMBER, "Low": GREEN}[t]


def explain(df, k=4):
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    c = clf.get_booster().predict(xgb.DMatrix(pre.transform(df), feature_names=names), pred_contribs=True)[0][:-1]
    agg = {}
    for n, v in zip(names, c):
        b = n.split("__", 1)[1]
        g = next((x for x in ["Neighbourhood", "age_band", "Gender"] if b.startswith(x)), b)
        agg[g] = agg.get(g, 0.0) + float(v)
    return sorted(agg.items(), key=lambda x: -abs(x[1]))[:k]


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white", font=dict(size=13, color="#222"),
                      title_font=dict(size=16, color=TEAL), legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(gridcolor="#E5E7EB")
    return fig


def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="small">{label}</div><div class="big">{value}</div><div class="small">{sub}</div></div>', unsafe_allow_html=True)


# ── Sidebar ──
st.sidebar.markdown("## 🏥 CareConnect")
st.sidebar.caption("Predictive Engagement System - CrescentCare Community Hospital Network")
page = st.sidebar.radio("Navigate", ["1. Overview", "2. Results & insights", "3. Explainability (XAI)",
                                     "4. Risk explorer & what-if", "5. Ethical compliance (fairness)", "6. Limitations & transparency",
                                     "7. Monitoring (drift & fairness)"])
st.sidebar.markdown("---")
st.sidebar.caption(f"BAN 6800 capstone - model v{VERSION}. Trained on the public Kaggle 'Medical Appointment No Shows' data "
                   "(Brazil, 2016). It has **not** been validated on CrescentCare patients.")
if POLICY:
    st.sidebar.caption(f"Outreach rule: **{POLICY['name']}** - {POLICY.get('status', '')}.")
st.sidebar.caption(f"Model artifact sha256 `{MODEL_SHA[:12]}` · built from commit `{BUILD_SHA}`")
st.sidebar.caption("Deployment: not deployed. No public API exists; the hosted dashboard serves the v1 results from `main` "
                   "until a merge is approved.")

# ═════════ 1 ═════════
if page.startswith("1"):
    st.title("CareConnect: what the model found")
    st.caption("A plain-language view of the Module 4 no-show model for non-technical stakeholders. "
               "This page describes findings; it does not make a recommendation.")
    c = st.columns(4)
    with c[0]: card("Hospital no-show rate (Module 1)", "28%", "Target < 15%")
    with c[1]: card("No-show rate in the training data", f"{A['descriptive']['overall']:.1%}", f"{A['descriptive']['n']:,} appointments")
    with c[2]: card("Ranking ability (ROC-AUC)", f"{OP['roc_auc']:.2f}", "0.50 = coin flip, 1.00 = perfect")
    with c[3]: card("No-shows caught (recall)", f"{OP['recall']:.0%}", f"but only {OP['precision']:.0%} of flags are real no-shows")
    st.markdown("### The model in plain language")
    st.markdown(f"""
CareConnect gives every booked appointment a **risk score** between 0% and 100% - the model's estimate of the chance the patient will not turn up.
It learned from {A['descriptive']['n']:,} past appointments, looking at things like how far ahead the visit was booked, the patient's past
attendance, age and whether a reminder SMS was sent.

* **What it is:** a *smoke detector* for appointments. It points at where a missed visit is more likely so a person can look closer.
* **What it is not:** a verdict on any patient, a way to decide who gets care, or a replacement for a coordinator's judgement.
* **Who acts on it:** care coordinators and social workers. Every outreach decision stays with a human.
""")
    st.markdown("### Where the model stands against the Module 4 acceptance targets")
    rows = [("ROC-AUC", "≥ 0.70", f"{OP['roc_auc']:.3f}", OP["roc_auc"] >= 0.70),
            ("Recall (single cut-off)", "≥ 0.65", f"{OP['recall']:.3f}", OP["recall"] >= 0.65),
            ("F1 score", "≥ 0.55", f"{OP['f1']:.3f}", OP["f1"] >= 0.55),
            ("Age-group gap, single cut-off", "≤ 0.05", f"{A['fairness']['age_band']['eod']:.3f}", A["fairness"]["age_band"]["pass_eod"]),
            ("Age-group gap, Option B (expected value)", "≤ 0.05", f"{MB['age_band']['eod']:.3f}", MB["age_band"]["eod"] <= 0.05),
            ("Gender gap, Option B (expected)", "≤ 0.05", f"{MB['Gender']['eod']:.3f}", MB["Gender"]["eod"] <= 0.05),
            ("Scholarship gap, Option B (expected)", "≤ 0.05", f"{MB['Scholarship']['eod']:.3f}", MB["Scholarship"]["eod"] <= 0.05)]
    st.dataframe(pd.DataFrame([{"Measure": a, "Target": b, "Result": c_, "Met?": "✅ Yes" if d else "❌ No"} for a, b, c_, d in rows]),
                 hide_index=True, width="stretch")
    st.caption("Results are on a 20% hold-out set the model never saw during training. The outreach rule configured for evaluation (not approved, not in operational use) is "
               f"Option B (equalized odds by age group), which flags {MB['flag_rate']:.0%} of appointments and catches {MB['recall']:.0%} of no-shows.")
    if GATE:
        st.info(f"**Promotion gate:** {GATE['decision']}. Automatic promotion is disabled; the AI Ethics Committee approves every version.")
        if GATE.get("recommendation"):
            st.error(f"**Recommendation: {GATE['recommendation']}.** Deployment status: {GATE.get('deployment_status', 'not deployed')}.")

# ═════════ 2 ═════════
elif page.startswith("2"):
    st.title("Results & insights")
    D = A["descriptive"]
    lt = pd.DataFrame(D["lead_time"])
    f = go.Figure(go.Bar(x=lt["bucket"], y=lt["mean"] * 100, marker_color=[GREEN, AMBER, AMBER, CORAL, CORAL, RED],
                         text=[f"{v:.0f}%" for v in lt["mean"] * 100], textposition="outside"))
    f.add_hline(y=D["overall"] * 100, line_dash="dot", annotation_text=f"Overall {D['overall']:.0%}")
    f.update_layout(title="Visits booked further ahead are missed more often", yaxis_title="% of appointments missed", yaxis_range=[0, 42])
    st.plotly_chart(style(f), width="stretch")
    st.caption("Same-day bookings are missed about 5% of the time; visits booked a month or more ahead about a third of the time.")
    c1, c2 = st.columns(2)
    dec = pd.DataFrame(A["deciles"])
    f = go.Figure()
    f.add_bar(x=[f"{i+1}" for i in dec["d"]], y=dec["actual"] * 100, name="Actually missed", marker_color=TEAL)
    f.add_scatter(x=[f"{i+1}" for i in dec["d"]], y=dec["pred"] * 100, name="Model's average score", line=dict(color=CORAL, width=3))
    f.update_layout(title="Do higher scores really mean more misses?", xaxis_title="Appointments sorted from lowest to highest score (10 groups)", yaxis_title="% missed")
    c1.plotly_chart(style(f, 400), width="stretch")
    ps = pd.DataFrame(D["prior"])
    f = go.Figure(go.Bar(x=["0 past misses", "1 past miss", "2 past misses", "3+ past misses"], y=ps["mean"] * 100,
                         marker_color=[GREEN, AMBER, CORAL, RED], text=[f"{v:.0f}%" for v in ps["mean"] * 100], textposition="outside"))
    f.update_layout(title="Past behaviour is a strong signal", yaxis_title="% missed", yaxis_range=[0, 75])
    c2.plotly_chart(style(f, 400), width="stretch")
    st.markdown("### What this means")
    st.markdown(f"""
* The lowest-scoring 10% of unseen appointments were missed **{dec['actual'].iloc[0]:.0%}** of the time; the highest-scoring 10% were missed **{dec['actual'].iloc[-1]:.0%}**.
* The overall rate in the same data is **{A['base_rate']:.0%}**, so the top group is roughly **{dec['actual'].iloc[-1]/A['base_rate']:.1f}x** the average.
* At the standard flagging rule the model flags **{OP['flag_rate']:.0%}** of appointments and catches **{OP['recall']:.0%}** of the real no-shows. About **{1-OP['precision']:.0%}** of flagged appointments are attended anyway.
""")

# ═════════ 3 ═════════
elif page.startswith("3"):
    st.title("What drives the predictions?")
    sh = A["shap"]["share"]
    items = list(sh.items())[:8]
    f = go.Figure(go.Bar(y=[PLAIN.get(k, k) for k, _ in items][::-1], x=[v * 100 for _, v in items][::-1], orientation="h", marker_color=TEAL,
                         text=[f"{v*100:.0f}%" for _, v in items][::-1], textposition="outside"))
    f.update_layout(title="Share of the model's decision-making attributed to each factor (SHAP)", xaxis_title="% of total influence", xaxis_range=[0, 72])
    st.plotly_chart(style(f, 440), width="stretch")
    top = items[0]
    st.markdown(f"""
**How to read this:** SHAP measures how far each factor pushes a prediction up or down. Across 4,000 sampled test appointments,
**{PLAIN[top[0]].split(' (')[0].lower()} accounts for about {top[1]*100:.0f}%** of the total push; the next two factors
({PLAIN[items[1][0]].lower()}, {PLAIN[items[2][0]].lower()}) add about {(items[1][1]+items[2][1])*100:.0f}% together.
Neighbourhood contributes only about {sh.get('Neighbourhood',0)*100:.0f}%.

These are **associations the model found in past data, not proven causes.** Changing a factor does not guarantee the outcome changes.
""")
    st.markdown("### Three example patients (from the unseen test set)")
    cols = st.columns(3)
    for col, s in zip(cols, sorted(A["samples"], key=lambda x: -x["prob"])):
        t = tier(s["prob"]); fe = s["features"]
        with col:
            st.markdown(f"<div class='card'><b style='color:{tier_color(t)}'>{t} risk - score {s['prob']:.0%}</b><br>"
                        f"<span class='small'>{fe['Gender']}, age {fe['Age']}, booked {fe['lead_time_days']} days ahead, "
                        f"{fe['prior_noShows']} missed of {fe['prior_appointments']} past visits, SMS {'sent' if fe['SMS_received'] else 'not sent'}</span></div>", unsafe_allow_html=True)
            for k, v in s["top"][:3]:
                st.markdown(f"{'🔺' if v > 0 else '🔻'} **{PLAIN.get(k, k)}** {'raises' if v > 0 else 'lowers'} the score")
            st.caption("This patient " + ("did not attend." if s["actual"] else "attended."))
    st.markdown("### About the reminder SMS")
    sm = {d["SMS_received"]: d["mean"] for d in A["descriptive"]["sms"]}
    st.info(f"In the raw data, appointments with an SMS were missed **more** often ({sm[1]:.0%}) than those without ({sm[0]:.0%}). "
            "SMS reminders are only sent for visits booked a few days or more ahead, and those are missed more often anyway. "
            f"After the model accounts for lead time, marking everyone as 'SMS sent' changes the average score by {A['whatif_avg_pp']['Everyone marked as SMS received']:+.1f} points. "
            "That is a small, uncertain effect and should not be read as proof that SMS works or fails.")
    if (FIGS / "shap_summary.png").exists():
        st.markdown("### SHAP summary (every dot is one appointment)")
        st.image(str(FIGS / "shap_summary.png"), width=760)
        st.caption("Right of zero = pushes the risk score up; colour = the feature's value (red high, blue low). "
                   "Long lead times (red, right) raise risk; same-day bookings (blue, left) lower it.")
    if EXPL:
        st.markdown("### Second opinion: LIME, and how far to trust it")
        st.caption("LIME fits a simple local model around one prediction. Two sampling set-ups were run on the same appointments; "
                   f"a fit counts as adequate only if R² ≥ {EXPL['fidelity_bar']['min_r2']} and LIME's own prediction is within "
                   f"{EXPL['fidelity_bar']['max_abs_error']:.2f} of the model's score (bar fixed before the run).")
        lt = pd.DataFrame([{"Set-up": e["config"].replace("_", " "), "Example": e["example"], "Model score": f"{e['model_score']:.1%}",
                            "LIME prediction": f"{e['lime_local_prediction']:.1%}", "R²": round(e["lime_r2"], 2),
                            "Top-3 shared with SHAP": f"{e['top3_overlap']}/3", "Fidelity": e["fidelity"]} for e in EXPL["lime"]])
        st.dataframe(lt, hide_index=True, width="stretch")
        weak = sum(e["fidelity"] == "weak" for e in EXPL["lime"] if e["config"] == "local_sampling")
        st.markdown(f"With local sampling, **{weak} of {sum(e['config'] == 'local_sampling' for e in EXPL['lime'])}** explanations remain weak. "
                    "Where LIME is weak, the exact SHAP reasons above are the explanation of record; LIME is shown as a cross-check, not a source.")
        cols = st.columns(3)
        for col, e in zip(cols, [x for x in EXPL["lime"] if x["config"] == "local_sampling" and not x["example"].startswith("cycle1")]):
            with col:
                img = FIGS / f"lime_{e['config']}_{e['example']}.png"
                if img.exists():
                    st.image(str(img), width="stretch")
        st.markdown("### Counterfactuals: what would stop this appointment being flagged by the configured policy (Option B)?")
        st.caption("Only two levers are searched: an earlier slot and adding a reminder. Age, gender, history, neighbourhood and "
                   "health are never suggested for change. Each change is re-scored through the model and the policy.")
        rows = []
        for c in EXPL["counterfactuals"]:
            cf, lo = c["counterfactual"], c["lowest_reachable"]
            rows.append({"Example (score tier)": c["score_tier"], "Score": f"{c['original_score']:.1%}",
                         "Policy flag probability now": f"{c['original_policy_flag_probability']:.0%}",
                         "Smallest change to reach 0%": (f"Book {cf['lead_time_days']} days ahead" + (", add SMS" if cf["SMS_received"] else "")) if cf
                         else ("Not flagged now" if not c["applicable"] else "None of the allowed changes is enough"),
                         "Lowest flag probability reachable": f"{lo['policy_flag_probability']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pop = EXPL.get("population")
        if pop:
            st.markdown(f"Across {pop['sampled_with_flag_probability_above_0']:,} sampled test appointments with any chance of a flag, an allowed change "
                        f"would remove the flag entirely for **{pop['share_unflaggable_by_actionable_change']:.0%}** and lower its probability for "
                        f"**{pop['share_flag_probability_reducible']:.0%}**. These are changes to the model's output, not evidence that attendance would change.")
    st.caption("Explanation store: every API prediction is logged with request id, model version, policy, score, decision and reason codes "
               "(logs/explanations.jsonl), so any past flag can be explained during an audit.")

# ═════════ 4 ═════════
elif page.startswith("4"):
    st.title("Risk explorer & what-if")
    st.caption("Live predictions from the trained model. Change the inputs on the left; the right shows what would change if one input were different.")
    L, R = st.columns([1, 1.25])
    with L:
        st.subheader("Patient / appointment")
        age = st.slider("Age", 0, 100, 34)
        gender = st.selectbox("Gender", ["F", "M"])
        lead = st.slider("Days between booking and visit", 0, 120, 21)
        sms = st.checkbox("Reminder SMS sent", value=True)
        pa = st.slider("Past visits on record", 0, 15, 3)
        pn = st.slider("Of those, how many missed", 0, max(pa, 1), min(1, pa)) if pa > 0 else 0
        hood = st.selectbox("Neighbourhood", HOODS, index=HOODS.index("JARDIM CAMBURI") if "JARDIM CAMBURI" in HOODS else 0)
        schol = st.checkbox("Enrolled in welfare scholarship (Bolsa Familia)", value=False)

    def row(**o):
        d = dict(Gender=gender, Age=age, Neighbourhood=hood, Scholarship=int(schol), Hipertension=0, Diabetes=0, Alcoholism=0, Handcap=0,
                 SMS_received=int(sms), lead_time_days=lead, appointment_dow=2, appointment_month=5, age_band=band(age),
                 prior_appointments=pa, prior_noShows=pn, prior_noShow_rate=(pn / pa if pa else 0.0))
        d.update(o)
        d["prior_noShow_rate"] = d["prior_noShows"] / d["prior_appointments"] if d["prior_appointments"] else 0.0
        return pd.DataFrame([d])

    base_df = row()
    p0 = float(model.predict_proba(base_df)[0, 1]); t0 = tier(p0)
    with R:
        st.subheader("Predicted risk")
        g = go.Figure(go.Indicator(mode="gauge+number", value=p0 * 100, number={"suffix": "%"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": tier_color(t0)},
                                          "steps": [{"range": [0, T * 100], "color": "#DDF1EE"}, {"range": [T * 100, HIGH * 100], "color": "#FCEBCB"}, {"range": [HIGH * 100, 100], "color": "#F6D5D1"}]}))
        st.plotly_chart(style(g, 260), width="stretch")
        pf = policy_prob(p0, band(age))
        c1_, c2_ = st.columns(2)
        with c1_:
            card("Model score tier", t0, "from the risk score only")
        with c2_:
            card("Outreach decision (policy)", "Flagged" if pf >= 1 else ("Not flagged" if pf <= 0 else f"{pf:.0%} chance"),
                 f"Option B rule for the {band(age).replace('_', ' ')} group")
        st.caption("The score tier describes the model's risk estimate. The outreach decision comes from the configured (not approved) fairness policy, Option B, "
                   "which uses different cut-offs per age group and, for scores inside a group's band, a logged random draw. "
                   "A 'low' tier appointment can therefore be flagged, and a 'medium' one not flagged.")
        st.markdown("**Top reasons for this score**")
        for k, v in explain(base_df):
            st.markdown(f"{'🔺' if v > 0 else '🔻'} {PLAIN.get(k, k)} - {'raises' if v > 0 else 'lowers'} risk")
        st.markdown("---")
        st.subheader("What-if: change one thing")
        sc = {"Book the visit sooner (7 days ahead)": dict(lead_time_days=min(lead, 7)),
              "Book further ahead (30 days)": dict(lead_time_days=max(lead, 30)),
              "Toggle SMS reminder": dict(SMS_received=0 if sms else 1),
              "No past missed visits": dict(prior_noShows=0),
              "Two more past missed visits": dict(prior_noShows=pn + 2, prior_appointments=pa + 2)}
        out = [(n, float(model.predict_proba(row(**o))[0, 1])) for n, o in sc.items()]
        out = [(n, p1, (p1 - p0) * 100) for n, p1 in out]
        f = go.Figure(go.Bar(y=[n for n, _, _ in out][::-1], x=[d for _, _, d in out][::-1], orientation="h",
                             marker_color=[GREEN if d < 0 else CORAL for _, _, d in out][::-1],
                             text=[f"{d:+.1f} pts" for _, _, d in out][::-1], textposition="outside"))
        f.update_layout(title="Change in predicted risk (percentage points)", xaxis_title="points vs. current score")
        st.plotly_chart(style(f, 330), width="stretch")
        st.caption("What-if results show how the model's score moves. They are not predictions that the real-world visit would change.")
        st.markdown("**Averages across all unseen test appointments:** " + "; ".join(f"{k.split(' (')[0].lower()}: {v:+.1f} pts" for k, v in A["whatif_avg_pp"].items()) + ".")

# ═════════ 5 ═════════
elif page.startswith("5"):
    st.title("Ethical compliance dashboard")
    st.markdown("The Ethical Charter (Module 1, Principle 2) sets a limit: the gap between any two groups must stay **at or below 0.05 (5 percentage points)**. "
                "Breaching it is defined as a halt condition.")
    over = [n for n, k in [("age group", "age_band"), ("gender", "Gender"), ("scholarship", "Scholarship")] if MB[k]["eod"] > 0.05]
    st.markdown('<div class="warn"><b>Honest status:</b> the single cut-off breaches the limit for <b>age group</b>'
                + (" and the <b>welfare-scholarship</b> proxy" if not A["fairness"]["Scholarship_grp"]["pass_eod"] else "")
                + ". The configured rule (not approved), <b>Option B (Fairlearn equalized odds by age group)</b>, "
                + ("brings all three comparisons within the limit <i>in expectation</i>; single sets of decisions can still exceed it (see below)" if not over else f"still leaves {', '.join(over)} above the limit even in expectation")
                + f", at the cost of flagging {MB['flag_rate']:.0%} of appointments instead of {MIT['before']['flag_rate']:.0%}. "
                  "It awaits AI Ethics Committee approval.</div>", unsafe_allow_html=True)
    st.write("")
    names = {"Gender": "Gender", "age_band": "Age group", "Scholarship_grp": "Welfare scholarship (socio-economic proxy)"}
    rows = []
    for k, n in names.items():
        r = A["fairness"][k]
        rows.append({"Group compared": n, "Flagging gap": r["dpd"], "Catch-rate / false-alarm gap": r["eod"], "Lowest ÷ highest flag rate": r["di_ratio"],
                     "Within 0.05 limit?": "✅ Yes" if (r["pass_dpd"] and r["pass_eod"]) else "❌ No"})
    st.dataframe(pd.DataFrame(rows).style.format({"Flagging gap": "{:.3f}", "Catch-rate / false-alarm gap": "{:.3f}", "Lowest ÷ highest flag rate": "{:.2f}"}), hide_index=True, width="stretch")
    st.caption("Flagging gap = demographic parity difference. Catch-rate / false-alarm gap = equalized-odds difference. Lowest ÷ highest flag rate should be ≥ 0.80.")
    attr = st.selectbox("Explore a group comparison", list(names), format_func=lambda k: names[k])
    g = A["fairness"][attr]["groups"]
    gl = list(g)
    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_bar(x=gl, y=[g[x]["no_show_rate"] * 100 for x in gl], name="Actual no-show rate", marker_color=SLATE)
    f.add_bar(x=gl, y=[g[x]["flag_rate"] * 100 for x in gl], name="Flagged by model", marker_color=TEAL)
    f.update_layout(title="Who actually misses visits vs. who gets flagged", barmode="group", yaxis_title="%")
    c1.plotly_chart(style(f, 380), width="stretch")
    f = go.Figure()
    f.add_bar(x=gl, y=[(g[x]["tpr"] or 0) * 100 for x in gl], name="No-shows caught", marker_color=GREEN)
    f.add_bar(x=gl, y=[(g[x]["fpr"] or 0) * 100 for x in gl], name="Attendees flagged by mistake", marker_color=CORAL)
    f.update_layout(title="Error rates by group", barmode="group", yaxis_title="%")
    c2.plotly_chart(style(f, 380), width="stretch")
    st.markdown("### Bias mitigation: what was tried")
    lab = {"before": "Standard rule (single threshold)", "A_group_thresholds": "A. Separate threshold per age group",
           "B_equalized_odds": "B. Fairlearn equalized odds (configured, not approved)", "C_demographic_parity": "C. Fairlearn demographic parity (comparison)"}
    lab = {k: v for k, v in lab.items() if k in MIT}
    mr = []
    for k, n in lab.items():
        m = MIT[k]
        mr.append({"Approach": n, "No-shows caught": f"{m['recall']:.0%}", "Flags that are real": f"{m['precision']:.0%}", "Appointments flagged": f"{m['flag_rate']:.0%}",
                   "Age-group gap": round(m["age_band"]["eod"], 3), "Gender gap": round(m["Gender"]["eod"], 3), "Scholarship gap": round(m["Scholarship"]["eod"], 3)})
    st.dataframe(pd.DataFrame(mr), hide_index=True, width="stretch")
    _sp = MB.get("realisation_spread", {}).get("age_band")
    spread_txt = (f": across {_sp['n_seeds']} simulated draws the realised age-group gap ran {_sp['p05']:.3f}-{_sp['p95']:.3f} "
                  f"(5th-95th percentile) and exceeded 0.05 in {_sp['share_above_limit']:.0%} of draws") if _sp else ""
    st.markdown(f"""
* **Found:** the model flags older patients far less often and catches fewer of their no-shows than younger patients' - the age-group gap is {A['fairness']['age_band']['eod']:.2f} against a 0.05 limit.
* **Option A** narrows the age gap to {MIT['A_group_thresholds']['age_band']['eod']:.2f} while keeping the catch rate near {MIT['A_group_thresholds']['recall']:.0%}, but does not reach 0.05.
* **Option B (configured, not approved)** gives expected age-group, gender and scholarship gaps of {MB['age_band']['eod']:.3f}, {MB['Gender']['eod']:.3f} and {MB['Scholarship']['eod']:.3f}, but flags {MB['flag_rate']:.0%} of appointments instead of {MIT['before']['flag_rate']:.0%}, and part of the rule is randomised.
* **Option B numbers are expected values.** Because part of the rule is randomised, any single month's decisions scatter around them{spread_txt}. Option B therefore meets the limit **on average, not reliably in any one set of decisions**.
* **Status:** Option B is configured for evaluation only and is not approved for operational use. The project team recommends no deployment until a better model or sufficient additional evidence is available; equal opportunity at TPR 0.85 is an alternative assessed, not a compliant solution. The full comparison of alternatives is below. **What it costs:** more outreach per real no-show, so coordinator capacity (Module 2 assumption A-3) must be confirmed before use.
* **Why not equal flag rates:** older patients genuinely miss fewer visits, so equal *flag rates* (demographic parity) would mean ignoring real differences. The Charter limit is applied to *error rates* instead.
""")
    if PCOMP:
        st.markdown("### Comparison of decision policies (same held-out test set)")
        st.caption("Every rule was fitted on training data and evaluated once on the 22,103 test appointments. Randomised rules show expected values. "
                   "Age gap = equalized-odds difference across the six age groups; the interval is a 95% percentile bootstrap (300 resamples of test "
                   "appointments). For randomised rules both the gap and its interval are computed from expected values (flag probabilities); "
                   "draw-to-draw variation is reported separately.")
        rows = []
        for k, v in PCOMP["policies"].items():
            ag = v["age_band"]
            rows.append({"Policy": k.split("_", 1)[1].replace("_", " "), "Randomised": f"{v['share_decisions_randomised']:.0%}",
                         "Recall": round(v["recall"], 3), "Precision": round(v["precision"], 3), "F1": round(v["f1"], 3),
                         "Flagged": f"{v['flag_rate']:.0%}", "Age gap (EOD)": round(ag["eod"], 3),
                         "Age 95% range": f"{ag['eod_bootstrap_95ci'][0]:.3f}-{ag['eod_bootstrap_95ci'][1]:.3f}",
                         "Age TPR gap": round(ag["tpr_gap"], 3), "Age FPR gap": round(ag["fpr_gap"], 3),
                         "Gender gap": round(v["Gender"]["eod"], 3), "Scholarship gap": round(v["Scholarship"]["eod"], 3),
                         "Age DI ratio": round(ag["di_ratio"], 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(PCOMP["note"])
        if "criteria" in PCOMP:
            st.markdown("#### Against the original acceptance criteria (unchanged after results)")
            cs = [c["id"] for c in PCOMP["criteria"]]
            st.dataframe(pd.DataFrame([{"Policy": k.split("_", 1)[1].replace("_", " "),
                                        **{c: ("✅" if v["acceptance"][c] else "❌") for c in cs},
                                        "Full fairness requirement": "✅" if v["acceptance"]["meets_full_fairness_requirement"] else "❌"}
                                       for k, v in PCOMP["policies"].items()]), hide_index=True, width="stretch")
            st.caption(" · ".join(f"{c['id']}: {c['criterion']}" for c in PCOMP["criteria"]))
            if PCOMP.get("no_policy_reliably_meets_fairness"):
                st.error("No evaluated policy reliably met the 0.05 equalized-odds limit across all required groups (age band, gender, scholarship).")
        mech = PCOMP.get("option_B_mechanism")
        if mech:
            st.markdown("#### Option B creates flags unrelated to predicted risk")
            st.markdown(f"To equalise error rates, Option B ignores the score for part of each age group whose ROC curve is above the senior group's "
                        f"and flags those appointments at the common false-positive rate ({mech['x_best_common_false_positive_rate']:.3f}). "
                        f"**{mech['share_of_flags_unrelated_to_risk']:.1%}** of its expected flags are unrelated to predicted risk, and "
                        f"**{mech['share_of_appointments_with_nonzero_flag_probability']:.1%}** of appointments have some chance of being flagged.")
            st.dataframe(pd.DataFrame([{"Age group": g.replace("_", " "), "Appointments (test)": r["n_test"], "p_ignore": round(r["p_ignore"], 3),
                                        "Flag probability whatever the score": round(r["risk_unrelated_flag_probability"], 3)}
                                       for g, r in sorted(mech["groups"].items())]), hide_index=True, width="stretch")
        eo = PCOMP["policies"].get("P1_equal_opportunity_tpr85")
        if eo:
            st.markdown("#### Alternative for committee consideration: deterministic equal opportunity (TPR 0.85)")
            st.markdown("**Equal opportunity** equalises only the share of real no-shows flagged (true-positive rate). **Equalized odds**, the Charter "
                        "requirement, also equalises false-alarm rates. This rule is deterministic and has no risk-unrelated flags, but it **does not meet "
                        "the full fairness requirement**.")
            st.dataframe(pd.DataFrame([{"Attribute": a, "TPR gap": round(eo[a]["tpr_gap"], 3), "FPR gap": round(eo[a]["fpr_gap"], 3),
                                        "Equalized-odds gap": round(eo[a]["eod"], 3),
                                        "95% range": f"{eo[a]['eod_bootstrap_95ci'][0]:.3f}-{eo[a]['eod_bootstrap_95ci'][1]:.3f}",
                                        "Groups (n test)": ", ".join(f"{g} {r['n']:,}" for g, r in eo[a]["groups"].items())}
                                       for a in ["age_band", "Gender", "Scholarship"]]), hide_index=True, width="stretch")
    if POLICY:
        st.markdown("#### The configured Option B rule, in full (auditable)")
        pr = [{"Age group": g.replace("_", " "), "Rule": (f"flag if score {r['op0'][0]} {r['op0'][1]:.3f}" if r["p0"] >= 0.999 else
                                                           f"{r['p0']:.0%}: score {r['op0'][0]} {r['op0'][1]:.3f}; {r['p1']:.0%}: score {r['op1'][0]} {r['op1'][1]:.3f}")}
              for g, r in POLICY["groups"].items()]
        st.dataframe(pd.DataFrame(pr), hide_index=True, width="stretch")
        st.caption("Where two thresholds are shown, the decision between them is a seeded random draw from the request id, logged in the explanation store.")

# ═════════ 7 ═════════
elif page.startswith("7"):
    st.title("Monitoring: drift, performance and fairness")
    if not MON:
        st.warning("No monitoring run found. Run `python -m careconnect.monitoring.monitor`.")
        st.stop()
    st.markdown('<div class="warn"><b>What this is:</b> a <b>replay</b> of held-out appointments, scored as if each period were new '
                'production data. Real outcomes are known, so performance and fairness can be checked. '
                'CareConnect is not live on hospital data, so this is not production monitoring.</div>', unsafe_allow_html=True)
    st.write("")
    FR, TH = MON["fairness_rule"], MON["thresholds"]
    emoji = {"ok": "🟢 ok", "warn": "🟠 warn", "alert": "🔴 alert", "halt": "⛔ halt", "inconclusive": "⚪ inconclusive",
             "insufficient data": "▫️ insufficient data", "not evaluable": "▫️ not evaluable"}
    R4 = MON.get("module4_rule")
    if R4:
        S4 = MON["summary"]["module4_s7_weekly"]
        st.markdown("#### Governing rule: original Module 4 §7 weekly fairness monitoring")
        st.markdown(f"Each calendar week, the realised error-rate disparity (the larger of the false-negative-rate and false-positive-rate "
                    f"gaps) is computed for **Gender** and **age group**. **Alert** above {R4['alert_above']:.2f}, **halt** above "
                    f"{R4['halt_above']:.2f}; no minimum group size (none is specified). A group with no no-shows or no attended visits "
                    "cannot be compared, and an attribute with fewer than two comparable groups is **not evaluable** - never a pass.")
        st.error(f"**Governing status over the replay: {S4['governing_status'].upper()}** - "
                 + ", ".join(f"{v} week(s) {k}" for k, v in S4["counts"].items()) + ".")
        st.dataframe(pd.DataFrame([{
            "Week": f"{w['start']} to {w['end']}", "Appointments": w["n"],
            "Age disparity (expected)": "-" if w["module4_rule"]["attributes"]["age_band"]["disparity"] is None else
            f"{w['module4_rule']['attributes']['age_band']['disparity']:.3f} ({w['module4_rule']['attributes']['age_band']['expected_disparity']:.3f})",
            "Gender disparity (expected)": "-" if w["module4_rule"]["attributes"]["Gender"]["disparity"] is None else
            f"{w['module4_rule']['attributes']['Gender']['disparity']:.3f} ({w['module4_rule']['attributes']['Gender']['expected_disparity']:.3f})",
            "Status": emoji[w["module4_rule"]["status"]], "Attribution": w["module4_rule"]["attribution"]}
            for w in MON["weeks_cycle2_rule"]]), hide_index=True, width="stretch")
        st.caption("Scholarship (the G3 stand-in) is computed too but is not part of the §7 rule. The rolling-window analysis below is "
                   "supplementary evidence about uncertainty; it does not replace or override this rule.")
    st.markdown("#### Supplementary: rolling-window uncertainty analysis (not the governing rule)")
    st.markdown(f"**Supplementary rule ({FR['window']}).** A group is evaluated only with at least {FR['min_no_shows_per_group']} no-shows and "
                f"{FR['min_attended_per_group']} attended visits. The age-group gap gets a {FR['bootstrap']}-sample bootstrap 95% interval: "
                "**halt** only if the whole interval is above 0.05, **ok** only if it is wholly at or below 0.05, otherwise **inconclusive**. "
                "Too few eligible groups gives **insufficient data**, which is never read as fair or unfair.")

    def wtable(rows):
        return pd.DataFrame([{"Window": r["batch"], "Appointments": r["n"], "ROC-AUC": None if r["auc"] is None else round(r["auc"], 3),
                              "Max input PSI": f"{r['max_feature_psi']:.2f} ({r['max_psi_feature']})", "Score PSI": round(r["score_psi"], 3),
                              "Age gap": None if r["fairness"]["eod"] is None else round(r["fairness"]["eod"], 3),
                              "95% interval": "-" if not r["fairness"]["ci95"] else f"{r['fairness']['ci95'][0]:.3f}-{r['fairness']['ci95'][1]:.3f}",
                              "Fairness status": emoji[r["fairness"]["status"]],
                              "Expected gap (Option B)": None if r.get("expected_age_eod_evaluated_groups") is None else round(r["expected_age_eod_evaluated_groups"], 3),
                              "Groups not evaluated": ", ".join(r["fairness"]["groups_not_evaluated"]) or "none",
                              "Data drift": emoji[r["status"]["data_drift"]]} for r in rows])
    SUM = MON.get("summary")
    if SUM:
        st.markdown("#### Outcomes, counted separately")
        st.dataframe(pd.DataFrame([{"Window set": n, **SUM[k]["counts"],
                                    "Expected gap > 0.05": f"{SUM[k]['windows_with_expected_gap_above_0.05']} of {SUM[k]['windows_with_expected_gap_evaluable']} evaluable"}
                                   for n, k in [("Rolling 28-day (all)", "rolling_28d_all"), ("Rolling 28-day (full windows)", "rolling_28d_full_windows_only"),
                                                ("Calendar months", "calendar_months"), ("Weekly (bootstrap rule)", "weeks_cycle2_rule")]]),
                     hide_index=True, width="stretch")
        st.warning("Under Option B the expected age-group gap exceeds 0.05 in every monitored time window used by the supplementary rolling analysis (rolling 28-day windows and "
                   "evaluable calendar months), even though it is within 0.05 on the pooled test set. 'Insufficient data' is never counted as a pass.")
    st.markdown("#### Rolling 28-day windows")
    st.dataframe(wtable(MON["rolling_28d"]), hide_index=True, width="stretch")
    st.markdown("#### Same seven weeks: cycle-1 rule vs supplementary bootstrap rule")
    cmp_ = pd.DataFrame([{"Week": a["batch"], "Cycle-1 gap": round(a["age_eod"], 3), "Cycle-1 status": emoji[a["status"]["bias_drift"]],
                          "Bootstrap status": emoji[b["fairness"]["status"]],
                          "Bootstrap 95% interval": "-" if not b["fairness"]["ci95"] else f"{b['fairness']['ci95'][0]:.3f}-{b['fairness']['ci95'][1]:.3f}",
                          "Not evaluated": ", ".join(b["fairness"]["groups_not_evaluated"]) or "none"}
                         for a, b in zip(MON["weeks_cycle1_rule"], MON["weeks_cycle2_rule"])])
    st.dataframe(cmp_, hide_index=True, width="stretch")
    st.caption("The cycle-1 rule (age only, groups with fewer than 30 no-shows skipped) halted every week. Weekly age groups are small, "
               "so a weekly point estimate is noisy; the supplementary 28-day bootstrap analysis shows that uncertainty. The governing "
               "decision remains the original Module 4 §7 weekly rule above.")
    st.markdown("#### Synthetic stress tests (labelled, not real data)")
    st.dataframe(pd.DataFrame([{"Test": t["name"], "Expected": t["expect"], "Fairness status": emoji[t["fairness"]["status"]],
                                "Data drift": emoji[t["status"]["data_drift"]], "Prediction drift": emoji[t["status"]["prediction_drift"]],
                                "Performance": emoji[t["status"]["performance"]],
                                "Not evaluated": ", ".join(t["fairness"]["groups_not_evaluated"]) or "none"} for t in MON["stress_tests"]]),
                 hide_index=True, width="stretch")
    W = pd.DataFrame(MON["weeks_cycle1_rule"])
    psi = pd.DataFrame([r["feature_psi"] for r in MON["weeks_cycle1_rule"]], index=W["batch"].str.replace("week of ", "")).T
    f = go.Figure(go.Heatmap(z=psi.values, x=psi.columns, y=[PLAIN.get(i, i) for i in psi.index], colorscale=[[0, "#EEF4F5"], [0.1, "#E9A03B"], [1, "#C0392B"]],
                             zmin=0, zmax=0.5, text=psi.round(2).values, texttemplate="%{text}", colorbar=dict(title="PSI")))
    f.update_layout(title="Data drift: Population Stability Index per input, by week (vs training data)")
    st.plotly_chart(style(f, 420), width="stretch")
    st.caption(f"PSI warn ≥ {TH['psi_warn']}, alert ≥ {TH['psi_alert']}. Early weeks show high PSI for past visits because the data start "
               "on 29 April, so nobody has history yet (a cold-start artefact, not drift). The week of 16 May shows a real change in reminder SMS volume.")
    ev = ART / "monitoring" / "evidently_drift_report.html"
    if ev.exists():
        with st.expander("Evidently AI data-drift report (last 28 days vs training data)"):
            import streamlit.components.v1 as components
            components.html(ev.read_text(), height=900, scrolling=True)
    st.caption("Drift types: data drift = inputs change; prediction drift = scores change; concept drift = the input-outcome relationship "
               "changes (performance falls without input drift); bias drift = group error-rate gaps widen.")

# ═════════ 6 ═════════
else:
    st.title("Limitations & transparency statement")
    st.markdown("### What the model can do")
    st.markdown(f"* Rank appointments by chance of a missed visit: the top 10% were missed {A['deciles'][-1]['actual']:.0%} of the time vs {A['deciles'][0]['actual']:.0%} in the bottom 10%.\n"
                f"* Explain each score in terms of the main factors behind it.\n* Support a coordinator who decides who to contact and how.")
    st.markdown("### What it cannot do")
    cm = A["confusion"]
    st.markdown(f"""
* **It is often wrong about individuals.** On {sum(cm.values()):,} unseen appointments it correctly flagged {cm['tp']:,} no-shows but also flagged {cm['fp']:,} patients who attended, and still missed {cm['fn']:,} no-shows.
* **It cannot say why** a patient may miss a visit, and it does not measure transport, childcare, cost or health literacy directly.
* **It was not trained on CrescentCare patients.** The data are public-health appointments from Brazil in 2016. The hospital's reported no-show rate (28%) is higher than in the data ({A['descriptive']['overall']:.0%}).
* **It does not meet every Module 4 target:** F1 is {OP['f1']:.2f} against a 0.55 target, and the single cut-off fails the age-group fairness limit. Option B narrows the gap only in expectation, by flagging more appointments (more false alarms) and partly at random; no evaluated policy reliably meets the fairness limit.
* **Some decisions are randomised** under the configured Option B rule, and some flags are unrelated to predicted risk. Each draw is reproducible from the logged request id, but staff and patients need this explained.
* **The Module 1 equity goal (Medicaid vs private insurance) cannot be measured here** because the dataset has no insurance field. Welfare-scholarship status is used as a rough socio-economic stand-in.
* **Scores are associations, not causes**, and only about six weeks of appointments are covered, so seasonal patterns are not captured.
""")
    st.markdown("### Transparency statement")
    st.success("CareConnect is decision-support only. It never denies, delays or restricts care, never penalises a patient, and never schedules or contacts anyone automatically. "
               "Every score comes with reasons, every outreach is a human decision, and fairness gaps are published here for the Ethics Committee.")
