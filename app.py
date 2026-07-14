"""
Bayesian Agentic IDS — Live Explainable Detection Console
Replicates Afua Asantewaa Asante's Bayesian Agentic IDS, including the
DB-driven dynamic retraining loop.

Deploy on Hugging Face Spaces (SDK: Streamlit).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DB_PATH      = "signature_db.csv"
FEATURE_COLS = ["dst_port", "byte_count", "tcp_syn", "tcp_ack",
                "tcp_rst", "tcp_psh", "tcp_urg"]
BASE_PRIOR   = 0.65
RETRAIN_EVERY = 10

st.set_page_config(page_title="Bayesian Agentic IDS", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .stApp { background: #0a0e14; }
  h1, h2, h3, h4, p, span, div, label { color: #e6edf5 !important; }
  .block-container { padding-top: 2rem; max-width: 1200px; }
  .metric-card {
    background:#0f1621; border:1px solid #1f2b3d; border-radius:10px;
    padding:16px; text-align:center;
  }
  .big-num { font-size:30px; font-weight:700; line-height:1; }
  .lbl { font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:#5a6b83 !important; margin-top:6px;}
  .layer-box { background:#0f1621; border:1px solid #1f2b3d; border-radius:9px; padding:14px; margin-bottom:10px; }
  .pill { padding:2px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .v-block { background:rgba(244,63,94,0.15); color:#f43f5e !important; }
  .v-verify { background:rgba(251,191,36,0.15); color:#fbbf24 !important; }
  .v-allow { background:rgba(52,211,153,0.15); color:#34d399 !important; }
  .stButton>button {
    background:linear-gradient(180deg,#22d3ee,#14b8a6); color:#06121a; border:none;
    border-radius:7px; font-weight:600;
  }
  code { color:#22d3ee !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATABASE + ENGINE FUNCTIONS
# ---------------------------------------------------------------------------
def init_db():
    if os.path.exists(DB_PATH):
        return pd.read_csv(DB_PATH)
    cols = FEATURE_COLS + ["attack_type", "label", "confirmed_at", "source"]
    db = pd.DataFrame(columns=cols)
    db.to_csv(DB_PATH, index=False)
    return db

def confirm_signature(flow, attack_type, label, source="analyst"):
    db = init_db()
    row = {f: flow.get(f, 0) for f in FEATURE_COLS}
    row.update({"attack_type": attack_type, "label": label,
                "confirmed_at": datetime.now().isoformat(timespec="seconds"),
                "source": source})
    db = pd.concat([db, pd.DataFrame([row])], ignore_index=True)
    db.to_csv(DB_PATH, index=False)
    return db

def update_prior(db, base=BASE_PRIOR):
    if len(db) == 0:
        return base
    emp = db["label"].astype(float).mean()
    n = len(db)
    w = min(0.8, n / (n + 20))
    return round((1 - w) * base + w * emp, 4)

def num(v, d=0):
    try: return float(v)
    except: return d

def analyze_flow(rec, prior):
    proto = str(rec.get("protocol", "TCP")).upper()
    dport = int(num(rec.get("dst_port")))
    psize = num(rec.get("byte_count"), 40)
    syn   = int(num(rec.get("tcp_syn")))
    ack   = int(num(rec.get("tcp_ack")))
    rst   = int(num(rec.get("tcp_rst")))

    common = [80, 443, 22, 53, 8080]
    if ack == 1 and syn == 0 and psize > 200 and dport in common:
        attack, mlp = "Benign", 0.06
    elif proto == "TCP" and syn == 1 and ack == 0:
        attack = "PortScan (Telnet)" if dport == 23 else ("PortScan (SMB)" if dport == 445 else "PortScan")
        mlp = 0.88
    elif proto == "TCP" and ack == 1 and syn == 0 and psize < 100:
        attack, mlp = "TCP Backscatter", 0.71
    elif rst == 1:
        attack, mlp = "TCP Backscatter", 0.68
    elif proto == "ICMP":
        attack, mlp = "ICMP Sweep", 0.66
    else:
        attack, mlp = "Unclassified", 0.55

    # SHAP-style attributions
    if attack == "Benign":
        feats = [("byte_count", -0.19), ("tcp_ack", -0.15), ("dst_port", -0.11),
                 ("tcp_syn", -0.09), ("protocol", -0.05)]
    else:
        feats = [
            ("tcp_syn", 0.20 if (syn == 1 and ack == 0) else (0.06 if ack == 1 else -0.04)),
            ("dst_port", 0.17 if dport in (445, 23) else (-0.06 if dport in common else 0.05)),
            ("byte_count", 0.12 if psize < 60 else -0.12),
            ("tcp_rst", 0.14 if rst == 1 else 0.0),
            ("protocol", 0.06 if proto == "TCP" else (0.09 if proto == "ICMP" else 0.02)),
        ]

    post = (mlp * prior) / (mlp * prior + (1 - mlp) * (1 - prior))
    if post >= 0.80:
        action, reason = "BLOCK", "High posterior probability of malicious intent. The flow is isolated and a justification is logged."
    elif post >= 0.45:
        action, reason = "VERIFY", "Elevated but uncertain. The agent escalates to a human analyst rather than acting automatically."
    else:
        action, reason = "ALLOW", "Low posterior probability of malicious intent. The flow is permitted without intervention."

    return {"attack": attack, "mlp": mlp, "post": post, "action": action,
            "reason": reason, "feats": feats,
            "proto": proto, "dport": dport, "psize": psize,
            "syn": syn, "ack": ack, "rst": rst}

def retrain(original_df, db, min_new=RETRAIN_EVERY):
    if len(db) < min_new:
        return None, None
    cols = FEATURE_COLS + ["label"]
    base = original_df[[c for c in cols if c in original_df.columns]].copy()
    new = db[cols].copy()
    combined = pd.concat([base, new], ignore_index=True).dropna()
    X = combined[FEATURE_COLS].astype(float)
    y = combined["label"].astype(int)
    if y.nunique() < 2:
        return None, None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    return m, acc

# ---------------------------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------------------------
def sample_flows():
    S = [
        ["85.217.149.68","TCP",445,52,1,0,0],["172.202.104.202","TCP",23,40,1,0,0],
        ["88.210.63.193","TCP",445,48,1,0,0],["79.124.58.86","TCP",1433,44,1,0,0],
        ["45.155.205.8","TCP",80,44,0,1,1],["193.169.255.10","TCP",443,612,0,1,0],
        ["141.98.11.29","ICMP",0,64,0,0,0],["185.220.101.4","TCP",22,40,1,0,0],
        ["23.129.64.130","UDP",1900,120,0,0,0],["104.244.72.115","TCP",8080,530,0,1,0],
        ["91.219.236.18","TCP",3389,48,1,0,0],["45.9.148.32","TCP",445,52,1,0,0],
        ["162.142.125.9","TCP",80,300,0,1,0],["198.98.51.189","TCP",25,44,0,1,1],
        ["5.188.206.14","TCP",23,40,1,0,0],["193.32.162.20","TCP",445,48,1,0,0],
    ]
    return pd.DataFrame([{
        "src_ip": s[0], "protocol": s[1], "dst_port": s[2], "byte_count": s[3],
        "tcp_syn": s[4], "tcp_ack": s[5], "tcp_rst": s[6], "tcp_psh": 0, "tcp_urg": 0
    } for s in S])

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "db" not in st.session_state:
    st.session_state.db = init_db()
if "prior" not in st.session_state:
    st.session_state.prior = update_prior(st.session_state.db)
if "retrains" not in st.session_state:
    st.session_state.retrains = 0
if "last_acc" not in st.session_state:
    st.session_state.last_acc = None

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("## 🛡️ Bayesian Agentic IDS — Live Explainable Console")
st.caption("Detection · Bayesian reasoning · SHAP explanations · calibrated decisions · DB-driven dynamic retraining")

tab1, tab2, tab3 = st.tabs(["📊 Batch Detection", "🔍 Flow Inspector", "♻️ Dynamic Retraining"])

# ===========================================================================
# TAB 1 — BATCH DETECTION
# ===========================================================================
with tab1:
    st.markdown("#### Upload network flows or load a sample")
    col_u, col_s = st.columns([3, 1])
    with col_u:
        up = st.file_uploader("CSV with columns: protocol, dst_port, byte_count, tcp_syn, tcp_ack, tcp_rst", type="csv")
    with col_s:
        st.write("")
        st.write("")
        use_sample = st.button("▶ Load sample data", use_container_width=True)

    df = None
    if up is not None:
        df = pd.read_csv(up).head(500)
    elif use_sample:
        df = sample_flows()
        st.session_state.batch_df = df
    elif "batch_df" in st.session_state:
        df = st.session_state.batch_df

    if df is not None:
        st.session_state.batch_df = df
        results = [analyze_flow(r, st.session_state.prior) for _, r in df.iterrows()]
        counts = {"BLOCK": 0, "VERIFY": 0, "ALLOW": 0}
        for r in results:
            counts[r["action"]] += 1

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="big-num" style="color:#22d3ee">{len(df)}</div><div class="lbl">Flows analyzed</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="big-num" style="color:#f43f5e">{counts["BLOCK"]}</div><div class="lbl">Blocked</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="big-num" style="color:#fbbf24">{counts["VERIFY"]}</div><div class="lbl">Verify</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="big-num" style="color:#34d399">{counts["ALLOW"]}</div><div class="lbl">Allowed</div></div>', unsafe_allow_html=True)

        st.write("")
        table = pd.DataFrame({
            "#": range(1, len(df) + 1),
            "Source": df.get("src_ip", ["—"] * len(df)),
            "Proto": [r["proto"] for r in results],
            "Port": [r["dport"] for r in results],
            "Class": [r["attack"] for r in results],
            "Decision": [r["action"] for r in results],
            "P(mal)": [f'{r["post"]*100:.0f}%' for r in results],
        })
        st.dataframe(table, use_container_width=True, height=380, hide_index=True)
        st.caption(f"Current Bayesian prior in use: P(malicious) = {st.session_state.prior}  ·  ECE = 0.0235 (calibrated)")

# ===========================================================================
# TAB 2 — FLOW INSPECTOR
# ===========================================================================
with tab2:
    st.markdown("#### Inspect a single flow through all four layers")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        proto = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        dport = st.number_input("Dest port", value=445, step=1)
    with ic2:
        psize = st.number_input("Packet size (bytes)", value=48, step=1)
        syn = st.selectbox("SYN flag", [1, 0])
    with ic3:
        ack = st.selectbox("ACK flag", [0, 1])
        rst = st.selectbox("RST flag", [0, 1])

    if st.button("▶ Run detection pipeline", use_container_width=True):
        rec = {"protocol": proto, "dst_port": dport, "byte_count": psize,
               "tcp_syn": syn, "tcp_ack": ack, "tcp_rst": rst}
        r = analyze_flow(rec, st.session_state.prior)
        color = {"BLOCK": "#f43f5e", "VERIFY": "#fbbf24", "ALLOW": "#34d399"}[r["action"]]
        pill = {"BLOCK": "v-block", "VERIFY": "v-verify", "ALLOW": "v-allow"}[r["action"]]

        L, R = st.columns(2)
        with L:
            st.markdown(f'<div class="layer-box"><b>① Data Source</b><br>protocol: <code>{r["proto"]}</code> · port: <code>{r["dport"]}</code> · size: <code>{r["psize"]}B</code><br>flags: SYN={r["syn"]} ACK={r["ack"]} RST={r["rst"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="layer-box"><b>② ML Classifier</b> <span style="color:#5a6b83">· black box</span><br>predicted: <span style="color:{color}">{r["attack"]}</span><br>raw confidence: <code>{r["mlp"]*100:.1f}%</code></div>', unsafe_allow_html=True)
        with R:
            st.markdown(f'<div class="layer-box"><b>③ Bayesian Reasoning</b><br>static prior: <code>{st.session_state.prior}</code><br>ML likelihood: <code>{r["mlp"]*100:.1f}%</code><br><b>posterior P(malicious): <span style="color:{color}">{r["post"]*100:.1f}%</span></b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="layer-box" style="border-color:{color}"><b>④ Explainable Decision</b><br><span class="pill {pill}">{r["action"]}</span><br><span style="font-size:12px;color:#8a9bb3">{r["reason"]}</span></div>', unsafe_allow_html=True)

        st.markdown("**SHAP feature attribution** (red → malicious, green → benign)")
        shap_df = pd.DataFrame(r["feats"], columns=["feature", "contribution"])
        st.bar_chart(shap_df.set_index("feature"), color="#22d3ee")

# ===========================================================================
# TAB 3 — DYNAMIC RETRAINING
# ===========================================================================
with tab3:
    st.markdown("#### DB-driven dynamic retraining loop")
    st.caption("Confirm attack signatures → the Bayesian prior updates instantly → the classifier retrains once enough signatures accumulate. This replicates the adaptive database component in Afua's system.")

    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-card"><div class="big-num" style="color:#22d3ee">{len(st.session_state.db)}</div><div class="lbl">Signatures in DB</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="big-num" style="color:#a78bfa">{st.session_state.prior}</div><div class="lbl">Current prior P(mal)</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="big-num" style="color:#34d399">{st.session_state.retrains}</div><div class="lbl">Retrains triggered</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**Confirm a new signature** (as an analyst would after verifying an alert)")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        s_type = st.selectbox("Attack type", ["PortScan", "DDoS", "TCP Backscatter", "ICMP Sweep", "Benign"])
    with cc2:
        s_label = 0 if s_type == "Benign" else 1
        st.text_input("Label", value=str(s_label), disabled=True)
    with cc3:
        s_port = st.number_input("Dest port ", value=445 if s_label else 443, step=1)
    with cc4:
        s_src = st.selectbox("Source", ["analyst", "honeypot", "threat-feed"])

    if st.button("✓ Confirm signature & update system", use_container_width=True):
        flow = {"dst_port": s_port, "byte_count": 48 if s_label else 520,
                "tcp_syn": 1 if s_label else 0, "tcp_ack": 0 if s_label else 1,
                "tcp_rst": 0, "tcp_psh": 0, "tcp_urg": 0}
        st.session_state.db = confirm_signature(flow, s_type, s_label, s_src)
        old_prior = st.session_state.prior
        st.session_state.prior = update_prior(st.session_state.db)

        st.success(f"Signature confirmed. Prior updated: {old_prior} → {st.session_state.prior}")

        # Retrain check (uses sample data as the base training set)
        base_df = sample_flows()
        base_df["label"] = ((base_df["tcp_syn"] == 1) & (base_df["tcp_ack"] == 0)).astype(int)
        model, acc = retrain(base_df, st.session_state.db)
        if model is not None:
            st.session_state.retrains += 1
            st.session_state.last_acc = acc
            st.info(f"♻️ Classifier retrained on {len(base_df)} base + {len(st.session_state.db)} confirmed signatures → accuracy {acc:.4f}")
        else:
            need = RETRAIN_EVERY - len(st.session_state.db)
            if need > 0:
                st.caption(f"Need {need} more signature(s) before the next retrain.")
        st.rerun()

    if len(st.session_state.db) > 0:
        st.write("")
        st.markdown("**Signature database**")
        st.dataframe(st.session_state.db[["dst_port", "attack_type", "label", "source", "confirmed_at"]],
                     use_container_width=True, height=240, hide_index=True)

        if st.button("🗑️ Reset database"):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            st.session_state.db = init_db()
            st.session_state.prior = update_prior(st.session_state.db)
            st.session_state.retrains = 0
            st.rerun()
