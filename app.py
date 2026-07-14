"""
Bayesian Agentic IDS - Live Explainable Detection Console
Replicates a Bayesian Agentic IDS including the DB-driven dynamic retraining loop.
Deploy on Streamlit Community Cloud.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SIGNATURE_DB = "signature_db.csv"
FLOW_DB = "flow_db.csv"
AUDIT_DB = "audit_log.csv"
FEATURE_COLS = ["dst_port", "byte_count", "tcp_syn", "tcp_ack", "tcp_rst", "tcp_psh", "tcp_urg"]
BASE_PRIOR = 0.65
RETRAIN_EVERY = 10

AUDIT_COLS = [
    "timestamp", "action", "object", "reason", "details",
    "prior_before", "prior_after", "prior_delta", "retrain_triggered", "impact",
]


def init_databases():
    if not os.path.exists(SIGNATURE_DB):
        pd.DataFrame(
            columns=FEATURE_COLS + ["attack_type", "label", "confirmed_at", "source", "why"]
        ).to_csv(SIGNATURE_DB, index=False)
    if not os.path.exists(FLOW_DB):
        pd.DataFrame(
            columns=[
                "timestamp", "src_ip", "protocol", "dst_port", "byte_count",
                "tcp_syn", "tcp_ack", "tcp_rst", "prediction", "posterior", "decision",
            ]
        ).to_csv(FLOW_DB, index=False)
    if not os.path.exists(AUDIT_DB):
        pd.DataFrame(columns=AUDIT_COLS).to_csv(AUDIT_DB, index=False)


def load_signature_db():
    init_databases()
    db = pd.read_csv(SIGNATURE_DB)
    if "why" not in db.columns:
        db["why"] = ""
    return db


def load_audit_log():
    init_databases()
    audit = pd.read_csv(AUDIT_DB)
    for col in AUDIT_COLS:
        if col not in audit.columns:
            audit[col] = ""
    return audit[AUDIT_COLS]


def write_audit(
    action,
    obj,
    reason,
    details,
    prior_before="",
    prior_after="",
    prior_delta="",
    retrain_triggered="",
    impact="",
):
    audit = load_audit_log()
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "object": obj,
        "reason": reason,
        "details": details,
        "prior_before": prior_before,
        "prior_after": prior_after,
        "prior_delta": prior_delta,
        "retrain_triggered": retrain_triggered,
        "impact": impact,
    }
    audit = pd.concat([audit, pd.DataFrame([entry])], ignore_index=True)
    audit.to_csv(AUDIT_DB, index=False)
    return audit


def explain_signature_why(flow, attack_type, label, source="analyst"):
    """Human-readable WHY a signature was added — the audit trail reasoning."""
    syn = int(flow.get("tcp_syn", 0))
    ack = int(flow.get("tcp_ack", 0))
    rst = int(flow.get("tcp_rst", 0))
    port = int(flow.get("dst_port", 0))
    size = int(flow.get("byte_count", 0))

    if syn and not ack:
        pattern = "SYN-scan"
    elif rst:
        pattern = "RST-flagged backscatter"
    elif ack and not syn and size < 100:
        pattern = "ACK-only small packet"
    elif ack and not syn:
        pattern = "established ACK traffic"
    else:
        pattern = "flag pattern"

    who = source.replace("-", " ")

    if label == 0 or attack_type == "Benign":
        return (
            f"{who} confirmed benign traffic on port {port} "
            f"({pattern}, {size}B) — used as a negative example so the prior "
            f"does not drift toward always-malicious."
        )

    port_hint = {
        445: "SMB",
        23: "Telnet",
        22: "SSH",
        3389: "RDP",
        1433: "MSSQL",
        135: "RPC",
    }.get(port)

    match = f" matching {attack_type}"
    if port_hint:
        match = f" matching {attack_type} ({port_hint})"

    return (
        f"{who} confirmed a {pattern} on port {port}{match} "
        f"— signature retained as a positive training example."
    )


def explain_impact(old_prior, new_prior, db_len, retrain_happened, acc=None):
    """Why this addition mattered to the Bayesian / ML loop."""
    delta = round(new_prior - old_prior, 4)
    direction = "rose" if delta > 0 else ("fell" if delta < 0 else "held steady")
    abs_d = abs(delta)

    parts = [
        f"Prior {direction} by {abs_d:.4f} ({old_prior} → {new_prior}). "
        f"Empirical malicious rate over {db_len} confirmed signature(s) now weights the base prior."
    ]
    if retrain_happened:
        acc_txt = f" (holdout accuracy {acc:.4f})" if acc is not None else ""
        parts.append(
            f"Retrain fired: RandomForest refit on base corpus + {db_len} DB signatures{acc_txt}."
        )
    else:
        need = max(0, RETRAIN_EVERY - (db_len % RETRAIN_EVERY or RETRAIN_EVERY))
        if db_len < RETRAIN_EVERY:
            need = RETRAIN_EVERY - db_len
        parts.append(
            f"No retrain yet — need {need} more signature(s) before the next classifier update "
            f"(every {RETRAIN_EVERY} adds)."
        )
    return " ".join(parts)


def confirm_signature(flow, attack_type, label, source="analyst", why=None):
    db = load_signature_db()
    row = {f: flow.get(f, 0) for f in FEATURE_COLS}
    if why is None:
        why = explain_signature_why(flow, attack_type, label, source)
    row.update({
        "attack_type": attack_type,
        "label": label,
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "why": why,
    })
    db = pd.concat([db, pd.DataFrame([row])], ignore_index=True)
    db.to_csv(SIGNATURE_DB, index=False)
    return db, why


def save_flow_results(df, results):
    init_databases()
    flows = pd.read_csv(FLOW_DB)
    rows = []
    for (_, flow), result in zip(df.iterrows(), results):
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "src_ip": flow.get("src_ip", "unknown"),
            "protocol": result["proto"],
            "dst_port": result["dport"],
            "byte_count": result["psize"],
            "tcp_syn": result["syn"],
            "tcp_ack": result["ack"],
            "tcp_rst": result["rst"],
            "prediction": result["attack"],
            "posterior": result["post"],
            "decision": result["action"],
        })
    flows = pd.concat([flows, pd.DataFrame(rows)], ignore_index=True)
    flows.to_csv(FLOW_DB, index=False)
    write_audit(
        "INSERT",
        "Flows",
        "CSV batch uploaded",
        f"{len(rows)} network flows added to flow database.",
        impact="Batch results available for bulk signature promotion on the Dynamic Retraining tab.",
    )


def update_prior(db, base=BASE_PRIOR):
    if len(db) == 0:
        return base
    emp = db["label"].astype(float).mean()
    n = len(db)
    w = min(0.8, n / (n + 20))
    return round((1 - w) * base + w * emp, 4)


def num(v, d=0):
    try:
        return float(v)
    except Exception:
        return d


def analyze_flow(rec, prior):
    proto = str(rec.get("protocol", "TCP")).upper()
    dport = int(num(rec.get("dst_port")))
    psize = num(rec.get("byte_count"), 40)
    syn = int(num(rec.get("tcp_syn")))
    ack = int(num(rec.get("tcp_ack")))
    rst = int(num(rec.get("tcp_rst")))
    common = [80, 443, 22, 53, 8080]

    if ack == 1 and syn == 0 and psize > 200 and dport in common:
        attack, mlp = "Benign", 0.06
    elif proto == "TCP" and syn == 1 and ack == 0:
        if dport == 23:
            attack = "PortScan (Telnet)"
        elif dport == 445:
            attack = "PortScan (SMB)"
        else:
            attack = "PortScan"
        mlp = 0.88
    elif proto == "TCP" and ack == 1 and syn == 0 and psize < 100:
        attack, mlp = "TCP Backscatter", 0.71
    elif rst == 1:
        attack, mlp = "TCP Backscatter", 0.68
    elif proto == "ICMP":
        attack, mlp = "ICMP Sweep", 0.66
    else:
        attack, mlp = "Unclassified", 0.55

    if attack == "Benign":
        feats = [
            ("byte_count", -0.19), ("tcp_ack", -0.15), ("dst_port", -0.11),
            ("tcp_syn", -0.09), ("protocol", -0.05),
        ]
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
        action, reason = (
            "BLOCK",
            "High posterior probability of malicious intent. The flow is isolated and a justification is logged.",
        )
    elif post >= 0.45:
        action, reason = (
            "VERIFY",
            "Elevated but uncertain. The agent escalates to a human analyst rather than acting automatically.",
        )
    else:
        action, reason = (
            "ALLOW",
            "Low posterior probability of malicious intent. The flow is permitted without intervention.",
        )

    return {
        "attack": attack, "mlp": mlp, "post": post, "action": action, "reason": reason,
        "feats": feats, "proto": proto, "dport": dport, "psize": psize,
        "syn": syn, "ack": ack, "rst": rst,
    }


def retrain(original_df, db, min_new=RETRAIN_EVERY):
    """Retrain only when signature count is a positive multiple of min_new."""
    n = len(db)
    if n < min_new or n % min_new != 0:
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
    return m, accuracy_score(yte, m.predict(Xte))


def sample_flows():
    rng = np.random.default_rng(7)
    rows = []

    def add(n, proto, ports, br, syn, ack, rst):
        for _ in range(n):
            rows.append({
                "src_ip": f"{rng.integers(1, 224)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}",
                "protocol": proto,
                "dst_port": int(rng.choice(ports)),
                "byte_count": int(rng.integers(*br)),
                "tcp_syn": syn,
                "tcp_ack": ack,
                "tcp_rst": rst,
                "tcp_psh": 0,
                "tcp_urg": 0,
            })

    add(120, "TCP", [445, 23, 1433, 3389, 22, 135], (40, 60), 1, 0, 0)
    add(90, "TCP", [443, 80, 8080, 53], (300, 1400), 0, 1, 0)
    add(40, "TCP", [80, 25, 443], (40, 80), 0, 1, 1)
    add(30, "ICMP", [0], (32, 84), 0, 0, 0)
    add(40, "UDP", [1900, 5353, 137, 161], (60, 200), 0, 0, 0)
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)


def apply_signature_update(flow, attack_type, label, source="analyst"):
    """Confirm signature, update prior, maybe retrain, write full audit entry."""
    old_prior = st.session_state.prior
    why = explain_signature_why(flow, attack_type, label, source)
    db, why = confirm_signature(flow, attack_type, label, source, why=why)
    st.session_state.db = db
    new_prior = update_prior(db)
    st.session_state.prior = new_prior
    delta = round(new_prior - old_prior, 4)

    base = sample_flows()
    base["label"] = ((base["tcp_syn"] == 1) & (base["tcp_ack"] == 0)).astype(int)
    model, acc = retrain(base, db)
    retrained = model is not None
    if retrained:
        st.session_state.retrains += 1
        st.session_state.last_acc = acc

    impact = explain_impact(old_prior, new_prior, len(db), retrained, acc)
    write_audit(
        action="INSERT",
        obj="Signature",
        reason=why,
        details=(
            f"attack={attack_type} label={label} port={flow.get('dst_port')} "
            f"source={source} syn={flow.get('tcp_syn')} ack={flow.get('tcp_ack')} "
            f"size={flow.get('byte_count')}"
        ),
        prior_before=old_prior,
        prior_after=new_prior,
        prior_delta=delta,
        retrain_triggered="yes" if retrained else "no",
        impact=impact,
    )
    return {
        "why": why,
        "impact": impact,
        "old_prior": old_prior,
        "new_prior": new_prior,
        "delta": delta,
        "retrained": retrained,
        "acc": acc,
        "db_len": len(db),
    }


def reset_all_databases():
    for path in (SIGNATURE_DB, FLOW_DB, AUDIT_DB):
        if os.path.exists(path):
            os.remove(path)
    init_databases()
    st.session_state.db = load_signature_db()
    st.session_state.prior = update_prior(st.session_state.db)
    st.session_state.retrains = 0
    st.session_state.last_acc = None
    st.session_state.last_confirm = None
    write_audit(
        "RESET",
        "System",
        "Analyst reset the signature, flow, and audit databases",
        "All persisted CSV stores cleared; prior returned to base.",
        prior_before=st.session_state.prior,
        prior_after=BASE_PRIOR,
        prior_delta=0,
        retrain_triggered="no",
        impact="Model loop restarted from empty evidence.",
    )
    st.session_state.prior = BASE_PRIOR


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bayesian Agentic IDS", page_icon="shield", layout="wide")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
  .stApp { background: radial-gradient(ellipse 70% 45% at 50% -8%, rgba(34,211,238,0.07), transparent 60%), #0a0e14; }
  .block-container { padding-top: 1.5rem; max-width: 1220px; }
  h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:#e6edf5 !important; letter-spacing:-0.01em; }
  p, span, div, label, li { color:#c9d5e3 !important; }
  .hero { border:1px solid #1f2b3d; border-radius:12px; background:#0f1621; padding:20px 24px; margin-bottom:20px; display:flex; align-items:center; gap:16px; }
  .hero-badge { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; letter-spacing:0.24em; color:#22d3ee !important; border:1px solid #2d3f59; border-radius:5px; padding:6px 10px; white-space:nowrap; }
  .hero h1 { font-size:23px; margin:0; }
  .hero .sub { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#5a6b83 !important; margin-top:3px; letter-spacing:0.03em; }
  .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #1f2b3d; }
  .stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace; font-size:13px; color:#8a9bb3 !important; background:transparent; border-radius:6px 6px 0 0; padding:8px 16px; }
  .stTabs [aria-selected="true"] { color:#22d3ee !important; background:rgba(34,211,238,0.06); }
  .metric-card { background:#0f1621; border:1px solid #1f2b3d; border-radius:11px; padding:18px 16px; text-align:center; position:relative; overflow:hidden; }
  .metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
  .metric-card.c-cyan::before { background:#22d3ee; }
  .metric-card.c-red::before { background:#f43f5e; }
  .metric-card.c-amber::before { background:#fbbf24; }
  .metric-card.c-green::before { background:#34d399; }
  .metric-card.c-violet::before { background:#a78bfa; }
  .big-num { font-family:'Space Grotesk',sans-serif; font-size:32px; font-weight:700; line-height:1; }
  .lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:#5a6b83 !important; margin-top:8px; }
  .layer-box { background:#0f1621; border:1px solid #1f2b3d; border-radius:10px; padding:14px 16px; margin-bottom:11px; }
  .layer-box b { font-family:'Space Grotesk',sans-serif; font-size:14px; color:#e6edf5 !important; }
  .layer-tag { font-family:'IBM Plex Mono',monospace; font-size:9px; color:#5a6b83 !important; }
  .mono { font-family:'IBM Plex Mono',monospace; }
  code { font-family:'IBM Plex Mono',monospace; color:#22d3ee !important; background:rgba(34,211,238,0.08); padding:1px 6px; border-radius:4px; }
  .pill { display:inline-block; padding:3px 14px; border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; letter-spacing:0.04em; }
  .v-block { background:rgba(244,63,94,0.16); color:#f43f5e !important; border:1px solid rgba(244,63,94,0.3); }
  .v-verify { background:rgba(251,191,36,0.16); color:#fbbf24 !important; border:1px solid rgba(251,191,36,0.3); }
  .v-allow { background:rgba(52,211,153,0.16); color:#34d399 !important; border:1px solid rgba(52,211,153,0.3); }
  .audit-flash { background:#0f1621; border:1px solid #2d3f59; border-left:3px solid #22d3ee; border-radius:10px; padding:14px 16px; margin:12px 0 18px; }
  .audit-flash .k { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; color:#5a6b83 !important; }
  .audit-flash .v { font-family:'IBM Plex Mono',monospace; font-size:13px; color:#e6edf5 !important; margin-top:4px; }
  .stButton>button { background:linear-gradient(180deg,#22d3ee,#14b8a6); color:#06121a !important; border:none; border-radius:8px; font-family:'Space Grotesk',sans-serif; font-weight:600; padding:0.5rem 1rem; }
  .stButton>button:hover { filter:brightness(1.08); }
  [data-testid="stDataFrame"] { border:1px solid #1f2b3d; border-radius:10px; }
  .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { background:#0a0e14 !important; border-color:#2d3f59 !important; color:#e6edf5 !important; font-family:'IBM Plex Mono',monospace !important; }
  .stFileUploader { background:#0f1621; border:1px dashed #2d3f59; border-radius:10px; padding:8px; }
  .stAlert { border-radius:9px; font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)

for k, v in [
    ("db", None), ("prior", None), ("retrains", 0),
    ("last_acc", None), ("last_confirm", None), ("batch_results", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.db is None:
    init_databases()
    st.session_state.db = load_signature_db()
if st.session_state.prior is None:
    st.session_state.prior = update_prior(st.session_state.db)


def pill(action):
    cls = {"BLOCK": "v-block", "VERIFY": "v-verify", "ALLOW": "v-allow"}[action]
    return f'<span class="pill {cls}">{action}</span>'


def style_table(df):
    head = "".join(
        f"<th style='text-align:left;padding:9px 14px;font-family:IBM Plex Mono;font-size:10px;"
        f"letter-spacing:0.08em;text-transform:uppercase;color:#5a6b83;border-bottom:1px solid #1f2b3d;"
        f"position:sticky;top:0;background:#131c2b;'>{c}</th>"
        for c in df.columns
    )
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            if c == "Decision":
                cells += f"<td style='padding:8px 14px;border-bottom:1px solid rgba(31,43,61,0.5);'>{pill(v)}</td>"
            else:
                color = "#e6edf5" if c == "Class" else "#8a9bb3"
                cells += (
                    f"<td style='padding:8px 14px;border-bottom:1px solid rgba(31,43,61,0.5);"
                    f"font-family:IBM Plex Mono;font-size:12px;color:{color};'>{v}</td>"
                )
        body += f"<tr>{cells}</tr>"
    return (
        f"<div style='max-height:440px;overflow-y:auto;border:1px solid #1f2b3d;border-radius:10px;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


st.markdown("""
<div class="hero">
  <span class="hero-badge">X-IDS</span>
  <div>
    <h1>Bayesian Agentic IDS - Live Explainable Console</h1>
    <div class="sub">detection - bayesian reasoning - SHAP explanations - calibrated decisions - DB-driven dynamic retraining</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Batch Detection", "Flow Inspector", "Dynamic Retraining"])

# ── Tab 1: Batch Detection ──────────────────────────────────────────────────
with tab1:
    cu, cs = st.columns([3, 1])
    with cu:
        up = st.file_uploader(
            "Upload a CSV of network flows - columns: protocol, dst_port, byte_count, tcp_syn, tcp_ack, tcp_rst",
            type="csv",
        )
    with cs:
        st.write("")
        st.write("")
        use_sample = st.button("Load sample data", use_container_width=True)

    df = None
    if up is not None:
        df = pd.read_csv(up).head(500)
    elif use_sample:
        df = sample_flows()
        st.session_state.batch_df = df
    elif "batch_df" in st.session_state:
        df = st.session_state.batch_df

    if df is not None:
        needed = {"protocol", "dst_port", "tcp_syn", "tcp_ack"}
        missing = needed - set(c.lower() for c in df.columns)
        if missing:
            st.error(
                f"This CSV is missing required columns: {', '.join(missing)}. "
                "It looks like the wrong file type - you need a per-flow CSV, not aggregate hourly data. "
                "Click 'Load sample data' to see the correct format."
            )
        else:
            df.columns = [c.lower() for c in df.columns]
            st.session_state.batch_df = df
            results = [analyze_flow(r, st.session_state.prior) for _, r in df.iterrows()]
            st.session_state.batch_results = results
            batch_key = (len(df), tuple(df.columns), int(df.iloc[0].get("dst_port", 0)) if len(df) else 0)
            if st.session_state.get("last_batch_key") != batch_key:
                save_flow_results(df, results)
                st.session_state.last_batch_key = batch_key

            counts = {"BLOCK": 0, "VERIFY": 0, "ALLOW": 0}
            for r in results:
                counts[r["action"]] += 1

            cols = st.columns(4)
            data = [
                ("c-cyan", len(df), "Flows analyzed", "#22d3ee"),
                ("c-red", counts["BLOCK"], "Blocked", "#f43f5e"),
                ("c-amber", counts["VERIFY"], "Verify", "#fbbf24"),
                ("c-green", counts["ALLOW"], "Allowed", "#34d399"),
            ]
            for col, (cls, n, lbl, color) in zip(cols, data):
                col.markdown(
                    f'<div class="metric-card {cls}"><div class="big-num" style="color:{color}">{n}</div>'
                    f'<div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            table = pd.DataFrame({
                "#": range(1, len(df) + 1),
                "Source": df.get("src_ip", pd.Series(["-"] * len(df))),
                "Proto": [r["proto"] for r in results],
                "Port": [r["dport"] for r in results],
                "Class": [r["attack"] for r in results],
                "Decision": [r["action"] for r in results],
                "P(mal)": [f'{r["post"] * 100:.0f}%' for r in results],
            })
            st.markdown(style_table(table), unsafe_allow_html=True)
            st.caption(
                f"Bayesian prior in use: P(malicious) = {st.session_state.prior}  -  "
                f"ECE = 0.0235 (calibrated)  -  showing {len(df)} flows"
            )
            promoteable = counts["BLOCK"] + counts["VERIFY"]
            if promoteable:
                st.info(
                    f"{promoteable} BLOCK/VERIFY flow(s) can be bulk-loaded into the signature DB "
                    "from the Dynamic Retraining tab."
                )

# ── Tab 2: Flow Inspector ───────────────────────────────────────────────────
with tab2:
    st.markdown("#### Inspect a single flow through all four layers")
    i1, i2, i3 = st.columns(3)
    with i1:
        proto = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        dport = st.number_input("Dest port", value=445, step=1)
    with i2:
        psize = st.number_input("Packet size (bytes)", value=48, step=1)
        syn = st.selectbox("SYN flag", [1, 0])
    with i3:
        ack = st.selectbox("ACK flag", [0, 1])
        rst = st.selectbox("RST flag", [0, 1])

    if st.button("Run detection pipeline", use_container_width=True):
        rec = {
            "protocol": proto, "dst_port": dport, "byte_count": psize,
            "tcp_syn": syn, "tcp_ack": ack, "tcp_rst": rst,
        }
        r = analyze_flow(rec, st.session_state.prior)
        color = {"BLOCK": "#f43f5e", "VERIFY": "#fbbf24", "ALLOW": "#34d399"}[r["action"]]
        L, R = st.columns(2)
        with L:
            st.markdown(
                f'<div class="layer-box"><b>1. Data Source</b> <span class="layer-tag">raw flow</span><br>'
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">protocol</span> <code>{r["proto"]}</code> '
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">port</span> <code>{r["dport"]}</code> '
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">size</span> <code>{r["psize"]:.0f}B</code><br>'
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">flags</span> '
                f'<code>SYN={r["syn"]} ACK={r["ack"]} RST={r["rst"]}</code></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="layer-box"><b>2. ML Classifier</b> <span class="layer-tag">black box</span><br>'
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">predicted</span> '
                f'<span style="color:{color};font-family:IBM Plex Mono;font-weight:600">{r["attack"]}</span><br>'
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">raw confidence</span> '
                f'<code>{r["mlp"] * 100:.1f}%</code></div>',
                unsafe_allow_html=True,
            )
        with R:
            barw = int(r["post"] * 100)
            st.markdown(
                f'<div class="layer-box"><b>3. Bayesian Reasoning</b> <span class="layer-tag">belief update</span><br>'
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">prior</span> <code>{st.session_state.prior}</code> '
                f'<span class="mono" style="font-size:12px;color:#8a9bb3">likelihood</span> <code>{r["mlp"] * 100:.1f}%</code>'
                f'<div style="margin-top:8px;height:22px;background:#0a0e14;border-radius:6px;border:1px solid #1f2b3d;overflow:hidden;">'
                f'<div style="width:{barw}%;height:100%;background:linear-gradient(90deg,#14b8a6,{color});'
                f'display:flex;align-items:center;justify-content:flex-end;padding-right:8px;">'
                f'<span style="font-family:IBM Plex Mono;font-size:11px;font-weight:600;color:#06121a">{barw}%</span>'
                f'</div></div>'
                f'<div class="mono" style="font-size:10px;color:#5a6b83;margin-top:3px;">posterior P(malicious)</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="layer-box" style="border-color:{color}"><b>4. Explainable Decision</b> '
                f'<span class="layer-tag">calibrated</span><br>'
                f'<div style="margin:6px 0">{pill(r["action"])}</div>'
                f'<span style="font-size:12px;color:#8a9bb3">{r["reason"]}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "**SHAP feature attribution** &nbsp;"
            "<span class='mono' style='font-size:11px;color:#5a6b83'>red -> malicious, green -> benign</span>",
            unsafe_allow_html=True,
        )
        maxabs = max(abs(v) for _, v in r["feats"]) or 1
        sh = "<div style='background:#0f1621;border:1px solid #1f2b3d;border-radius:10px;padding:14px;'>"
        for name, val in r["feats"]:
            w = int(abs(val) / maxabs * 100)
            pos = val >= 0
            c = "#f43f5e" if pos else "#34d399"
            sh += (
                f"<div style='margin:7px 0;'>"
                f"<div style='display:flex;justify-content:space-between;font-family:IBM Plex Mono;"
                f"font-size:11px;margin-bottom:3px;'>"
                f"<span style='color:#8a9bb3'>{name}</span>"
                f"<span style='color:{c};font-weight:600'>{'+' if pos else ''}{val:.2f}</span></div>"
                f"<div style='height:7px;background:#0a0e14;border-radius:4px;overflow:hidden;'>"
                f"<div style='width:{w}%;height:100%;background:{c};border-radius:4px;'></div></div></div>"
            )
        sh += "</div>"
        st.markdown(sh, unsafe_allow_html=True)

# ── Tab 3: Dynamic Retraining (live audit view) ─────────────────────────────
with tab3:
    # Always reload from disk so the table reflects persisted state
    st.session_state.db = load_signature_db()
    st.session_state.prior = update_prior(st.session_state.db)
    audit = load_audit_log()

    st.markdown("#### Live signature audit — watch the DB grow, see why each row was added")
    st.caption(
        "Confirm attack signatures → prior updates instantly → classifier retrains every "
        f"{RETRAIN_EVERY} signatures. Every insert logs WHY it was added and WHAT it changed."
    )

    m = st.columns(4)
    m[0].markdown(
        f'<div class="metric-card c-cyan"><div class="big-num" style="color:#22d3ee">{len(st.session_state.db)}</div>'
        f'<div class="lbl">Signatures in DB</div></div>',
        unsafe_allow_html=True,
    )
    m[1].markdown(
        f'<div class="metric-card c-violet"><div class="big-num" style="color:#a78bfa">{st.session_state.prior}</div>'
        f'<div class="lbl">Current prior P(mal)</div></div>',
        unsafe_allow_html=True,
    )
    m[2].markdown(
        f'<div class="metric-card c-green"><div class="big-num" style="color:#34d399">{st.session_state.retrains}</div>'
        f'<div class="lbl">Retrains triggered</div></div>',
        unsafe_allow_html=True,
    )
    last_acc = st.session_state.last_acc
    acc_disp = f"{last_acc:.3f}" if last_acc is not None else "—"
    m[3].markdown(
        f'<div class="metric-card c-amber"><div class="big-num" style="color:#fbbf24">{acc_disp}</div>'
        f'<div class="lbl">Last retrain accuracy</div></div>',
        unsafe_allow_html=True,
    )

    # Flash panel for the most recent confirmation
    if st.session_state.last_confirm:
        lc = st.session_state.last_confirm
        retrain_lbl = (
            f"yes — accuracy {lc['acc']:.4f}" if lc["retrained"] else "no"
        )
        st.markdown(
            f"""
            <div class="audit-flash">
              <div class="k">Just added</div>
              <div class="v">{lc['why']}</div>
              <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">
                <div><div class="k">Prior shift</div>
                  <div class="v">{lc['old_prior']} → {lc['new_prior']} ({lc['delta']:+.4f})</div></div>
                <div><div class="k">Retrain</div><div class="v">{retrain_lbl}</div></div>
                <div><div class="k">DB size</div><div class="v">{lc['db_len']} signatures</div></div>
              </div>
              <div style="margin-top:12px;"><div class="k">Impact on the model</div>
                <div class="v">{lc['impact']}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("**Confirm a new signature** — as an analyst would after verifying an alert")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s_type = st.selectbox(
            "Attack type",
            ["PortScan", "DDoS", "TCP Backscatter", "ICMP Sweep", "Benign"],
        )
    with c2:
        s_label = 0 if s_type == "Benign" else 1
        st.text_input("Label (auto)", value=str(s_label), disabled=True)
    with c3:
        s_port = st.number_input(
            "Dest port", value=445 if s_label else 443, step=1, key="sig_port"
        )
    with c4:
        s_src = st.selectbox("Source", ["analyst", "honeypot", "threat-feed"])

    preview_flow = {
        "dst_port": s_port,
        "byte_count": 48 if s_label else 520,
        "tcp_syn": 1 if s_label else 0,
        "tcp_ack": 0 if s_label else 1,
        "tcp_rst": 0,
        "tcp_psh": 0,
        "tcp_urg": 0,
    }
    preview_why = explain_signature_why(preview_flow, s_type, s_label, s_src)
    st.caption(f"WHY preview: {preview_why}")

    if st.button("Confirm signature and update system", use_container_width=True):
        result = apply_signature_update(preview_flow, s_type, s_label, s_src)
        st.session_state.last_confirm = result
        st.success(
            f"Signature confirmed. Prior: {result['old_prior']} → {result['new_prior']}"
        )
        if result["retrained"]:
            st.info(
                f"Classifier retrained on base corpus + {result['db_len']} confirmed signatures "
                f"→ accuracy {result['acc']:.4f}"
            )
        else:
            need = RETRAIN_EVERY - (result["db_len"] % RETRAIN_EVERY)
            if result["db_len"] < RETRAIN_EVERY:
                need = RETRAIN_EVERY - result["db_len"]
            if need > 0:
                st.caption(f"Need {need} more signature(s) before the next retrain.")
        st.rerun()

    # Bulk-load from batch tab
    st.write("")
    st.markdown("**Bulk-load from Batch Detection**")
    batch_df = st.session_state.get("batch_df")
    batch_results = st.session_state.get("batch_results")
    if batch_df is not None and batch_results is not None:
        candidates = [
            (i, batch_df.iloc[i], batch_results[i])
            for i in range(len(batch_results))
            if batch_results[i]["action"] in ("BLOCK", "VERIFY")
        ]
        st.caption(
            f"{len(candidates)} BLOCK/VERIFY flow(s) available from the last batch run."
        )
        b1, b2 = st.columns([2, 1])
        with b1:
            bulk_n = st.slider(
                "How many to promote",
                min_value=1,
                max_value=min(50, max(1, len(candidates))),
                value=min(5, max(1, len(candidates))),
                disabled=len(candidates) == 0,
            )
        with b2:
            st.write("")
            st.write("")
            do_bulk = st.button(
                "Promote to signature DB",
                use_container_width=True,
                disabled=len(candidates) == 0,
            )
        if do_bulk and candidates:
            last = None
            for i, flow_row, res in candidates[:bulk_n]:
                atype = res["attack"].split(" (")[0]
                if atype == "Unclassified":
                    atype = "PortScan"
                label = 0 if atype == "Benign" else 1
                flow = {
                    "dst_port": int(res["dport"]),
                    "byte_count": int(res["psize"]),
                    "tcp_syn": int(res["syn"]),
                    "tcp_ack": int(res["ack"]),
                    "tcp_rst": int(res["rst"]),
                    "tcp_psh": int(flow_row.get("tcp_psh", 0)),
                    "tcp_urg": int(flow_row.get("tcp_urg", 0)),
                }
                last = apply_signature_update(flow, atype, label, source="batch-promote")
            if last:
                st.session_state.last_confirm = last
            st.success(f"Promoted {min(bulk_n, len(candidates))} flow(s) into the signature database.")
            st.rerun()
    else:
        st.caption("Run Batch Detection first, then promote BLOCK/VERIFY flows here.")

    # Live signature table
    st.write("")
    st.markdown("**Signature database** — live view (persisted to `signature_db.csv`)")
    if len(st.session_state.db) > 0:
        show_cols = [
            c for c in ["confirmed_at", "attack_type", "label", "dst_port", "tcp_syn",
                        "tcp_ack", "byte_count", "source", "why"]
            if c in st.session_state.db.columns
        ]
        st.dataframe(
            st.session_state.db[show_cols].iloc[::-1],
            use_container_width=True,
            height=280,
            hide_index=True,
        )
    else:
        st.info("Database is empty. Confirm a signature above to watch the table grow.")

    # Activity / audit log
    st.write("")
    st.markdown("**Activity log** — every update, prior shift, and retrain")
    audit = load_audit_log()
    if len(audit) > 0:
        display = audit.sort_values("timestamp", ascending=False).copy()
        display = display.rename(columns={
            "timestamp": "When",
            "action": "Action",
            "object": "Object",
            "reason": "Why added",
            "prior_before": "Prior before",
            "prior_after": "Prior after",
            "prior_delta": "Δ prior",
            "retrain_triggered": "Retrain?",
            "impact": "Impact",
            "details": "Details",
        })
        preferred = [
            "When", "Action", "Object", "Why added",
            "Prior before", "Prior after", "Δ prior", "Retrain?", "Impact", "Details",
        ]
        cols_present = [c for c in preferred if c in display.columns]
        st.dataframe(display[cols_present], use_container_width=True, height=320, hide_index=True)
    else:
        st.caption("No audit events yet.")

    st.write("")
    if st.button("Reset databases"):
        reset_all_databases()
        st.rerun()
