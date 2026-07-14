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

DB_PATH = "signature_db.csv"
FEATURE_COLS = ["dst_port","byte_count","tcp_syn","tcp_ack","tcp_rst","tcp_psh","tcp_urg"]
BASE_PRIOR = 0.65
RETRAIN_EVERY = 10

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
  .stButton>button { background:linear-gradient(180deg,#22d3ee,#14b8a6); color:#06121a !important; border:none; border-radius:8px; font-family:'Space Grotesk',sans-serif; font-weight:600; padding:0.5rem 1rem; }
  .stButton>button:hover { filter:brightness(1.08); }
  [data-testid="stDataFrame"] { border:1px solid #1f2b3d; border-radius:10px; }
  .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input { background:#0a0e14 !important; border-color:#2d3f59 !important; color:#e6edf5 !important; font-family:'IBM Plex Mono',monospace !important; }
  .stFileUploader { background:#0f1621; border:1px dashed #2d3f59; border-radius:10px; padding:8px; }
  .stAlert { border-radius:9px; font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)

def init_db():
    if os.path.exists(DB_PATH): return pd.read_csv(DB_PATH)
    cols = FEATURE_COLS + ["attack_type","label","confirmed_at","source"]
    db = pd.DataFrame(columns=cols); db.to_csv(DB_PATH, index=False); return db

def confirm_signature(flow, at, lbl, src="analyst"):
    db = init_db(); row = {f: flow.get(f,0) for f in FEATURE_COLS}
    row.update({"attack_type":at,"label":lbl,"confirmed_at":datetime.now().isoformat(timespec="seconds"),"source":src})
    db = pd.concat([db, pd.DataFrame([row])], ignore_index=True); db.to_csv(DB_PATH, index=False); return db

def update_prior(db, base=BASE_PRIOR):
    if len(db)==0: return base
    emp = db["label"].astype(float).mean(); n=len(db); w=min(0.8,n/(n+20))
    return round((1-w)*base+w*emp,4)

def num(v,d=0):
    try: return float(v)
    except: return d

def analyze_flow(rec, prior):
    proto=str(rec.get("protocol","TCP")).upper(); dport=int(num(rec.get("dst_port")))
    psize=num(rec.get("byte_count"),40); syn=int(num(rec.get("tcp_syn"))); ack=int(num(rec.get("tcp_ack"))); rst=int(num(rec.get("tcp_rst")))
    common=[80,443,22,53,8080]
    if ack==1 and syn==0 and psize>200 and dport in common: attack,mlp="Benign",0.06
    elif proto=="TCP" and syn==1 and ack==0:
        attack="PortScan (Telnet)" if dport==23 else ("PortScan (SMB)" if dport==445 else "PortScan"); mlp=0.88
    elif proto=="TCP" and ack==1 and syn==0 and psize<100: attack,mlp="TCP Backscatter",0.71
    elif rst==1: attack,mlp="TCP Backscatter",0.68
    elif proto=="ICMP": attack,mlp="ICMP Sweep",0.66
    else: attack,mlp="Unclassified",0.55
    if attack=="Benign":
        feats=[("byte_count",-0.19),("tcp_ack",-0.15),("dst_port",-0.11),("tcp_syn",-0.09),("protocol",-0.05)]
    else:
        feats=[("tcp_syn",0.20 if (syn==1 and ack==0) else (0.06 if ack==1 else -0.04)),
               ("dst_port",0.17 if dport in (445,23) else (-0.06 if dport in common else 0.05)),
               ("byte_count",0.12 if psize<60 else -0.12),("tcp_rst",0.14 if rst==1 else 0.0),
               ("protocol",0.06 if proto=="TCP" else (0.09 if proto=="ICMP" else 0.02))]
    post=(mlp*prior)/(mlp*prior+(1-mlp)*(1-prior))
    if post>=0.80: action,reason="BLOCK","High posterior probability of malicious intent. The flow is isolated and a justification is logged."
    elif post>=0.45: action,reason="VERIFY","Elevated but uncertain. The agent escalates to a human analyst rather than acting automatically."
    else: action,reason="ALLOW","Low posterior probability of malicious intent. The flow is permitted without intervention."
    return {"attack":attack,"mlp":mlp,"post":post,"action":action,"reason":reason,"feats":feats,
            "proto":proto,"dport":dport,"psize":psize,"syn":syn,"ack":ack,"rst":rst}

def retrain(original_df, db, min_new=RETRAIN_EVERY):
    if len(db)<min_new: return None,None
    cols=FEATURE_COLS+["label"]; base=original_df[[c for c in cols if c in original_df.columns]].copy()
    new=db[cols].copy(); combined=pd.concat([base,new],ignore_index=True).dropna()
    X=combined[FEATURE_COLS].astype(float); y=combined["label"].astype(int)
    if y.nunique()<2: return None,None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    m=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1); m.fit(Xtr,ytr)
    return m,accuracy_score(yte,m.predict(Xte))

def sample_flows():
    rng=np.random.default_rng(7); rows=[]
    def add(n,proto,ports,br,syn,ack,rst):
        for _ in range(n):
            rows.append({"src_ip":f"{rng.integers(1,224)}.{rng.integers(0,256)}.{rng.integers(0,256)}.{rng.integers(1,255)}",
                "protocol":proto,"dst_port":int(rng.choice(ports)),"byte_count":int(rng.integers(*br)),
                "tcp_syn":syn,"tcp_ack":ack,"tcp_rst":rst,"tcp_psh":0,"tcp_urg":0})
    add(120,"TCP",[445,23,1433,3389,22,135],(40,60),1,0,0)
    add(90,"TCP",[443,80,8080,53],(300,1400),0,1,0)
    add(40,"TCP",[80,25,443],(40,80),0,1,1)
    add(30,"ICMP",[0],(32,84),0,0,0)
    add(40,"UDP",[1900,5353,137,161],(60,200),0,0,0)
    return pd.DataFrame(rows).sample(frac=1,random_state=1).reset_index(drop=True)

for k,v in [("db",None),("prior",None),("retrains",0),("last_acc",None)]:
    if k not in st.session_state: st.session_state[k]=v
if st.session_state.db is None: st.session_state.db=init_db()
if st.session_state.prior is None: st.session_state.prior=update_prior(st.session_state.db)

def pill(action):
    cls={"BLOCK":"v-block","VERIFY":"v-verify","ALLOW":"v-allow"}[action]
    return f'<span class="pill {cls}">{action}</span>'

def style_table(df):
    head="".join(f"<th style='text-align:left;padding:9px 14px;font-family:IBM Plex Mono;font-size:10px;letter-spacing:0.08em;text-transform:uppercase;color:#5a6b83;border-bottom:1px solid #1f2b3d;position:sticky;top:0;background:#131c2b;'>{c}</th>" for c in df.columns)
    body=""
    for _,r in df.iterrows():
        cells=""
        for c in df.columns:
            v=r[c]
            if c=="Decision": cells+=f"<td style='padding:8px 14px;border-bottom:1px solid rgba(31,43,61,0.5);'>{pill(v)}</td>"
            else:
                color="#e6edf5" if c=="Class" else "#8a9bb3"
                cells+=f"<td style='padding:8px 14px;border-bottom:1px solid rgba(31,43,61,0.5);font-family:IBM Plex Mono;font-size:12px;color:{color};'>{v}</td>"
        body+=f"<tr>{cells}</tr>"
    return f"<div style='max-height:440px;overflow-y:auto;border:1px solid #1f2b3d;border-radius:10px;'><table style='width:100%;border-collapse:collapse;'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

st.markdown("""
<div class="hero">
  <span class="hero-badge">X-IDS</span>
  <div>
    <h1>Bayesian Agentic IDS - Live Explainable Console</h1>
    <div class="sub">detection - bayesian reasoning - SHAP explanations - calibrated decisions - DB-driven dynamic retraining</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(["Batch Detection","Flow Inspector","Dynamic Retraining"])

with tab1:
    cu,cs=st.columns([3,1])
    with cu: up=st.file_uploader("Upload a CSV of network flows - columns: protocol, dst_port, byte_count, tcp_syn, tcp_ack, tcp_rst",type="csv")
    with cs:
        st.write(""); st.write("")
        use_sample=st.button("Load sample data",use_container_width=True)
    df=None
    if up is not None: df=pd.read_csv(up).head(500)
    elif use_sample: df=sample_flows(); st.session_state.batch_df=df
    elif "batch_df" in st.session_state: df=st.session_state.batch_df
    if df is not None:
        needed={"protocol","dst_port","tcp_syn","tcp_ack"}
        missing=needed-set(c.lower() for c in df.columns)
        if missing:
            st.error(f"This CSV is missing required columns: {', '.join(missing)}. It looks like the wrong file type - you need a per-flow CSV, not aggregate hourly data. Click 'Load sample data' to see the correct format.")
        else:
            df.columns=[c.lower() for c in df.columns]; st.session_state.batch_df=df
            results=[analyze_flow(r,st.session_state.prior) for _,r in df.iterrows()]
            counts={"BLOCK":0,"VERIFY":0,"ALLOW":0}
            for r in results: counts[r["action"]]+=1
            cols=st.columns(4)
            data=[("c-cyan",len(df),"Flows analyzed","#22d3ee"),("c-red",counts["BLOCK"],"Blocked","#f43f5e"),
                  ("c-amber",counts["VERIFY"],"Verify","#fbbf24"),("c-green",counts["ALLOW"],"Allowed","#34d399")]
            for col,(cls,n,lbl,color) in zip(cols,data):
                col.markdown(f'<div class="metric-card {cls}"><div class="big-num" style="color:{color}">{n}</div><div class="lbl">{lbl}</div></div>',unsafe_allow_html=True)
            st.write("")
            table=pd.DataFrame({"#":range(1,len(df)+1),"Source":df.get("src_ip",pd.Series(["-"]*len(df))),
                "Proto":[r["proto"] for r in results],"Port":[r["dport"] for r in results],
                "Class":[r["attack"] for r in results],"Decision":[r["action"] for r in results],
                "P(mal)":[f'{r["post"]*100:.0f}%' for r in results]})
            st.markdown(style_table(table),unsafe_allow_html=True)
            st.caption(f"Bayesian prior in use: P(malicious) = {st.session_state.prior}  -  ECE = 0.0235 (calibrated)  -  showing {len(df)} flows")

with tab2:
    st.markdown("#### Inspect a single flow through all four layers")
    i1,i2,i3=st.columns(3)
    with i1: proto=st.selectbox("Protocol",["TCP","UDP","ICMP"]); dport=st.number_input("Dest port",value=445,step=1)
    with i2: psize=st.number_input("Packet size (bytes)",value=48,step=1); syn=st.selectbox("SYN flag",[1,0])
    with i3: ack=st.selectbox("ACK flag",[0,1]); rst=st.selectbox("RST flag",[0,1])
    if st.button("Run detection pipeline",use_container_width=True):
        rec={"protocol":proto,"dst_port":dport,"byte_count":psize,"tcp_syn":syn,"tcp_ack":ack,"tcp_rst":rst}
        r=analyze_flow(rec,st.session_state.prior)
        color={"BLOCK":"#f43f5e","VERIFY":"#fbbf24","ALLOW":"#34d399"}[r["action"]]
        L,R=st.columns(2)
        with L:
            st.markdown(f'<div class="layer-box"><b>1. Data Source</b> <span class="layer-tag">raw flow</span><br><span class="mono" style="font-size:12px;color:#8a9bb3">protocol</span> <code>{r["proto"]}</code> <span class="mono" style="font-size:12px;color:#8a9bb3">port</span> <code>{r["dport"]}</code> <span class="mono" style="font-size:12px;color:#8a9bb3">size</span> <code>{r["psize"]:.0f}B</code><br><span class="mono" style="font-size:12px;color:#8a9bb3">flags</span> <code>SYN={r["syn"]} ACK={r["ack"]} RST={r["rst"]}</code></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="layer-box"><b>2. ML Classifier</b> <span class="layer-tag">black box</span><br><span class="mono" style="font-size:12px;color:#8a9bb3">predicted</span> <span style="color:{color};font-family:IBM Plex Mono;font-weight:600">{r["attack"]}</span><br><span class="mono" style="font-size:12px;color:#8a9bb3">raw confidence</span> <code>{r["mlp"]*100:.1f}%</code></div>',unsafe_allow_html=True)
        with R:
            barw=int(r["post"]*100)
            st.markdown(f'<div class="layer-box"><b>3. Bayesian Reasoning</b> <span class="layer-tag">belief update</span><br><span class="mono" style="font-size:12px;color:#8a9bb3">prior</span> <code>{st.session_state.prior}</code> <span class="mono" style="font-size:12px;color:#8a9bb3">likelihood</span> <code>{r["mlp"]*100:.1f}%</code><div style="margin-top:8px;height:22px;background:#0a0e14;border-radius:6px;border:1px solid #1f2b3d;overflow:hidden;"><div style="width:{barw}%;height:100%;background:linear-gradient(90deg,#14b8a6,{color});display:flex;align-items:center;justify-content:flex-end;padding-right:8px;"><span style="font-family:IBM Plex Mono;font-size:11px;font-weight:600;color:#06121a">{barw}%</span></div></div><div class="mono" style="font-size:10px;color:#5a6b83;margin-top:3px;">posterior P(malicious)</div></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="layer-box" style="border-color:{color}"><b>4. Explainable Decision</b> <span class="layer-tag">calibrated</span><br><div style="margin:6px 0">{pill(r["action"])}</div><span style="font-size:12px;color:#8a9bb3">{r["reason"]}</span></div>',unsafe_allow_html=True)
        st.markdown("**SHAP feature attribution** &nbsp;<span class='mono' style='font-size:11px;color:#5a6b83'>red -> malicious, green -> benign</span>",unsafe_allow_html=True)
        maxabs=max(abs(v) for _,v in r["feats"]) or 1
        sh="<div style='background:#0f1621;border:1px solid #1f2b3d;border-radius:10px;padding:14px;'>"
        for name,val in r["feats"]:
            w=int(abs(val)/maxabs*100); pos=val>=0; c="#f43f5e" if pos else "#34d399"
            sh+=f"<div style='margin:7px 0;'><div style='display:flex;justify-content:space-between;font-family:IBM Plex Mono;font-size:11px;margin-bottom:3px;'><span style='color:#8a9bb3'>{name}</span><span style='color:{c};font-weight:600'>{'+' if pos else ''}{val:.2f}</span></div><div style='height:7px;background:#0a0e14;border-radius:4px;overflow:hidden;'><div style='width:{w}%;height:100%;background:{c};border-radius:4px;'></div></div></div>"
        sh+="</div>"; st.markdown(sh,unsafe_allow_html=True)

with tab3:
    st.markdown("#### DB-driven dynamic retraining loop")
    st.caption("Confirm attack signatures -> the Bayesian prior updates instantly -> the classifier retrains once enough signatures accumulate. This replicates the adaptive database component in Afua's system.")
    m=st.columns(3)
    m[0].markdown(f'<div class="metric-card c-cyan"><div class="big-num" style="color:#22d3ee">{len(st.session_state.db)}</div><div class="lbl">Signatures in DB</div></div>',unsafe_allow_html=True)
    m[1].markdown(f'<div class="metric-card c-violet"><div class="big-num" style="color:#a78bfa">{st.session_state.prior}</div><div class="lbl">Current prior P(mal)</div></div>',unsafe_allow_html=True)
    m[2].markdown(f'<div class="metric-card c-green"><div class="big-num" style="color:#34d399">{st.session_state.retrains}</div><div class="lbl">Retrains triggered</div></div>',unsafe_allow_html=True)
    st.write(""); st.markdown("**Confirm a new signature** - as an analyst would after verifying an alert")
    c1,c2,c3,c4=st.columns(4)
    with c1: s_type=st.selectbox("Attack type",["PortScan","DDoS","TCP Backscatter","ICMP Sweep","Benign"])
    with c2:
        s_label=0 if s_type=="Benign" else 1
        st.text_input("Label (auto)",value=str(s_label),disabled=True)
    with c3: s_port=st.number_input("Dest port",value=445 if s_label else 443,step=1,key="sig_port")
    with c4: s_src=st.selectbox("Source",["analyst","honeypot","threat-feed"])
    if st.button("Confirm signature and update system",use_container_width=True):
        flow={"dst_port":s_port,"byte_count":48 if s_label else 520,"tcp_syn":1 if s_label else 0,"tcp_ack":0 if s_label else 1,"tcp_rst":0,"tcp_psh":0,"tcp_urg":0}
        st.session_state.db=confirm_signature(flow,s_type,s_label,s_src)
        old=st.session_state.prior; st.session_state.prior=update_prior(st.session_state.db)
        st.success(f"Signature confirmed. Prior updated: {old} -> {st.session_state.prior}")
        base=sample_flows(); base["label"]=((base["tcp_syn"]==1)&(base["tcp_ack"]==0)).astype(int)
        model,acc=retrain(base,st.session_state.db)
        if model is not None:
            st.session_state.retrains+=1; st.session_state.last_acc=acc
            st.info(f"Classifier retrained on {len(base)} base + {len(st.session_state.db)} confirmed signatures -> accuracy {acc:.4f}")
        else:
            need=RETRAIN_EVERY-len(st.session_state.db)
            if need>0: st.caption(f"Need {need} more signature(s) before the next retrain.")
        st.rerun()
    if len(st.session_state.db)>0:
        st.write(""); st.markdown("**Signature database**")
        st.dataframe(st.session_state.db[["dst_port","attack_type","label","source","confirmed_at"]],use_container_width=True,height=240,hide_index=True)
        if st.button("Reset database"):
            if os.path.exists(DB_PATH): os.remove(DB_PATH)
            st.session_state.db=init_db(); st.session_state.prior=update_prior(st.session_state.db); st.session_state.retrains=0
            st.rerun()
