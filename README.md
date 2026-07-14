---
Title: Bayesian Agentic IDS
---

# Bayesian Agentic IDS — Live Explainable Detection Console

An interactive replication of a Bayesian Agentic Intrusion Detection System with
explainable AI and a database-driven dynamic retraining loop.

## What it does

The system layers probabilistic reasoning on top of a machine learning classifier
to turn opaque alerts into calibrated, explainable decisions. Traffic flows through
four layers:

1. **Data Source** — raw network flow features
2. **ML Classifier** — fast pattern recognition (the "black box")
3. **Bayesian Reasoning** — belief updating that combines a prior with the classifier's evidence via Bayes' rule
4. **Explainable Decision** — a calibrated BLOCK / VERIFY / ALLOW action with SHAP feature attribution

## Three tabs

- **Batch Detection** — upload a CSV of flows (or load a sample) and see every flow triaged
- **Flow Inspector** — run a single flow through all four layers and view its SHAP explanation
- **Dynamic Retraining** — confirm attack signatures into a database; the Bayesian prior updates instantly and the classifier retrains as signatures accumulate

## CSV format

Expected columns: `protocol`, `dst_port`, `byte_count`, `tcp_syn`, `tcp_ack`, `tcp_rst`
(optional: `src_ip`, `tcp_psh`, `tcp_urg`)

Built as a research demonstration. The classifier and Bayesian components are
faithful working models of the architecture.
