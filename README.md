# Representation Disentanglement RL for ETFs

Learns to disentangle exogenous (macro-driven) vs endogenous (market-driven) sources of variation in ETF returns. The disentangled representation feeds downstream predictors – a form of causal representation learning tailored to time series. The per‑ETF score is the disentanglement quality.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- LSTM encoder for sequence processing
- Disentangled exogenous and endogenous factor encoders
- β-VAE style disentanglement loss
- Score = disentanglement quality (higher = better factor separation)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-representation-disentanglement-rl-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to neural net training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High disentanglement quality → factors are well separated → better causal representation.
- Low disentanglement quality → factors are entangled.

## Requirements

See `requirements.txt`.
