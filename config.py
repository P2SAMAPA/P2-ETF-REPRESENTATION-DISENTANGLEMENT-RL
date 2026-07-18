import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-representation-disentanglement-rl-results"

WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "SOXX", "SMH", "XLB", 
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "SOXX", "SMH", "XLB",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# All available macro variables
MACRO_VARS = [
    "VIX", "DXY", "T10Y2Y", "TBILL_3M",
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS7",
    "DGS10", "DGS20", "DGS30"
]

# Disentanglement RL parameters
LATENT_DIM = 8               # dimension of disentangled representation
EXOGENOUS_DIM = 4            # dimension for exogenous (macro) factors
ENDOGENOUS_DIM = 4           # dimension for endogenous (market) factors
BETA = 0.5                   # disentanglement strength (β-VAE style)
SEQ_LEN = 10
BATCH_SIZE = 16
EPOCHS = 30
TOP_N = 3
