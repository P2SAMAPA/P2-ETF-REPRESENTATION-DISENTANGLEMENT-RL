import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DisentanglementEncoder(nn.Module):
    """
    Encoder that disentangles exogenous and endogenous factors.
    """
    def __init__(self, input_size, latent_dim=8, exo_dim=4, endo_dim=4, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.exo_dim = exo_dim
        self.endo_dim = endo_dim
        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_size, 64, 2, batch_first=True)
        # Disentangled encoders
        self.exo_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, exo_dim)
        )
        self.endo_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, endo_dim)
        )
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        h = lstm_out[:, -1, :]
        # Disentangled factors
        exo = self.exo_encoder(h)
        endo = self.endo_encoder(h)
        # Combine
        z = torch.cat([exo, endo], dim=1)
        # Reconstruction
        recon = self.decoder(z)
        return z, exo, endo, recon

    def disentangle(self, x):
        """Just the disentangled factors."""
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        exo = self.exo_encoder(h)
        endo = self.endo_encoder(h)
        return exo, endo

class DisentanglementRL:
    """
    Disentanglement RL agent: learns to separate exogenous and endogenous factors.
    """
    def __init__(self, input_size, latent_dim=8, exo_dim=4, endo_dim=4, seq_len=10, beta=0.5):
        self.encoder = DisentanglementEncoder(input_size, latent_dim, exo_dim, endo_dim, seq_len)
        self.beta = beta
        self.seq_len = seq_len
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)

    def train_step(self, X, y):
        """Train on a batch of sequences."""
        self.encoder.train()
        z, exo, endo, recon = self.encoder(X)
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, X[:, -1, :])  # reconstruct last time step
        # Disentanglement loss (β-VAE style)
        # Encourage exo and endo to be independent
        exo_std = torch.std(exo, dim=0).mean()
        endo_std = torch.std(endo, dim=0).mean()
        # Encourage factors to be orthogonal (disentangled)
        # Compute cross-correlation between exo and endo
        exo_norm = (exo - exo.mean(dim=0)) / (exo.std(dim=0) + 1e-8)
        endo_norm = (endo - endo.mean(dim=0)) / (endo.std(dim=0) + 1e-8)
        corr = (exo_norm.T @ endo_norm) / exo_norm.shape[0]
        cross_corr_loss = torch.mean(corr**2)
        # Total loss
        loss = recon_loss + self.beta * cross_corr_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def disentanglement_score(self, X):
        """
        Compute disentanglement quality score: how well exo and endo are separated.
        Higher score = better disentanglement.
        """
        self.encoder.eval()
        with torch.no_grad():
            exo, endo = self.encoder.disentangle(X)
            # Normalise
            exo_norm = (exo - exo.mean(dim=0)) / (exo.std(dim=0) + 1e-8)
            endo_norm = (endo - endo.mean(dim=0)) / (endo.std(dim=0) + 1e-8)
            # Compute correlation between factors
            corr = (exo_norm.T @ endo_norm) / exo_norm.shape[0]
            # Disentanglement score = 1 - abs(correlation) (higher = better)
            score = 1.0 - torch.abs(corr).mean().item()
        return score

def prepare_data(returns, macro_df, seq_len=10):
    """Prepare sequences for training."""
    if len(returns) < seq_len + 1:
        return None, None
    common_idx = returns.index.intersection(macro_df.index)
    ret_aligned = returns.loc[common_idx]
    macro_aligned = macro_df.loc[common_idx]
    X, y = [], []
    for i in range(seq_len, len(ret_aligned)):
        ret_seq = ret_aligned.iloc[i-seq_len:i].values.reshape(-1, 1)
        macro_seq = macro_aligned.iloc[i-seq_len:i].values
        seq_features = np.concatenate([ret_seq, macro_seq], axis=1)
        X.append(seq_features)
        y.append(ret_aligned.iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

def disentanglement_rl_score(returns, macro_df, latent_dim=8, exo_dim=4, endo_dim=4, seq_len=10, epochs=30, batch_size=16, beta=0.5):
    """
    Train disentanglement RL and return disentanglement quality score.
    """
    X, y = prepare_data(returns, macro_df, seq_len)
    if X is None or len(X) < batch_size:
        return 0.0
    input_size = X.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DisentanglementRL(input_size, latent_dim, exo_dim, endo_dim, seq_len, beta)
    agent.encoder.to(device)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            loss = agent.train_step(X_batch, y_batch)
            epoch_loss += loss
    # Compute disentanglement score on the last sequence
    last_seq = np.concatenate([
        returns.iloc[-seq_len:].values.reshape(-1, 1),
        macro_df.iloc[-seq_len:].values
    ], axis=1)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    score = agent.disentanglement_score(last_seq_tensor)
    return float(score)
