import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random

# Vocabulary setup
chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()_+-=[]}{\\|;:'\",<.>/? ")
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
vocab_size = len(chars)

# Hyperparameters
latent_dim = 64
embedding_dim = 32
hidden_dim = 128
max_length = 32  # Fixed password length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VAE Model
class PasswordFeatureVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, feature_dim, max_length):
        super(PasswordFeatureVAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim + feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + feature_dim, latent_dim)
        self.decoder_rnn = nn.LSTM(latent_dim + feature_dim, hidden_dim, batch_first=True)  # Include feature dimension
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def encode(self, password_seq, features):
        embedded = self.embedding(password_seq)
        _, (hidden, _) = self.encoder_rnn(embedded)
        hidden = hidden[-1]
        combined = torch.cat([hidden, features], dim=1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, features):
        z = z.unsqueeze(1).repeat(1, self.max_length, 1)  # Repeat z to match sequence length
        combined_input = torch.cat([z, features.unsqueeze(1).repeat(1, self.max_length, 1)], dim=-1)  # Concatenate z and features
        outputs, _ = self.decoder_rnn(combined_input)
        logits = self.fc_out(outputs)
        return logits

    def forward(self, password_seq, features):
        mu, logvar = self.encode(password_seq, features)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, features)  # Pass features along with z to the decoder
        return logits, mu, logvar
    
# Load normalized features
normalized_features = np.load("bin/VAE.npy")
feature_dim = normalized_features.shape[1]

# Reinitialize model and optimizer
loaded_model = PasswordFeatureVAE(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                  latent_dim=latent_dim, feature_dim=feature_dim, max_length=max_length).to(device)
loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)

# Load state dictionaries
checkpoint = torch.load("bin/VAE.pth", weights_only=True, map_location=device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Set the model to evaluation mode if using for inference
loaded_model.eval()

# Function to generate passwords
def generate_passwords(model=loaded_model, num_passwords=1):
    model.eval()
    with torch.no_grad():
        # Generate latent vectors
        x = random.randint(1, 2000000)
        y = random.randint(1, 68)
        z = x+y
        features = torch.tensor(normalized_features[x:z], dtype=torch.float32).to(device)
        z = torch.randn(num_passwords, latent_dim).to(device)
        
        # Combine latent vector with features for full input
        z_combined = torch.cat([z, features[:num_passwords].to(device)], dim=1)

        # Decode only the latent vector with features
        logits = model.decode(z, features[:num_passwords].to(device))  # Pass features along with latent vectors
        passwords = torch.argmax(logits, dim=-1)
        
        # Convert token indices to characters
        return [''.join(idx_to_char[idx.item()] for idx in password) for password in passwords][0]