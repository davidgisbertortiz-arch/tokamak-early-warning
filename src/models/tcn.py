"""
Temporal Convolutional Network (TCN) for tokamak early warning.

TCN uses dilated causal convolutions to capture multi-scale temporal patterns
while maintaining strict causality (no future information leakage).

Key advantages for early warning:
1. Multi-scale patterns: Dilated convolutions capture both short and long-term dynamics
2. Parallelizable: Unlike LSTM, all positions computed simultaneously (CPU-friendly)
3. Fixed receptive field: Clear understanding of temporal context used
4. Causal: Only uses past information, suitable for real-time prediction

Architecture:
- Input: (batch, seq_len, n_features)
- Several dilated causal conv blocks with residual connections
- Global pooling + classifier head
- Output: (batch, 1) probability of event

Reference:
- Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and 
  Recurrent Networks for Sequence Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class TCNConfig:
    """Configuration for TCN model."""
    # Architecture
    n_features: int = 80  # Number of input features (set after feature engineering)
    seq_len: int = 20  # Sequence length (window size)
    hidden_channels: int = 32  # Channels in conv layers
    n_blocks: int = 3  # Number of TCN blocks
    kernel_size: int = 3  # Convolution kernel size
    dropout: float = 0.2  # Dropout rate
    
    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 30
    patience: int = 5  # Early stopping patience
    
    # Class imbalance
    pos_weight: float = 10.0  # Weight for positive class in loss
    
    # Reproducibility
    random_state: int = 42


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.
    
    Ensures output at time t only depends on inputs at time <= t.
    Achieved by left-padding the input.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        # Left-pad to ensure causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Single TCN block with dilated causal convolutions and residual connection.
    
    Structure:
    - CausalConv -> BatchNorm -> ReLU -> Dropout
    - CausalConv -> BatchNorm -> ReLU -> Dropout
    - Residual connection (with 1x1 conv if channels differ)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = self.residual(x)
        
        return F.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for binary classification.
    
    Uses exponentially increasing dilation to achieve large receptive field
    efficiently: dilation = 2^i for block i.
    
    Receptive field = 1 + n_blocks * (kernel_size - 1) * (2^n_blocks - 1)
    """
    
    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        
        # TCN blocks with exponentially increasing dilation
        blocks = []
        in_channels = config.n_features
        
        for i in range(config.n_blocks):
            dilation = 2 ** i
            blocks.append(TCNBlock(
                in_channels=in_channels if i == 0 else config.hidden_channels,
                out_channels=config.hidden_channels,
                kernel_size=config.kernel_size,
                dilation=dilation,
                dropout=config.dropout
            ))
        
        self.tcn_blocks = nn.Sequential(*blocks)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
            
        Returns:
            Logits of shape (batch, 1)
        """
        # Transpose to (batch, n_features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        x = self.tcn_blocks(x)
        
        # Use last time step output (most recent, for prediction)
        x = x[:, :, -1]  # (batch, hidden_channels)
        
        # Classify
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    discharge_col: str = "discharge_ID"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windowed sequences for TCN training.
    
    For each time point t, creates a window [t-seq_len+1, ..., t] and
    predicts label at time t.
    
    CRITICAL: Windows do not cross discharge boundaries to prevent leakage.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        seq_len: Sequence length (window size)
        discharge_col: Discharge ID column name
        
    Returns:
        Tuple of (X_sequences, y_labels, discharge_ids)
        X_sequences: (n_samples, seq_len, n_features)
        y_labels: (n_samples,)
        discharge_ids: (n_samples,) for tracking
    """
    X_list = []
    y_list = []
    discharge_list = []
    
    for discharge_id in df[discharge_col].unique():
        discharge_df = df[df[discharge_col] == discharge_id].sort_values("time")
        
        features = discharge_df[feature_cols].values
        labels = discharge_df[target_col].values
        
        n_samples = len(discharge_df)
        
        # Create sequences
        for i in range(seq_len - 1, n_samples):
            # Window: [i - seq_len + 1, ..., i]
            X_seq = features[i - seq_len + 1:i + 1]  # (seq_len, n_features)
            y_label = labels[i]  # Label at current time
            
            X_list.append(X_seq)
            y_list.append(y_label)
            discharge_list.append(discharge_id)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    discharges = np.array(discharge_list)
    
    return X, y, discharges


class TCNClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for TCN.
    
    Provides fit(), predict(), predict_proba() interface for integration
    with existing evaluation code.
    """
    
    def __init__(
        self,
        config: TCNConfig = None,
        device: str = "cpu",
        verbose: bool = True
    ):
        self.config = config if config is not None else TCNConfig()
        self.device = device
        self.verbose = verbose
        self.model: Optional[TCN] = None
        self.feature_cols: Optional[List[str]] = None
        self.classes_ = np.array([0, 1])
        self._is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> "TCNClassifier":
        """
        Train TCN model.
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            y: Training labels (n_samples,)
            X_val: Validation sequences (optional, for early stopping)
            y_val: Validation labels (optional)
            
        Returns:
            self
        """
        # Set random seeds
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        # Update config with actual dimensions
        self.config.n_features = X.shape[2]
        self.config.seq_len = X.shape[1]
        
        # Create model
        self.model = TCN(self.config).to(self.device)
        
        # Loss function with class weighting
        pos_weight = torch.tensor([self.config.pos_weight]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        # Note: verbose parameter removed in PyTorch 2.9+
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Convert to tensors
        X_train = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        if X_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        n_batches = (len(X_train) + self.config.batch_size - 1) // self.config.batch_size
        
        for epoch in range(self.config.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffle training data
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(X_train))
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= n_batches
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                
                scheduler.step(val_loss)
                
                if self.verbose:
                    print(f"  Epoch {epoch+1}/{self.config.n_epochs}: "
                          f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        if self.verbose:
                            print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                if self.verbose:
                    print(f"  Epoch {epoch+1}/{self.config.n_epochs}: train_loss={epoch_loss:.4f}")
        
        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self._is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            
        Returns:
            Probabilities (n_samples, 2) for [P(y=0), P(y=1)]
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Return (n_samples, 2) array for sklearn compatibility
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            threshold: Classification threshold
            
        Returns:
            Predictions (n_samples,)
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_cols': self.feature_cols,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TCNClassifier":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device)
        
        classifier = cls(config=checkpoint['config'], device=device)
        classifier.model = TCN(checkpoint['config']).to(device)
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.feature_cols = checkpoint.get('feature_cols')
        classifier._is_fitted = True
        
        return classifier


def train_tcn_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "density_limit_phase",
    config: TCNConfig = None,
    verbose: bool = True
) -> Tuple[TCNClassifier, Dict[str, Any]]:
    """
    Train TCN model end-to-end.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        config: TCN configuration
        verbose: Print progress
        
    Returns:
        Tuple of (trained model, training info dict)
    """
    if config is None:
        config = TCNConfig()
    
    if verbose:
        print("Creating sequences...")
    
    # Create sequences
    X_train, y_train, _ = create_sequences(
        train_df, feature_cols, target_col, config.seq_len
    )
    X_val, y_val, _ = create_sequences(
        val_df, feature_cols, target_col, config.seq_len
    )
    
    if verbose:
        print(f"  Train: {X_train.shape[0]} sequences")
        print(f"  Val:   {X_val.shape[0]} sequences")
        print(f"  Sequence shape: {X_train.shape[1:]} (seq_len, n_features)")
        print(f"  Positive rate - Train: {y_train.mean()*100:.2f}%, Val: {y_val.mean()*100:.2f}%")
    
    # Train model
    if verbose:
        print("\nTraining TCN...")
    
    classifier = TCNClassifier(config=config, verbose=verbose)
    classifier.feature_cols = feature_cols
    classifier.fit(X_train, y_train, X_val, y_val)
    
    # Compute training info
    info = {
        "n_train_sequences": len(X_train),
        "n_val_sequences": len(X_val),
        "n_features": X_train.shape[2],
        "seq_len": X_train.shape[1],
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
    }
    
    return classifier, info
