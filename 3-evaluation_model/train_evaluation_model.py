"""Train the attention-based MLP evaluation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, TrainingConfig
from src.get_training_data import load_training_data


class Attention(nn.Module):
    def __init__(self, k_dim: int, v_dim: int, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, k_dim)
        self.key = nn.Linear(d_model, k_dim)
        self.value = nn.Linear(d_model, v_dim)
        self.scale = k_dim**0.5
        self.embedding = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        attention_weights_batch_mean = attention_weights.mean(dim=0)
        return attention_output, attention_weights_batch_mean


class MLPWithAttention(nn.Module):
    def __init__(self, v_dim: int, k_dim: int, hidden_dims: tuple[int, int, int, int], d_model: int) -> None:
        super().__init__()
        self.attention = Attention(k_dim, v_dim, d_model)
        self.linear = nn.Linear(v_dim, 1)
        self.fc1 = nn.Linear(len(FEATURE_COLUMNS), hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.fc5 = nn.Linear(hidden_dims[3], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_weights_batch_mean = self.attention(x)
        x = self.linear(attention_output)
        x = x.squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x, attention_weights_batch_mean


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_training_data(data: pd.DataFrame) -> None:
    required_columns = ["index"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")
    unexpected_features = [
        column
        for column in data.columns
        if column not in {"index", TARGET_COLUMN, *FEATURE_COLUMNS}
    ]
    if unexpected_features:
        raise ValueError(
            "Objective metrics contain columns that are not represented by the model: "
            f"{unexpected_features}"
        )


def build_stratified_dataset(data: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    data = data.copy()
    bins = np.arange(0, 1.1, 0.1)
    labels = [f"{i / 10}-{(i + 1) / 10}" for i in range(10)]
    data["score_bin"] = pd.cut(data[TARGET_COLUMN], bins=bins, labels=labels, include_lowest=True)

    stratified_samples = []
    for bin_label in labels:
        bin_data = data[data["score_bin"] == bin_label]
        print(f"score_bin={bin_label}, samples={len(bin_data)}")
        if len(bin_data) < config.samples_per_score_bin:
            stratified_samples.append(bin_data)
        else:
            stratified_samples.append(
                bin_data.sample(n=config.samples_per_score_bin, random_state=config.sample_random_state)
            )
    stratified_data = pd.concat(stratified_samples)
    return stratified_data.sample(frac=1, random_state=config.shuffle_random_state).reset_index(drop=True)


def prepare_tensors(config: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    data = load_training_data(config.metrics_path, config.scores_path)
    validate_training_data(data)
    stratified_data = build_stratified_dataset(data, config)

    x_values = stratified_data[FEATURE_COLUMNS].values
    y_values = stratified_data[TARGET_COLUMN].values
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=config.test_size, random_state=config.seed
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, y_test, x_test_scaled


def train_model(
    model: MLPWithAttention,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    losses = []
    last_epoch_attention_weights = []

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        batch_attention_weights = []

        for start in range(0, len(x_train), config.batch_size):
            x_batch = x_train[start : start + config.batch_size].unsqueeze(-1)
            y_batch = y_train[start : start + config.batch_size]

            output, attention_weights_batch_mean = model(x_batch)
            batch_attention_weights.append(attention_weights_batch_mean.detach().numpy())
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            y_pred, _ = model(x_test.unsqueeze(-1))
            val_loss = criterion(y_pred, y_test)

        train_loss = epoch_loss / max(1, len(x_train) // config.batch_size)
        val_loss_value = val_loss.item()
        losses.append({"Epoch": epoch + 1, "Train Loss": train_loss, "Validation Loss": val_loss_value})
        last_epoch_attention_weights = batch_attention_weights
        print(
            f"Epoch [{epoch + 1}/{config.epochs}], "
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss_value:.4f}"
        )

    return pd.DataFrame(losses), last_epoch_attention_weights


def save_training_outputs(
    model: MLPWithAttention,
    x_test: torch.Tensor,
    y_test_tensor: torch.Tensor,
    y_test: np.ndarray,
    losses: pd.DataFrame,
    attention_weights: list[np.ndarray],
    config: TrainingConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        y_pred, _ = model(x_test.unsqueeze(-1))
        test_loss = criterion(y_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

    torch.save(model.state_dict(), config.output_dir / "evaModel_lightweight.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(losses["Epoch"], losses["Train Loss"], label="Train Loss")
    plt.plot(losses["Epoch"], losses["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.output_dir / "loss.png", dpi=300)
    plt.close()

    mean_attention_weights = np.mean(attention_weights, axis=0)
    relation_matrix = pd.DataFrame(mean_attention_weights, columns=FEATURE_COLUMNS, index=FEATURE_COLUMNS)

    plt.figure(figsize=(10, 6))
    sns.heatmap(relation_matrix, annot=True, cmap="coolwarm")
    plt.title("Attention-Based Feature Relationship Matrix")
    plt.tight_layout()
    plt.savefig(config.output_dir / "attention_matrix.png", dpi=300)
    plt.close()


def run(config: TrainingConfig) -> None:
    set_seed(config.seed)
    x_train, y_train, x_test, y_test_tensor, y_test, _ = prepare_tensors(config)
    model = MLPWithAttention(config.v_dim, config.k_dim, config.hidden_dims, config.d_model)
    losses, attention_weights = train_model(model, x_train, y_train, x_test, y_test_tensor, config)
    save_training_outputs(model, x_test, y_test_tensor, y_test, losses, attention_weights, config)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the attention-based MLP evaluation model.")
    parser.add_argument("--metrics-path", type=Path, default=TrainingConfig.metrics_path)
    parser.add_argument("--scores-path", type=Path, default=TrainingConfig.scores_path)
    parser.add_argument("--output-dir", type=Path, default=TrainingConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    args = parser.parse_args()
    return TrainingConfig(
        metrics_path=args.metrics_path,
        scores_path=args.scores_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    run(parse_args())
