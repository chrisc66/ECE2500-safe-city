import logging
import math
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.init as init
from sklearn.metrics import mean_absolute_error, mean_squared_error


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module
    """

    def __init__(self, embed_dim, max_len=1):
        super(PositionalEncoding, self).__init__()
        self.logger = logging.getLogger(__name__)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return x


class ST_TEM(nn.Module):
    def __init__(self, embedding_module, embed_dim, num_heads, num_layers, max_len=1):
        super(ST_TEM, self).__init__()
        self.logger = logging.getLogger(__name__)

        # Embedding module (CombinedEmbedding) and positional encoding
        self.embedding_module = embedding_module
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Prediction head
        self.fc_out = nn.Linear(in_features=embed_dim, out_features=1)  # Output: predicting the number of events

    def forward(
        self,
        neighborhood_ids,
        time_features,
        building_type_ids,
        building_counts,
        population,
        event_type_ids,
        equipment_ids,
    ):
        # Generate combined embeddings from the embedding module
        x = self.embedding_module(
            neighborhood_ids,
            time_features,
            building_type_ids,
            building_counts,
            population,
            event_type_ids,
            equipment_ids,
        )

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        predictions = self.fc_out(x.squeeze(0)).squeeze(-1)  # Shape: [batch_size]

        # Apply ReLU to ensure non-negative predictions
        self.logger.debug(f"Model predictions before activation: {predictions}")

        # predictions = torch.relu(predictions)
        # self.logger.debug(f"Model predictions after activation: {predictions}")

        self.logger.debug(f"Predictions from forward pass: {predictions}")
        if torch.isnan(predictions).any():
            self.logger.error("NaN values detected in model predictions.")

        return predictions

    def train_model(self, dataloader, optimizer, criterion, num_epochs, save_model, logger=None):
        start = time.time()  # Record the start time

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0

            for i, mini_batch in enumerate(dataloader):
                (
                    _neighborhood_ids,
                    _time_features,
                    _building_type_ids,
                    _building_counts,
                    _population,
                    _event_type_ids,
                    _equipment_ids,
                    _targets,
                ) = mini_batch

                optimizer.zero_grad()

                # Forward pass
                predictions = self(
                    _neighborhood_ids,
                    _time_features,
                    _building_type_ids,
                    _building_counts,
                    _population,
                    _event_type_ids,
                    _equipment_ids,
                )

                # Align targets with predictions
                if _targets.dim() == 3:  # If _targets has an extra dimension
                    _targets = _targets.squeeze(-1)  # Remove the last dimension

                # Check for NaN values
                if torch.isnan(predictions).any():
                    logger.error("NaN detected in predictions during training.")
                    predictions = torch.nan_to_num(predictions, nan=0.0)  # Replace NaN with 0

                if torch.isnan(_targets).any():
                    logger.error("NaN detected in targets during training.")
                    _targets = torch.nan_to_num(_targets, nan=0.0)  # Replace NaN with 0

                # Compute loss
                loss = criterion(predictions, _targets)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                mini_batch_loss = loss.item()
                epoch_loss += mini_batch_loss

                self.logger.debug(
                    f"Epoch [{epoch+1}/{num_epochs}], Mini-batch [{i + 1}/{len(dataloader)}], Loss: {mini_batch_loss:.4f}"
                )
            # fmt: on

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {(epoch_loss / len(dataloader)):.4f}"
                )
            else:
                self.logger.debug(
                    f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {(epoch_loss / len(dataloader)):.4f}"
                )

        if save_model:
            torch.save(self.state_dict(), "transformer_model_weekly.pth")
            self.logger.info("Model saved successfully.")

        end = time.time()  # Record the end time
        self.logger.info(f"Finish training transformer model, time elapsed {(end - start):.2f}s")

    def validate_model(self, dataloader):
        """
        Validate the model and calculate evaluation metrics.

        Args:
            dataloader (DataLoader): Validation DataLoader.

        Returns:
            tuple: All predictions and all targets.
        """
        self.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for mini_batch in dataloader:
                (
                    _neighborhood_ids,
                    _time_features,
                    _building_type_ids,
                    _building_counts,
                    _population,
                    _event_type_ids,
                    _equipment_ids,
                    _targets,
                ) = mini_batch

                # Forward pass
                predictions = self(
                    _neighborhood_ids,
                    _time_features,
                    _building_type_ids,
                    _building_counts,
                    _population,
                    _event_type_ids,
                    _equipment_ids,
                )

                # Collect predictions and targets
                all_predictions.append(predictions.cpu())
                all_targets.append(_targets.cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0).squeeze(-1)  # Flatten to [num_samples]
        all_targets = torch.cat(all_targets, dim=0).squeeze(-1)  # Flatten to [num_samples]

        # Convert to NumPy for sklearn metrics
        all_predictions_np = all_predictions.numpy()
        all_targets_np = all_targets.numpy()

        # Calculate metrics
        mae = mean_absolute_error(all_targets_np, all_predictions_np)
        mse = mean_squared_error(all_targets_np, all_predictions_np)
        self.logger.info(f"MAE: {mae}, MSE: {mse}")

        return all_predictions_np, all_targets_np

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def plot_result(self, predictions, targets, figure_path):
        start = time.time()

        # Figure 1: predicted weekly events in neighbourhood heat map
        fig1 = plt.figure(figsize=(12, 100))
        predictions_np = predictions[0, :].reshape(-1, 1)  # shape (377, 1)
        sns.heatmap(
            predictions_np,
            annot=True,
            cmap="coolwarm",
            cbar=True,
            fmt=".2f",
            xticklabels=[f"Week {i+1}" for i in range(predictions_np.shape[1])],
            yticklabels=[f"Neighborhood {i+1}" for i in range(predictions_np.shape[0])],
        )
        plt.title("Predicted Number of Events per Day for Each Neighborhood")
        plt.xlabel("Day of the Week")
        plt.ylabel("Neighborhood")
        fig1.savefig(os.path.join(figure_path, "1-heat-map.png"), dpi=fig1.dpi)

        # Figure 2 - Aggregate Predictions Across Neighborhoods
        # Aggregate across neighborhoods
        aggregated_predictions = predictions.mean(axis=1)  # Shape: [125]
        aggregated_targets = targets.mean(axis=1)  # Shape: [125]

        # Generate x-axis values
        x = np.arange(aggregated_predictions.shape[0])  # Shape: [125]

        # Plot
        fig2 = plt.figure(figsize=(10, 10))
        width = 0.35
        plt.bar(x - width / 2, aggregated_predictions, width=width, label="Predictions", color="skyblue")
        plt.bar(x + width / 2, aggregated_targets, width=width, label="Targets", color="orange")
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Events")
        plt.title("Predicted vs Actual Events (Aggregated Across Neighborhoods)")
        plt.legend()
        fig2.savefig(os.path.join(figure_path, "2-overall-predicted-vs-actual.png"), dpi=fig2.dpi)

        # Figure 3 - Plot Predictions for Individual Neighborhoods
        # Select a specific neighborhood (e.g., the first one)
        neighborhood_idx = 0
        single_neighborhood_predictions = predictions[:, neighborhood_idx]  # Shape: [125]
        single_neighborhood_targets = targets[:, neighborhood_idx]  # Shape: [125]

        # Generate x-axis values
        x = np.arange(single_neighborhood_predictions.shape[0])  # Shape: [125]

        # Plot
        fig3 = plt.figure(figsize=(10, 10))
        width = 0.35
        plt.bar(x - width / 2, single_neighborhood_predictions, width=width, label="Predictions", color="skyblue")
        plt.bar(x + width / 2, single_neighborhood_targets, width=width, label="Targets", color="orange")
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Events")
        plt.title(f"Predicted vs Actual Events for Neighborhood {neighborhood_idx}")
        plt.legend()
        fig3.savefig(os.path.join(figure_path, "3-neighbourhood-predicted-vs-actual.png"), dpi=fig3.dpi)

        # Figure 4 - Plot Multiple Neighborhoods Using Subplots
        # Number of neighborhoods to plot
        num_neighborhoods_to_plot = 5

        # Create subplots
        fig4, axes = plt.subplots(
            num_neighborhoods_to_plot, 1, figsize=(10, num_neighborhoods_to_plot * 3), sharex=True
        )

        for i in range(num_neighborhoods_to_plot):
            neighborhood_predictions = predictions[:, i]
            neighborhood_targets = targets[:, i]
            x = np.arange(neighborhood_predictions.shape[0])

            axes[i].bar(x - width / 2, neighborhood_predictions, width=width, label="Predictions", color="skyblue")
            axes[i].bar(x + width / 2, neighborhood_targets, width=width, label="Targets", color="orange")
            axes[i].set_title(f"Neighborhood {i}")
            axes[i].set_ylabel("Number of Events")
            axes[i].legend()

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout()
        fig4.savefig(os.path.join(figure_path, "4-neighbourhood-events.png"), dpi=fig4.dpi)

        end = time.time()
        self.logger.info(f"Finish plotting prediction result, time elapsed {(end - start):.2f}s")

        return
