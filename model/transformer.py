import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module
    """

    def __init__(self, logger, embed_dim, max_len=7):  # 7 days in a week
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        logger.info(f"sin position {position.shape}, div_term {div_term.shape}, pe {pe.shape}")
        pe[:, 0::2] = torch.sin(position * div_term)
        logger.info(f"cos position {position.shape}, div_term {div_term.shape}, pe {pe.shape}")
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return x


class ST_TEM(nn.Module):
    def __init__(self, logger, embedding_module, embed_dim, num_heads, num_layers, max_len=7):
        super(ST_TEM, self).__init__()
        self.logger = logger

        # Embedding module (CombinedEmbedding) and positional encoding
        self.embedding_module = embedding_module
        self.positional_encoding = PositionalEncoding(logger=logger, embed_dim=embed_dim, max_len=max_len)

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

        # Prediction layer (we apply it to each element in the sequence)
        # Albert: predictions = self.fc_out(x).squeeze(-1)  # Shape: [batch_size]
        predictions = self.fc_out(x.squeeze(1)).squeeze(-1)  # Shape: [batch_size]

        return predictions

    def train_model(self, optimizer, criterion, num_epochs, dataloader, save_model):
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0

            # fmt: off
            for i, (_neighborhood_ids, _time_features, _building_type_ids, _building_counts,
                    _population, _event_type_ids, _equipment_ids, _targets) in enumerate(dataloader):
                optimizer.zero_grad()

                # Forward pass
                _predictions = self(_neighborhood_ids, _time_features, _building_type_ids,
                                    _building_counts, _population, _event_type_ids, _equipment_ids)

                # Compute loss
                loss = criterion(_predictions, _targets)

                # Backward pass and optimization
                loss.backward()
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

        self.logger.info(f"Training loop finished")

        if save_model:
            torch.save(self.state_dict(), "transformer_model_weekly.pth")
            self.logger.info("Model saved successfully.")

        return

    def validate_model(self, dataloader):
        self.eval()  # Set model to evaluation mode
        all_predictions = []
        all_targets = []

        with torch.no_grad():  # Disable gradient computation
            # fmt: off
            for i, (_neighborhood_ids, _time_features, _building_type_ids, _building_counts,
                _population, _event_type_ids, _equipment_ids, _targets) in enumerate(dataloader):
                
                # Forward pass
                _predictions = self(_neighborhood_ids, _time_features, _building_type_ids,
                                    _building_counts, _population, _event_type_ids, _equipment_ids)

                # Collect predictions and true values
                all_predictions.append(_predictions.cpu().numpy())
                all_targets.append(_targets.cpu().numpy())
            # fmt: on

        # Combine all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Compute metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_predictions)

        self.logger.info(f"Validation Results: MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
        self.logger.info(f"Target shape {all_targets.shape}, Prediction shape {all_predictions.shape}")
        self.logger.info(f"Last predictions {_predictions}")

        return all_predictions, all_targets

    def plot_result(self, predictions, targets):
        # Figure 1: predicted weekly events in neighbourhood heat map
        fig1 = plt.figure(figsize=(12, 100))
        predictions_np = predictions.reshape(-1, 1)  # shape (377, 1)
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
        fig1.savefig("figure_1.png", dpi=fig1.dpi)

        # Figure 2
        fig2 = plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5, edgecolor="k")
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], "r--")
        plt.title("Predicted vs True Event Counts")
        plt.xlabel("True Event Counts")
        plt.ylabel("Predicted Event Counts")
        plt.grid()
        fig2.savefig("figure_2.png", dpi=fig2.dpi)

        # Figure 3
        fig3 = plt.figure(figsize=(30, 5))
        x = np.arange(len(predictions))  # Positions for the first list
        width = 0.4  # Width of the bars
        plt.bar(x - width / 2, predictions, width=width, label="predictions", color="skyblue")
        plt.bar(x + width / 2, targets, width=width, label="targets", color="orange")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.title("Side-by-Side Bar Plot of Two Lists")
        plt.xticks(x, [f"Item {i}" for i in range(len(predictions))])  # Label x-axis ticks
        plt.legend()
        plt.tight_layout()
        fig3.savefig("figure_3.png", dpi=fig3.dpi)

        return
