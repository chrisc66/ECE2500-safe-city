import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluation Loop Code
# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for (neighborhood_ids, time_features, building_type_ids, building_counts,
             population, event_type_ids, equipment_ids, targets) in dataloader:
            
            # Forward pass
            predictions = model(neighborhood_ids, time_features, building_type_ids,
                                building_counts, population, event_type_ids, equipment_ids)
            
            # Collect predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Combine all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)

    print(f"Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    return mae, mse, rmse, r2

# Example usage
test_dataset = NeighborhoodDataset(test_neighborhood_ids, test_time_features, test_building_type_ids,
                                   test_building_counts, test_population, test_event_type_ids,
                                   test_equipment_ids, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate the model
evaluate_model(model, test_dataloader)

# Visualizing Predictions
import matplotlib.pyplot as plt

# Scatter plot: Predicted vs True Values
plt.figure(figsize=(8, 8))
plt.scatter(all_targets, all_predictions, alpha=0.5)
plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
plt.title("Predicted vs True Event Counts")
plt.xlabel("True Event Counts")
plt.ylabel("Predicted Event Counts")
plt.grid()
plt.show()

# Save the model for future use
torch.save(model.state_dict(), "emergency_event_predictor.pth")

# To load the model later
model = EmergencyEventPredictor(
    embedding_module=embedding_module,
    embed_dim=64,
    num_heads=4,
    num_layers=2
)
model.load_state_dict(torch.load("emergency_event_predictor.pth"))
model.eval()



