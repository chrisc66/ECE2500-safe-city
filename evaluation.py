import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Splitting neighborhoods into train and validation sets
train_indices, val_indices = train_test_split(
    np.arange(num_neighborhoods), test_size=0.2, random_state=42
)

# Prepare validation tensors
val_neighborhood_ids = neighborhood_ids[val_indices]
val_time_features = time_features[val_indices]
val_building_type_ids = building_type_ids[val_indices]
val_building_counts = building_counts[val_indices]
val_population = population[val_indices]
val_event_type_ids = event_type_ids[val_indices]
val_equipment_ids = equipment_ids[val_indices]
val_targets = targets[val_indices]

# Create Validation Dataset and DataLoader
val_dataset = NeighborhoodDataset(
    neighborhood_ids=val_neighborhood_ids,
    time_features=val_time_features,
    building_type_ids=val_building_type_ids,
    building_counts=val_building_counts,
    population=val_population,
    event_type_ids=val_event_type_ids,
    equipment_ids=val_equipment_ids,
    targets=val_targets
)

val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Validation Loop
def validate_model(model, val_dataloader):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation
        for (neighborhood_ids, time_features, building_type_ids, building_counts,
             population, event_type_ids, equipment_ids, targets) in val_dataloader:
            
            # Forward pass
            predictions = model(neighborhood_ids, time_features, building_type_ids,
                                building_counts, population, event_type_ids, equipment_ids)
            
            # Collect predictions and true values
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

    print(f"Validation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return all_predictions, all_targets

# Scatter Plot: Predicted vs. True Values
def plot_predictions_vs_true(all_predictions, all_targets):
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_predictions, alpha=0.5, edgecolor='k')
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], 'r--')
    plt.title("Predicted vs True Event Counts")
    plt.xlabel("True Event Counts")
    plt.ylabel("Predicted Event Counts")
    plt.grid()
    plt.show()

# Heatmap: Neighborhood Predictions Over Time
def plot_heatmap(all_predictions, neighborhood_ids, days=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]):
    plt.figure(figsize=(12, 10))
    sns.heatmap(all_predictions, annot=True, cmap="coolwarm", cbar=True, fmt=".2f",
                xticklabels=days,
                yticklabels=[f"Neighborhood {i}" for i in neighborhood_ids])
    plt.title("Predicted Event Counts per Neighborhood per Day")
    plt.xlabel("Days")
    plt.ylabel("Neighborhood")
    plt.show()

# Perform validation
all_predictions, all_targets = validate_model(model, val_dataloader)

# Visualize results
plot_predictions_vs_true(all_predictions, all_targets)
plot_heatmap(all_predictions, val_neighborhood_ids)
