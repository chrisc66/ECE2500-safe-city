# Save the model's state dictionary
# Define the file path to save the model
model_save_path = "emergency_event_predictor.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

# Load the model's state dictionary
# Define the file path where the model was saved
model_save_path = "emergency_event_predictor.pth"

# Recreate the model architecture - this is just an example
loaded_model = EmergencyEventPredictor(
    embedding_module=embedding_module,  
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    max_len=7
)

# Load the state dictionary into the model
loaded_model.load_state_dict(torch.load(model_save_path))

# Set the model to evaluation mode
loaded_model.eval()

print("Model loaded successfully and set to evaluation mode.")

# Code to test this save and load process
# Test on validation data
all_predictions, all_targets = validate_model(loaded_model, val_dataloader)

# Visualize results
plot_predictions_vs_true(all_predictions, all_targets)
plot_heatmap(all_predictions, val_neighborhood_ids)
