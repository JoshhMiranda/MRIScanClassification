import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model

# Load your model
model = load_model("artifacts/mri_classifier_local_v3.h5")

# Set up MLflow experiment
mlflow.set_experiment("mri_scan_classification_v1")

with mlflow.start_run() as run:
    # Log the Keras model
    mlflow.keras.save_model(model, "mri_scan_classifier")

    # Print the run ID
    print("Run ID:", run.info.run_id)

print("Model saved to MLflow")