import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from helper_functions import log_info, log_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Define paths
ARTIFACTS_PATH = "D:/099-Lab/MLOPSLab/mlops2025-DSC/Artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")
def training_pipeline(X_train, y_train, experiment_name="xgb-classifier-exp", model_name="xgb-classifier"):
    """
    Trains an XGBoost classifier, logs to MLflow, and saves model locally and to the MLflow Model Registry.
    """
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            model = XGBClassifier(
                n_estimators=100, 
                max_depth=3, 
                learning_rate=0.1, 
                use_label_encoder=False, 
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train)
            
            # Local save
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            log_info(f"Model trained and saved locally at {MODEL_PATH}")
            
            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Log parameters, metrics, and model to MLflow
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 3)
            mlflow.log_param("learning_rate", 0.1)

            acc = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", acc)

            # Log model and register it
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="xgb-model",
                signature=signature,
                registered_model_name=model_name
            )

            log_info(f"Model logged to MLflow under name '{model_name}'.")

        return model
    except Exception as e:
        log_error(f"Error during model training with MLflow: {e}")
        raise


def load_model():
    """
    Loads the trained model from file.
    """
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        log_info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        log_error(f"Model file not found at {MODEL_PATH}")
        raise

def prediction_pipeline(X_val):
    """
    Makes predictions using the trained model.
    """
    try:
        model = load_model()
        with open(LABEL_ENCODER_PATH, 'rb') as file:
            label_encoder = pickle.load(file)
        
        predictions = model.predict(X_val)
        predictions = label_encoder.inverse_transform(predictions)
        
        return predictions
    except FileNotFoundError as e:
        log_error(f"Error loading model or label encoder: {e}")
        raise

def evaluation_matrices(X_val, y_val):
    """
    Evaluates the model using confusion matrix, accuracy, and classification report.
    Logs metrics to MLflow.
    """
    try:
        pred_vals = prediction_pipeline(X_val)

        with open(LABEL_ENCODER_PATH, 'rb') as file:
            label_encoder = pickle.load(file)
        decoded_y_vals = label_encoder.inverse_transform(y_val)

        conf_matrix = confusion_matrix(decoded_y_vals, pred_vals, labels=label_encoder.classes_)
        acc_score = accuracy_score(decoded_y_vals, pred_vals)
        class_report = classification_report(decoded_y_vals, pred_vals, output_dict=True)

        mlflow.log_metric("val_accuracy", acc_score)
        for cls, metrics in class_report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"{cls}_precision", metrics.get("precision", 0))
                mlflow.log_metric(f"{cls}_recall", metrics.get("recall", 0))

        log_info("Model evaluation completed.")
        log_info(f"Confusion Matrix:\n{conf_matrix}")
        log_info(f"Accuracy Score: {acc_score}")
        log_info(f"Classification Report:\n{classification_report(decoded_y_vals, pred_vals)}")

        return conf_matrix, acc_score, class_report
    except FileNotFoundError:
        log_error("Label encoder file not found.")
        raise
