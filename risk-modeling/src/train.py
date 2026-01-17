import os
import joblib
import numpy as np
from xgboost import XGBClassifier


def main():
    # Paths
    ARTIFACTS_DIR = "../artifacts"
    DATA_DIR = os.path.join(ARTIFACTS_DIR, "transformed_data")
    MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load transformed data
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    # Class imbalance ratio (used during training)
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    # Final XGBoost model (fixed, tuned configuration)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    model_path = os.path.join(MODEL_DIR, "xgb_final_model.pkl")
    joblib.dump(model, model_path)

    print("Training complete.")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
