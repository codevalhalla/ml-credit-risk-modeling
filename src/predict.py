import os
import joblib
import numpy as np
import pandas as pd


class CreditRiskPredictor:
    def __init__(self):
        # Resolve base directory (risk-modeling/)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Paths
        self.model_path = os.path.join(self.base_dir, "artifacts/model/xgb_final_model.pkl")
        self.cat_path = os.path.join(self.base_dir, "artifacts/preprocessing/categorical_transformer.pkl")
        self.num_path = os.path.join(self.base_dir, "artifacts/preprocessing/numerical_transformer.pkl")
        self.threshold_path = os.path.join(self.base_dir, "artifacts/threshold.txt")

        # Load artifacts
        self.model = joblib.load(self.model_path)
        self.categorical_transformer = joblib.load(self.cat_path)
        self.numerical_transformer = joblib.load(self.num_path)

        if os.path.exists(self.threshold_path):
            with open(self.threshold_path) as f:
                self.threshold = float(f.read())
        else:
            # Default fallback (should not happen in production)
            self.threshold = 0.60


    def predict(self, input_data: dict):
        """
        input_data: dictionary with raw feature values
        returns: probability and final prediction
        """
        df = pd.DataFrame([input_data])

        # Numerical features
        num_features = self.numerical_transformer.feature_names_in_
        X_num = self.numerical_transformer.transform(df[num_features])

        # Categorical features
        cat_features = self.categorical_transformer.transformers_[0][2]
        X_cat = self.categorical_transformer.transform(df[cat_features])

        # Combine features
        X = np.concatenate([X_num, X_cat], axis=1)

        # Predict
        prob = self.model.predict_proba(X)[0, 1]
        prediction = int(prob >= self.threshold)

        return {
            "default_probability": float(prob),
            "prediction": prediction
        }


if __name__ == "__main__":
    # Example test case
    predictor = CreditRiskPredictor()

    sample_input = {
        "age": 35,
        "income": 55000,
        "loanamount": 15000,
        "creditscore": 680,
        "monthsemployed": 48,
        "numcreditlines": 4,
        "interestrate": 12.5,
        "loanterm": 36,
        "dtiratio": 0.32,
        "education": "Bachelor's",
        "employmenttype": "Full-time",
        "maritalstatus": "Married",
        "hasmortgage": "Yes",
        "hasdependents": "No",
        "loanpurpose": "Auto",
        "hascosigner": "No"
    }

    result = predictor.predict(sample_input)
    print(result)
