import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Hyperparameter Tuning using Grid Search
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            # Using F1-Score for Supply Chain Risk Classification
            test_model_score = f1_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        # --- FIX 1: MUST RETURN THE REPORT ---
        return report

    # --- FIX 2: MUST HAVE EXCEPT BLOCK ---
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Loads a python object (like a model or scaler) from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
