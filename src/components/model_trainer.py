import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Tournament of Classifiers
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            # Hyperparameter Grids
            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy']},
                "Random Forest": {'n_estimators': [32, 64, 128]},
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [32, 64]
                },
                "Logistic Regression": {},
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [32, 64]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.5],
                    'n_estimators': [32, 64]
                }
            }

            logging.info("Starting model evaluation with hyperparameter tuning...")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
         
            # --- ADD THIS SECTION TO PRINT THE LEADERBOARD ---
            print("\n" + "="*35)
            print("MODEL PERFORMANCE LEADERBOARD (F1)")
            print("="*35)
            # Sort the models by score so the best is at the top
            sorted_report = dict(sorted(model_report.items(), key=lambda item: item[1], reverse=True))
            
            for model_name, score in sorted_report.items():
                print(f"{model_name:<20} : {score:.4f}")
            print("="*35 + "\n")
            # ------------------------------------------------

            # Get best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance")
            
            logging.info(f"Tournament Winner: {best_model_name} with F1 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

# Final Execution Block
if __name__ == "__main__":
    try:
        # Load the arrays created by the transformation script
        # Note: In a full pipeline run, these would be passed directly
        import numpy as np
        # This assumes you are running after transformation successful
        # For testing, we can simulate the flow:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation
        
        ingestion = DataIngestion()
        train_p, test_p = ingestion.initiate_data_ingestion()
        
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_p, test_p)
        
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Final Model F1 Score: {score}")
        
    except Exception as e:
        print(f"Training Failed: {e}")