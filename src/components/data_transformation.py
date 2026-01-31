import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    kmeans_model_file_path: str = os.path.join('artifacts', "kmeans.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Sets up the preprocessing pipeline for categorical and numerical features."""
        try:
            # We keep these based on your EDA importance
            numerical_columns = ['Sales', 'Order Item Quantity', 'Order Item Total', 'OrderDuration', 'Order Item Discount']
            categorical_columns = ['Type', 'Shipping Mode', 'Order Region', 'Market', 'segment_id']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Categorical and Numerical pipelines created.")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Calculating OrderDuration (Shipping Date - Order Date)")
            for df in [train_df, test_df]:
                df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
                df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])
                df['OrderDuration'] = (df['shipping date (DateOrders)'] - df['order date (DateOrders)']).dt.days

            # --- HYBRID K-MEANS STEP ---
            logging.info("Starting K-Means Segmentation")
            segment_cols = ['Order Item Quantity', 'Sales', 'Benefit per order', 'Order Item Total', 'OrderDuration']
            scaler_km = StandardScaler()
            kmeans = KMeans(n_clusters=4, random_state=42)
            
            # Fitting K-Means on scaled numerical subset
            train_df['segment_id'] = kmeans.fit_predict(scaler_km.fit_transform(train_df[segment_cols]))
            test_df['segment_id'] = kmeans.predict(scaler_km.transform(test_df[segment_cols]))
            
            # Save the K-Means model to use later in prediction
            save_object(self.data_transformation_config.kmeans_model_file_path, kmeans)

            # --- PREPARING FOR XGBOOST ---
            target_column_name = "Late_delivery_risk"
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            
            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # --- ADD THIS FIX HERE ---
            # If the output is a sparse matrix, convert it to a dense array
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()
            # -------------------------

            # Now np.c_ will work perfectly
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
        
# --- CORRECT THIS SECTION BELOW ---
# Ensure there are ZERO spaces before 'if'
if __name__ == "__main__":
    try:
        # Paths to the artifacts created by your data_ingestion.py
        train_path = os.path.join("artifacts", "train.csv")
        test_path = os.path.join("artifacts", "test.csv")
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        print("Verification Successful!")
        print(f"Shape of Transformed Training Data: {train_arr.shape}")
        
    except Exception as e:
        print(f"Transformation Failed: {e}")