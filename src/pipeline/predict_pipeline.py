import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            kmeans_path = os.path.join("artifacts", "kmeans.pkl")

            logging.info("Loading models and preprocessors")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            kmeans = load_object(file_path=kmeans_path)

            # 1. Engineering features exactly like the training data
            features['order date (DateOrders)'] = pd.to_datetime(features['order date (DateOrders)'])
            features['shipping date (DateOrders)'] = pd.to_datetime(features['shipping date (DateOrders)'])
            features['OrderDuration'] = (features['shipping date (DateOrders)'] - features['order date (DateOrders)']).dt.days

            # 2. Hybrid Step: Assign Segment ID using saved K-Means
            segment_cols = ['Order Item Quantity', 'Sales', 'Benefit per order', 'Order Item Total', 'OrderDuration']
            features['segment_id'] = kmeans.predict(features[segment_cols])

            # 3. Transform and Predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """Maps web/app inputs to a DataFrame format the model understands."""
    def __init__(self, type_val, shipping_mode, region, market, sales, qty, total, benefit, discount, order_date, ship_date):
        self.type_val = type_val
        self.shipping_mode = shipping_mode
        self.region = region
        self.market = market
        self.sales = sales
        self.qty = qty
        self.total = total
        self.benefit = benefit
        self.discount = discount
        self.order_date = order_date
        self.ship_date = ship_date

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Type": [self.type_val],
                "Shipping Mode": [self.shipping_mode],
                "Order Region": [self.region],
                "Market": [self.market],
                "Sales": [self.sales],
                "Order Item Quantity": [self.qty],
                "Order Item Total": [self.total],
                "Benefit per order": [self.benefit],
                "Order Item Discount": [self.discount],
                "order date (DateOrders)": [self.order_date],
                "shipping date (DateOrders)": [self.ship_date]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
# Add this at the bottom of src/pipeline/predict_pipeline.py to test
if __name__ == "__main__":
    # 1. Create dummy data (simulating a single order)
    # Ensure these names match what your model expects
    data = CustomData(
        type_val="DEBIT",
        shipping_mode="Standard Class",
        region="Southeast Asia",
        market="Pacific Asia",
        sales=327.75,
        qty=1,
        total=314.64,
        benefit=91.25,
        discount=13.11,
        order_date="1/31/2018 22:56",
        ship_date="2/3/2018 22:56"
    )
    
    # 2. Convert to DataFrame
    df = data.get_data_as_data_frame()
    
    # 3. Run through pipeline
    pipeline = PredictPipeline()
    result = pipeline.predict(df)
    
    print(f"Prediction Result: {'LATE RISK' if result[0] == 1 else 'ON TIME'}")