import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion")
        try:
            df = pd.read_csv('notebook\data\DataCoSupplyChainDataset.csv', encoding="ISO-8859-1")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion complete: Train and Test sets saved in artifacts")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)
        
        # Save this at the bottom of data_ingestion.py and run: python src/components/data_ingestion.py
if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        print(f"Success! Files saved at: {train_path} and {test_path}")
    except Exception as e:
        print(f"Failed! Error: {e}")