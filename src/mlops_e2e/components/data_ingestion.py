from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging

from mlops_e2e.entity.config_entity import DataIngestionConfig
from mlops_e2e.entity.artifact_entity import DataIngestionArtifact

import os 
import sys
import pandas as pd
import numpy as np
import pymongo
import certifi
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            logging.info(f"Connecting to MongoDB at {MONGO_DB_URL}")
            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
            db = client[self.data_ingestion_config.database_name]
            collection = db[self.data_ingestion_config.collection_name]
            logging.info(f"Fetching data from collection: {self.data_ingestion_config.collection_name}")
            data = list(collection.find())
            df = pd.DataFrame(data)
            
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            df.replace(to_replace="na", value=np.nan, inplace=True)

            logging.info(f"Data fetched successfully with shape: {df.shape}")
            return df
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, df: pd.DataFrame) -> str:
        try:
            
            feature_store_file_path = self.data_ingestion_config.feature_store_dir
            os.makedirs(
                os.path.dirname(feature_store_file_path), 
                exist_ok=True
                )
            df.to_csv(feature_store_file_path, index=False)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")
            return df
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            train_file_path =self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)

            logging.info(f"Train and test data saved at {train_file_path} and {test_file_path} respectively.")
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)