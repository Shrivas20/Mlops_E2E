''' 
  Sample connection to MongoDB

    import certifi
    import os
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
    from dotenv import load_dotenv

    load_dotenv()
    uri = os.getenv("MONGO_DB_URL")

    client = MongoClient(
        uri,
        server_api=ServerApi("1"),
        tlsCAFile=certifi.where()
    )

    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
'''


import json
import os 
import sys
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np 
from src.mlops_e2e.logging.logger import logging
from src.mlops_e2e.exception.execption import NetworkSecurityException

load_dotenv()  # Load environment variables from .env file

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# print(f"MONGO_DB_URL: {MONGO_DB_URL}")


class NetworkDataExtract():
    def __init__(self):
        try:
           pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json(self,filepath):
        try:
            df = pd.read_csv(filepath)
            df.reset_index(drop=True, inplace=True)
            json_data = json.loads(df.to_json(orient='records'))
            return json_data
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def push_data_to_mongodb(self, data, database, collection_name):
        try:
            self.database = database
            self.collection_name = collection_name
            self.data = data

            self.client = MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where()
            )
            self.db = self.client[self.database]
            self.collection = self.db[self.collection_name]
            self.collection.insert_many(self.data)
            logging.info(f"Data pushed to MongoDB collection: {self.collection_name} in database {self.database}")

            return len(self.data)
        
        except Exception as e:
                raise NetworkSecurityException(e, sys)

if __name__ == "__main__": 

    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE_NAME = "NetworkSecurity"
    COLLECTION_NAME = "PhishingData"

    network_data_extract = NetworkDataExtract()
    json_data = network_data_extract.csv_to_json(FILE_PATH)
    records_inserted = network_data_extract.push_data_to_mongodb(json_data, DATABASE_NAME, COLLECTION_NAME)
    logging.info(f"Number of records inserted: {records_inserted}")