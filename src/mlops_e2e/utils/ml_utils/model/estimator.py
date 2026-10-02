import os, sys

from mlops_e2e.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging 

class NetworkSecurityModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def predict(self, X):
        try:
            X = self.preprocessor.transform(X)
            return self.model.predict(X)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def save_model(self, file_path:str)->None:
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                import dill
                dill.dump(self.model, file_obj)
        except Exception as e:
            raise NetworkSecurityException(e, sys)