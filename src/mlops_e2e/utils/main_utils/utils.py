import yaml
from mlops_e2e.exception.execption import NetworkSecurityException
from mlops_e2e.logging.logger import logging
import os, sys
import numpy as np
import pandas as pd
import dill
import pickle


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def write_yaml_file(file_path:str, content:dict=None, replace:bool=False)->None:
    try:
        if replace:
            os.remove(file_path)
        with open(file_path, "w") as yaml_file:
            if content is not None:
                yaml.dump(content, yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_numpy_array_data(file_path:str, array:np.array)->None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_object(file_path:str, obj:object)->None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
