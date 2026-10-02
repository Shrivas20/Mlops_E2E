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