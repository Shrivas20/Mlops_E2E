import sys 
from mlops_e2e.logging.logger import logging

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        self.errormessage = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
        

    def __str__(self):
        return f"Error occurred in script: [{self.filename}] at line number: [{self.lineno}] error message: [{self.errormessage}]"

if __name__ == "__main__":
    try:
        a = 1/0
        print("This will not be printed",a)
    except Exception as e:
        logging.info("Divide by zero error")
        raise NetworkSecurityException(e, sys)