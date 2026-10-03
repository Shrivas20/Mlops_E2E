import os, sys, certifi, pymongo
from dotenv import load_dotenv
import pandas as pd 

from src.mlops_e2e.exception.execption import NetworkSecurityException
from src.mlops_e2e.logging.logger import logging
from src.mlops_e2e.pipeline.training_pipeline import TrainingPipeline
from src.mlops_e2e.utils.main_utils.utils import load_object
from src.mlops_e2e.utils.ml_utils.model.estimator import NetworkSecurityModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run 
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.mlops_e2e.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME


mongo_db_url = os.getenv("MONGO_DB_URL")
client = pymongo.MongoClient(mongo_db_url, tlsCAFile = certifi.where())

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)

@app.post("/predict")
async def predict_route(request:Request, file:UploadFile=(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkSecurityModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df['predicted_coulum'] = y_pred

        df.to_csv("prediction_output/output.csv")
        table_html = df.to_html(classes='table table-stripped')
        return templates.TemplateResponse(
                request=request,
                name="table.html",
                context={
                    "request": request,
                    "table": table_html
                }
            )
    except Exception as e:
        raise NetworkSecurityException(e,sys)

if __name__ == "__main__":
    app_run(app,host="0.0.0.0",port=8080)