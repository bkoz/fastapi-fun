# -*- coding: utf-8 -*-
"""
A simple ML model server using FastAPI.
"""

from anyio import Condition
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
import joblib
import numpy
import json
import urllib.request
import os
import logging

class TheRequest(BaseModel):
    """
    Define the request schema.
    """
    X: list = Field(default=[1,2,3,4])

logging.basicConfig(level=logging.INFO)
logging.info("model_server: is running.")

app = FastAPI()

logging.debug(f"model_server: TheRequest =\
              {TheRequest.schema_json(indent=2)}")

# 
# Load the scikit-learn model from storage.
#
url = "https://koz.s3.amazonaws.com/models/sklearn/iris-svc.joblib"
filename = os.path.join(os.getcwd(), "iris-svc.joblib")
logging.info(f'model_server: Loading {url}')
urllib.request.urlretrieve(url, filename)
logging.info(f'model_server: Retrieved {url}')
logging.info(f"model_server: Loading model: {filename}")
clf = joblib.load('iris-svc.joblib')

logging.debug(f"model_server: Model params: {clf.get_params()}")

@app.get("/")
@app.get("/v2")
async def welcome()-> str:
    """
    Welcome greeting string
    """
    logging.info("model_server: get()")
    return {"greeting": "Welcome to the ML model server."}

@app.get("/v2/models/iris-svm")
async def usage()-> str:
    """ 
    Return model schema
    """
    schema = {
        "X": [
            1.5,
            2.8,
            3.4,
            4.3
        ]
    }
    return schema

@app.post("/v2/models/iris-svm/infer")
async def predict(body: TheRequest) -> str:
    """
    Description: The prediction endpoint. 
    Converts the REST request body to a numpy array
    and returns a prediction.

    Args:
                 body: The request body.

    Returns:
                 A json output string where:
                 0, 1, or 2 representing the Iris flower class
    """
    X = numpy.array([body.X])
    ret = {}
    ret['output'] = clf.predict(X).tolist()
    logging.debug(f"model_server: request = {X}, prediction = {ret}")
    return ret


