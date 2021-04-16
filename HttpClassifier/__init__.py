import logging

import azure.functions as func
import numpy as np
import joblib


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    classifier_model = joblib.load('model_desafio2_RandomForestClassifier.pkl')

    j_data = np.array(req.get_json()['data'])
    #y_hat = np.array2string(classifier_model.predict(j_data))
    y_hat = classifier_model.predict(j_data).tolist()
    return func.HttpResponse(str(y_hat))