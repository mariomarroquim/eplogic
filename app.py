from waitress import serve
from flask import Flask, request, render_template, jsonify, make_response

import warnings; warnings.simplefilter('ignore')
import pandas                  as pd
import numpy                   as np
import sklearn.model_selection as ms
import sklearn.metrics         as mt
from imblearn.under_sampling   import RandomUnderSampler
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import RandomizedSearchCV
from joblib                    import dump, load

eplogic = load('notebooks/eplogic.joblib')

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
  return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
  try:
    eplet_data = [[(0, 1)[request.args.get('eplet_locus') == 'abc'],
                   (0, 1)[request.args.get('eplet_locus') == 'drb'],
                   (0, 1)[request.args.get('eplet_locus') == 'dq'],
                   (0, 1)[request.args.get('eplet_locus') == 'dp'],
                   int(request.args.get('panel_nc')),
                   int(request.args.get('panel_pc')),
                   int(request.args.get('eplet_allele_qtd')),
                   int(request.args.get('eplet_min_mfi')),
                   int(request.args.get('eplet_max_mfi'))]]


    results = eplogic.predict(eplet_data)
    probabilities = eplogic.predict_proba(eplet_data)

    predictions = jsonify(label=str(results[0]), score0=str(probabilities[0][0]), score1=str(probabilities[0][1]))

    response = make_response(predictions)
    response.mimetype = 'application/json'
    return response
  except:
    return "Please, verify if all parameters (eplet_locus, eplet_allele_qtd, eplet_min_mfi, eplet_max_mfi, panel_nc, panel_pc) are present and contain integer values.", 500

if __name__ == "__main__":
  #app.run(host='0.0.0.0')
  serve(app, host='0.0.0.0', port=80)
