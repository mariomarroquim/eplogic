from waitress import serve
from flask import Flask, request, render_template, jsonify, make_response
import pandas as pd
from pycaret.classification import *

vxm_model = load_model('notebook/vxm_model')

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
  return render_template('index.html')

@app.route("/predict", methods=['GET'])
def predict():
  try:
    data = {
      'max_mfi_a':   [None],
      'max_mfi_b':   [None],
      'max_mfi_c':   [None],
      'max_mfi_drb': [None],
      'max_mfi_dqa': [None],
      'max_mfi_dqb': [None],
      'max_mfi_dpa': [None],
      'max_mfi_dpb': [None]
    }

    # Refactor this ugly code
    if request.args.get('max_mfi_a') != None and request.args.get('max_mfi_a') != '':
      data['max_mfi_a'] = [request.args.get('max_mfi_a')]

    if request.args.get('max_mfi_b') != None and request.args.get('max_mfi_b') != '':
      data['max_mfi_b'] = [request.args.get('max_mfi_b')]

    if request.args.get('max_mfi_c') != None and request.args.get('max_mfi_c') != '':
      data['max_mfi_c'] = [request.args.get('max_mfi_c')]

    if request.args.get('max_mfi_drb') != None and request.args.get('max_mfi_drb') != '':
      data['max_mfi_drb'] = [request.args.get('max_mfi_drb')]

    if request.args.get('max_mfi_dqa') != None and request.args.get('max_mfi_dqa') != '':
      data['max_mfi_dqa'] = [request.args.get('max_mfi_dqa')]

    if request.args.get('max_mfi_dqb') != None and request.args.get('max_mfi_dqb') != '':
      data['max_mfi_dqb'] = [request.args.get('max_mfi_dqb')]

    if request.args.get('max_mfi_dpa') != None and request.args.get('max_mfi_dpa') != '':
      data['max_mfi_dpa'] = [request.args.get('max_mfi_dpa')]

    if request.args.get('max_mfi_dpb') != None and request.args.get('max_mfi_dpb') != '':
      data['max_mfi_dpb'] = [request.args.get('max_mfi_dpb')]

    instances = pd.DataFrame(data, columns = ['max_mfi_a',
                                              'max_mfi_b',
                                              'max_mfi_c',
                                              'max_mfi_drb',
                                              'max_mfi_dqa',
                                              'max_mfi_dqb',
                                              'max_mfi_dpa',
                                              'max_mfi_dpb'])

    new_predictions = predict_model(vxm_model, data=instances)

    output = jsonify(label=str(new_predictions['Label'][0]), score=str(new_predictions['Score'][0]))

    response = make_response(output)
    response.mimetype = 'application/json'
    return response

  except:
    return "Please, verify if all parameters (max_mfi_a, max_mfi_b, max_mfi_c, max_mfi_drb, max_mfi_dqa, max_mfi_dqb, max_mfi_dpa and max_mfi_dpb) contain integer values.", 500

if __name__ == "__main__":
  #app.run(host='0.0.0.0')
  serve(app, host='0.0.0.0', port=80)
