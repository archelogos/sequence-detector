# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import tensorflow as tf
import numpy as np

import sys
sys.path.append('svhn')
import model
import preprocessing

# Input data.
x = tf.placeholder(tf.float32, shape=(1, 32, 32, 1))
CNN = model.CNN()
pred_aux = CNN.predict(x)
raw_preds = pred_aux
y = tf.transpose(tf.argmax(pred_aux, 2))
# Create a saver.
saver = tf.train.Saver(tf.all_variables())

# Build an initialization operation to run below.
init = tf.initialize_all_variables()

# Start running operations on the Graph.
session = tf.Session()
session.run(init)

saver.restore(session, "svhn/trained_models/model-gcp-200k.ckpt")
print('Model Restored')


def predict_svhn(inp):
  arr_data = session.run([raw_preds, y], feed_dict={x: inp})
  data = {}
  data['raw_preds'] = arr_data[0].flatten().tolist()
  data['y'] = arr_data[1].flatten().tolist()
  return data


################################################

import logging


from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
  #print(request.json['imageId'])
  img_index = request.json['imageId']
  original, processed, label = preprocessing.rtp_data_processing(img_index)
  processed = processed.reshape((-1, 32, 32, 1)).astype(np.float32)
  output = predict_svhn(processed)
  return jsonify(results=output)

@app.route('/')
def main():
  return render_template('index.html')

@app.errorhandler(500)
def server_error(e):
  logging.exception('An error occurred during a request.')
  return """
  An internal error occurred: <pre>{}</pre>
  See logs for full stacktrace.
  """.format(e), 500

if __name__ == '__main__':
  # This is used when running locally. Gunicorn is used to run the
  # application on Google App Engine. See entrypoint in app.yaml.
  app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
