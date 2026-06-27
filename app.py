import pickle
from flask import Flask,request,render_template,jsonify,url_for,app
import pandas as pd
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = list(map(float, request.form.values()))
    data = np.array(data).reshape(1, -1)

    output = model.predict(data)[0]

    if output == 0:
        return render_template(
            'output.html',
            prediction="You are safe buddy ❤️"
        )

    return render_template(
        'output.html',
        prediction="This is the end 🫠"
    )
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
    # app.run(debug=True)