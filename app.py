import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np 
import pandas as pd


app = Flask(__name__)

# load the model
model=pickle.load(open("regmodel.pkl", 'rb'))
scalar=pickle.load(open("scaling.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    input_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scalar.transform(input_array)
    output = model.predict(new_data)
    print(output[0])
    return jsonify({"prediction": float(output[0])})


if __name__ == "__main__":
    app.run(debug=True)

