from flask import Flask,request
import pickle
import numpy as np

model = pickle.load(open('final_write.sav','rb'))

app = Flask(__name__)
@app.route('/',methods=['POST'])

def predict_write():
    df = request.json
    math = df['Math']
    read = df['Reading']
    prediction  = model.predict([[math,read]])
    return np.array_str(prediction)

if __name__ == '__main__':
    app.run(port=5000,debug=True)