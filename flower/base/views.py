from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'index.html')

def getPredictions(sepal_length, sepal_width, petal_length, petal_width):
    model = pickle.load(open("d:/lol/Iris/Model/ml_model.sav", "rb"))
    scaler = pickle.load(open("d:/lol/Iris/Model/scaler.sav", "rb"))
    
    prediction = model.predict(scaler.transform([[
        float(sepal_length), 
        float(sepal_width), 
        float(petal_length), 
        float(petal_width)
    ]]))
    
    return prediction[0]

def result(request):
    sepal_length = request.GET['sepal_length']
    sepal_width = request.GET['sepal_width']
    petal_length = request.GET['petal_length']
    petal_width = request.GET['petal_width']
    
    result = getPredictions(sepal_length, sepal_width, petal_length, petal_width)
    
    return render(request, 'result.html', {'result': result})
