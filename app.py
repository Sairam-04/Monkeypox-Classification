
from flask import Flask, render_template, request
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array


from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

train_Loss = {'ResNet50V2': 0.0581667460501194, 'Mobilenet': 0.035449396818876266, 'CNN': 0.05624306946992874, 'VGG16' : 0.6895  }
test_Loss = {'ResNet50V2': 0.08349521458148956, 'Mobilenet': 0.5040205717086792, 'CNN': 1.8795151710510254, 'VGG16' : 0.6797 }
train_Accuracy = {'ResNet50V2': 0.9878618121147156, 'Mobilenet': 0.9920634627342224, 'CNN': 0.9859943985939026, 'VGG16' : 0.5425 }
test_Accuracy = {'ResNet50V2': 0.9444444179534912, 'Mobilenet': 0.8383233547210693, 'CNN': 0.538922131061554, 'VGG16' : 0.5988}

mobilenet_model = load_model("./Models/mobilenetmodelfinal.h5")
resnet_model = load_model("./Models/ResNet50V2final.h5")
cnn_model = load_model("./Models/CnnModelfinal.h5")
vgg_model = load_model("./Models/vgg16.h5")

def mobilenet_predict(filepath):
    global mpred
    img = plt.imread(filepath)
    temp_img = img
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    img = img/255.0

    mobilenet_prediction = mobilenet_model.predict(img)
    print(mobilenet_prediction)
    mpred = mobilenet_prediction
    mobilenet_prediction = np.argmax(mobilenet_prediction)
    if mobilenet_prediction==1:
        mobilenet_prediction = "Non Monkeypox"
        return 1
    else:
        mobilenet_prediction = "Monkeypox"
        return 0

def vgg_predict(filepath):
    global vpred
    img = plt.imread(filepath)
    temp_img = img
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    img = img/255.0

    vgg_prediction = vgg_model.predict(img)
    vpred = vgg_prediction
    vgg_prediction = np.argmax(vgg_prediction)
    if vgg_prediction==1:
        vgg_prediction = "Non Monkeypox"
        return 1
    else:
        vgg_prediction = "Monkeypox"
        return 0


def resnet_predict(filepath):
    global rpred
    img = plt.imread(filepath)
    temp_img = img
    img = cv2.resize(img,(256,256))
    img = img.reshape(1,256,256,3)
    img = img/255.0
    resnet_prediction = resnet_model.predict(img)
    rpred = resnet_prediction
    resnet_prediction = np.argmax(resnet_prediction)
    if resnet_prediction==1:
        resnet_prediction = "Non Monkeypox"
        return 1
    else:
        resnet_prediction = "Monkeypox"
        return 0  
  

    
def cnn_predict(filepath):
    global cpred
    img = plt.imread(filepath)
    temp_img = img
    img = cv2.resize(img,(256,256))
    img = img.reshape(1,256,256,3)
    img = img/255.0

    cnn_prediction = cnn_model.predict(img)
    cpred = cnn_prediction
    cnn_prediction = np.argmax(cnn_prediction)
    if cnn_prediction==1:
        cnn_prediction = "Non Monkeypox"
        return 1
    else:
        cnn_prediction = "Monkeypox"
        return 0


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/test')
def test():
    return render_template("index.html")

@app.route('/aboutus')
def aboutus():
    return render_template("aboutus.html")

@app.route('/upload', methods=['POST'])
def upload_img():
    global path
    img = request.files['image']
    img_name = img.filename
    
    path = os.path.join('static/images/',img_name)
    img.save(path)    

    return render_template("index.html",img_src=path)

@app.route('/resnet')
def resnet():
    result_resnet = ""
    prediction_resnet = resnet_predict(path)
    if(prediction_resnet == 0):
        result_resnet+="Monkeypox"
    else:
        result_resnet+="Non Monkeypox"

    return render_template('resnet.html',prob_resnet = rpred, result_resnet = result_resnet,train_acc_r = train_Accuracy["ResNet50V2"],test_acc_r = test_Accuracy["ResNet50V2"],train_loss_r = train_Loss["ResNet50V2"],test_loss_r = test_Loss['ResNet50V2'],img_src = path)

@app.route('/mobilenet')
def mobilenet():
    result_mobilenet = ""
    prediction_mobilenet = mobilenet_predict(path)
    if(prediction_mobilenet == 0):
        result_mobilenet+="Monkeypox"
    else:
        result_mobilenet+="Non Monkeypox"

    return render_template('mobilenet.html',prob_mobilenet = mpred, result_mobilenet = result_mobilenet,train_acc_m = train_Accuracy["Mobilenet"],test_acc_m = test_Accuracy["Mobilenet"],train_loss_m = train_Loss["Mobilenet"],test_loss_m = test_Loss['Mobilenet'],img_src = path)

@app.route('/vgg')
def vgg():
    result_vgg = ""
    prediction_vgg = vgg_predict(path)
    if(prediction_vgg == 0):
        result_vgg+="Monkeypox"
    else:
        result_vgg+="Non Monkeypox"

    return render_template('vgg.html', result_vgg = result_vgg,train_acc_v = train_Accuracy["VGG16"],test_acc_v = test_Accuracy["VGG16"],train_loss_v = train_Loss["VGG16"],test_loss_v = test_Loss["VGG16"],prob_vgg = vpred ,img_src = path)


@app.route('/cnn')
def cnn():
    result_cnn = ""
    prediction_cnn = cnn_predict(path)
    if(prediction_cnn == 0):
        result_cnn+="Monkeypox"
    else:
        result_cnn+="Non Monkeypox"

    return render_template('cnn.html', result_cnn = result_cnn,train_acc_c = train_Accuracy["CNN"],test_acc_c = test_Accuracy["CNN"],train_loss_c = train_Loss["CNN"],test_loss_c = test_Loss["CNN"],prob_cnn = cpred ,img_src = path)


@app.route('/summary')
def summary():   
    prediction1 = mobilenet_predict(path)
    prediction2 = resnet_predict(path)
    prediction3 = cnn_predict(path)
    prediction4 = vgg_predict(path)
    result = []
    temp = []
    temp.append("Mobilenet")
    pred1 = "Mobilenet Model : "
    if prediction1 == 0:
        pred1 += " Monkeypox "
        result.append(["Mobilenet","True","False"])
    else :
        pred1 += " Non Monkeypox "
        result.append(["Mobilenet","False","True"])

    pred2 = "Resnet Model : " 
    if prediction2 == 0:
        pred2 += " Monkeypox"
        result.append(["Resnet","True","False"])
    else :
        pred2 += " Non Monkeypox"
        result.append(["Resnet","False","True"])

    pred3 = "CNN Model : " + str(cpred)
    if prediction3 == 0:
        pred3 += " Monkeypox"
        result.append(["CNN","True","False"])
    else :
        pred3 += " Non Monkeypox"
        result.append(["CNN","False","True"])

    pred4 = "VGG Model : " + str(vpred)
    if prediction4 == 0:
        pred4 += " Monkeypox"
        result.append(["VGG","True","False"])
    else :
        pred4 += " Non Monkeypox"
        result.append(["VGG","False","True"])


    return render_template("summary.html", result = result,  img_src = path)


if __name__ == "__main__":
    app.run()
