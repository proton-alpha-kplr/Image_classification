import os
#Le module os (système d'exploitation) fournit des fonctions et des variables
#qui vous permettent d'interagir avec le système d'exploitation dans lequel votre code Python s'exécute.

import numpy as np
#Numpy bibliothèque de Manipulation de matrice

# Keras
from keras.models import load_model
import keras.utils as image
#Keras est une bibliothèque de réseaux de neurones de haut niveau pour Python./deep learning

# Flask utils
from flask import Flask, request, render_template
#Flask est un framework web pour Python qui vous permet de développer rapidement des applications web.

from werkzeug.utils import secure_filename
#Werkzeug fournit des outils pour gérer les requêtes HTTP et les réponses

# Image loader
import cv2
#une bibliothèque de vision par ordinateur open source pour Python charger et enregistrer des images et
#des vidéos effectuer des opérations de traitement d'image ...

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './static/model/Model.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):#chemin vers le fichier image à classifier , modèle de machine learning qui sera utilisé pour classer l'image
    img = image.load_img(img_path, target_size=(28, 28)) #lire le fichier image et le redimensionne à une image
    img_array = np.asarray(img)#convertir l'objet img donné en un tableau NumPy
    x = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)#convertir l'image en niveaux de gris
    result = int(img_array[0][0][0])#vérifie la valeur du premier pixel de l'image
    print(result)
    if result > 128:
      img = cv2.bitwise_not(x)# la fonction inverse l'image à l'aide d'une opération NOT binaire
    else:
      img = x
    img = img/255#normalisée l'image
    img = (np.expand_dims(img,0)) #emodeléer l'image à l'aide de np.expand_dims pour ajouter une dimension supplémentaire à la matrice d'image

    preds =  model.predict(img)#classer l'image
    print(preds)
    return preds#renvoie la prédiction en sortie


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__) #basepath:chemin de base du fichier actuel
        file_path = os.path.join(
            basepath, './static/upload', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        predicted_label = np.argmax(preds)
        result = class_names[predicted_label]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
