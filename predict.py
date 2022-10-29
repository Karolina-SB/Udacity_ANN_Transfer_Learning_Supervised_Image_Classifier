import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import tensorflow_hub as hub
import seaborn as sns
from PIL import Image
import time
import operator
import os


# python predict.py --image_path './test_images/wild_pansy.jpg'  --model_path './saved_models/2022-07-27_20-16.h5' --top_k 5 --class_map './label_map.json'

warnings.filterwarnings('ignore')
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
tfds.disable_progress_bar()



def process_image(image): 
    
    image_size = 224
    
    img = tf.convert_to_tensor(image)
    img = tf.image.resize(img, (image_size, image_size))
    img /= 255.0
    
    img = img.numpy()
    return img


def predict(image_path, model, top_k, class_names):
    
    with Image.open(image_path) as img:
        img = np.asarray(img, dtype=np.float32)
        
    preprocessed_image = process_image(img )
    #print((processed_image).shape)
    preprocessed_input_with_batch_dimention = np.expand_dims(preprocessed_image, axis=0)


    reloaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer})
    
    #---------------------------------------------------------------------------------------------------
    # .predict() returns list of probabilities for each consecutive class for the image I'm checking.
    # Value prediction[0] in the below list involves ____class '1'____, because class labels starts from '1'
    # and ends on "102" (not 0-101, like it would be intuitive while operating on python list numeration...)
    # So I need to add +1 to every class number, so it match the real numeration.
    prediction = reloaded_model.predict(preprocessed_input_with_batch_dimention)
    predictions = np.array(prediction.squeeze()) # unpacking and transforming to np.array
    
    indexes = np.argpartition(predictions, -top_k)[-top_k:] # fast way to get N indexes of max values from array
    probabilities = np.take(predictions, indexes)
    
    best_probabilities = dict()  # dictionary of indexex - i.e "positions" of class labels and corresponding propabilities

    for i, x in zip(indexes, probabilities):
        best_probabilities[i] = x
            
    sorted_best_probabilities = dict( sorted(best_probabilities.items(), key=operator.itemgetter(1),reverse=True))
    #print('Dictionary in descending order by value : ', sorted_best_probabilities)
    
    probs = list(sorted_best_probabilities.values())
    classes = list(sorted_best_probabilities.keys())
    actual_classes = [x+1 for x in classes]   # here is the +1 adding mentioned before
    
    #print(probs)
    #print(actual_classes)
    
    print("_______________________________________________________________________________________________")
    print("Probabilities and Class numbers:") 
    for c,p in zip(actual_classes, probs):
        print("class: {} ({}), probability {:.3%}\n".format(c, class_names[str(c)], p ))
        
    return probs, actual_classes





def main(): 



    print('Enter Your Arguments')
    
    parser = argparse.ArgumentParser(description='Please, provide arguments.')

    parser.add_argument ('--image_path', default='./test_images/cautleya_spicata.jpg', help = 'Path to image you want to classify.', type = str)
    parser.add_argument ('--model_path', default='./saved_models/2022-07-27_20-16.h5', help='Path to pretrained machine learning model.', type=str)
    parser.add_argument ('--top_k', default = 5, help = 'Number of top most likely classes predicted for your image.', type = int)
    parser.add_argument ('--class_map' , default = 'label_map.json', help = 'json file providing mapping of numerical categories to real flower names.', type = str)

    args = parser.parse_args()
    print(args)
    print('arg1:', args.image_path)
    print('arg2:', args.model_path)
    print('arg3:', args.top_k)
    print('arg3:', args.class_map)

    
    image_path = args.image_path
    
    #model_path = tf.keras.models.load_model(args.model_path ,custom_objects={'KerasLayer':hub.KerasLayer} ,compile=False)
    model_path = args.model_path



    top_k = args.top_k
    if top_k is None: 
        top_k = 5
    
    with open(args.class_map, 'r') as f:
        class_names = json.load(f)


    probs, classes = predict(image_path, model_path, top_k, class_names)

    
    with Image.open(image_path) as img:
        img = np.asarray(img, dtype=np.float32)
        
    img = process_image(img)
    
    labels = []
    for i in classes:
        labels.append(class_names[str(i)])
    
    
    fig = plt.figure(figsize = (8,4))  
    plt.grid(False)


    plt.subplot(1, 2, 1)
    plt.grid(False)
    plt.imshow(img)


    plt.subplot(1, 2, 2)
    plt.barh(labels, probs)
    plt.xlabel('Probability')
    plt.title('Class probability')
    fig.tight_layout(w_pad=4, h_pad=4)
    plt.grid(False)

    plt.show()


if __name__ == "__main__":

    main()