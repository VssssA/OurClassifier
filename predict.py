import os

import click
import keras
import numpy as np
import tensorflow as tf
@click.command()
@click.argument('image_path', type=click.Path(exists=True))
def main(image_path):
    resnet50 = keras.models.load_model('/content/drive/MyDrive/TrainModel10')
    make_prediction(resnet50, image_path)

"""
Make Prediction [using pre-trained model]
"""
def make_prediction(model, path=None):
    if path is None:
        raise UserWarning('Image path should not be None!')

    categories = os.listdir('/content/drive/MyDrive/testdata10models/test')

    # preprocessing
    img = keras.preprocessing.image.load_img(path, target_size=(64, 64))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.utils.normalize(x)

    # make prediction
    preds = model.predict(x)
    print("Model predicts a "{}" with {:.2f}% probability".format(categories[np.argmax(preds[0])], preds[0][np.argmax(preds)] * 100))

if name == 'main':
    main()
