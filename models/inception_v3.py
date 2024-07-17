import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2


class InceptionV3Model:
    def __init__(self):
        self.model = InceptionV3(weights='imagenet')

    def predict(self, frames):
        predictions = []
        for frame in frames:
            image = cv2.resize(frame, (299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            preds = self.model.predict(image)
            decoded_preds = decode_predictions(preds, top=3)[0]
            predictions.append(decoded_preds)
        return predictions

    def search_object(self, predictions, query):
        results = []
        for i, preds in enumerate(predictions):
            for _, label, _ in preds:
                if query.lower() in label.lower():
                    results.append((i, preds))
        return results
