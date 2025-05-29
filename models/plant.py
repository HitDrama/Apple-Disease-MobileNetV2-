import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PlantDiseaseModel:
    def __init__(self, model_path='models/plant_disease_model.h5'):
        self.model_path = model_path
        self.model = self.load_model()
        self.class_names = [
            'Táo - Bệnh ghẻ lá',
            'Táo - Thối đen',
            'Táo - Gỉ sắt tuyết tùng',
            'Táo - Khỏe mạnh'
        ]
        self.img_size = (224, 224)  # Kích thước ảnh đầu vào của mô hình

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        print(f"[INFO] Loading model from {self.model_path}")
        return load_model(self.model_path)

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
        img_array = img_array / 255.0  # Chuẩn hóa ảnh
        return img_array

    def predict(self, img_path):
        img_array = self.preprocess_image(img_path)
        prediction = self.model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        predicted_class = self.class_names[class_index]
        return predicted_class, confidence
