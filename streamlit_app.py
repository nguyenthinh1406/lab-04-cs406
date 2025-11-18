import streamlit as st
import cv2
import numpy as np
import keras
from keras import layers, models
from keras.models import load_model
import tensorflow as tf
import time
import pandas as pd
from PIL import Image
import timm
import torch.nn as nn
import torch
import torchvision.transforms as T

from keras.applications import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocessor
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocessor
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocessor
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocessor

augment = keras.Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(factor = 0.2),
    layers.RandomContrast(factor = 0.5)
])

vit_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def base_model(model, preprocessor, input_shape, num_classes):
    model.trainable = False
    inputs = keras.Input(shape = input_shape)
    x = augment(inputs)
    x = preprocessor(x)
    x = model(x, training = False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias = False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation = 'softmax')(x)
   
    model = keras.Model(inputs, outputs)

    return model

@st.cache_resource
def load_all_models():
    vgg_model = keras.applications.VGG19(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape
    )
    vgg_model = base_model(vgg_model, vgg_preprocessor, input_shape, num_classes)
    vgg_model.load_weights('vgg_weights.weights.h5')


    resnet_model = keras.applications.ResNet50(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape

    )
    resnet_model = base_model(resnet_model, resnet_preprocessor, input_shape, num_classes)
    resnet_model.load_weights('resnet_weights.weights.h5')


    mobile_model = keras.applications.MobileNetV2(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape

    )
    mobile_model = base_model(mobile_model, mobile_preprocessor, input_shape, num_classes)
    mobile_model.load_weights('mobile_weights.weights.h5')


    efficient_model = EfficientNetB0(
        weights = 'imagenet',
        include_top = False,
        input_shape = input_shape

    )
    efficient_model = base_model(efficient_model, efficient_preprocessor, input_shape, num_classes)
    efficient_model.load_weights('efficient_weights.weights.h5')


    vit_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained = True,
        num_classes = 6
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model.load_state_dict(torch.load('vit_weights.pth'), map_location = device)
    vit_model.to(device)
    
    return vgg_model, resnet_model, mobile_model, efficient_model, vit_model

batch_size = 32
input_shape = (224, 224, 3)
num_classes = 6

try:
    model_vgg, model_resnet, model_mobile, model_efficient, model_vit = load_all_models()
    st.success("Đã tải tất cả 5 model!")
except Exception as e:
    st.error(f"Lỗi khi tải model: {e}")
    st.stop()

upload_im = st.file_uploader("Chọn ảnh của bạn", type=["png", "jpg", "jpeg"])

if upload_im is not None:
    img_original = np.asarray(bytearray(upload_im.read()), dtype = np.uint8)
    img_original = cv2.cvtColor(cv2.imdecode(img_original, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_original)
    st.image(img_original, caption = 'Ảnh đã tải lên.', use_column_width = True)

    model_list = [model_vgg, model_resnet, model_mobile, model_efficient, model_vit]
    model_names = ['VGG19', 'ResNet50', 'MobileNetV2', 'EfficientNetB0', 'ViTNetB16']
    preprocess_funcs = [vgg_preprocessor, resnet_preprocessor, mobile_preprocessor, efficient_preprocessor, None]
    class_name = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    predicted_class = []
    confidence_score = []
    inference_time = []

    for model, name in zip(model_list, model_names):
        st.write(f"--- \nĐang xử lý với model: **{name}**")
        if name == 'ViTNetB16':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            start = time.time()
            img_tensor = vit_transform(img_pil).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                logits = model(img_tensor)

            probabilities = torch.nn.functional.softmax(logits, dim=1)
            conf = probabilities.max().item() * 100
            pred_idx = probabilities.argmax().item()
            pred_class = class_name[pred_idx]
            end = time.time()

            inference_time.append(end - start)
            predicted_class.append(pred_class)
            confidence_score.append(conf)
        else:
            target_size = (224, 224)
            img_resized = cv2.resize(img_original, target_size)
            img_batch = np.expand_dims(img_resized, axis=0)

            start = time.time()
            pred = model.predict(img_batch)
            end = time.time()

            inference_time.append(end - start)
            predicted_class.append(class_name[np.argmax(pred)])
            confidence_score.append(np.max(pred) * 100)

    df = pd.DataFrame({
        'Model': model_names,
        'Predicted class': predicted_class,
        'Confidence(%)': confidence_score,
        'Inference time(s)': inference_time
    })

    st.dataframe(df)
    st.bar_chart(df, x = 'Model', y = 'Confidence(%)')
    st.bar_chart(df, x = 'Model', y = 'Inference time(s)')
