import tf_keras as keras
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from google import genai

st.title('♻️Green Campus Waste Mismanagement System♻️')

client = genai.Client(api_key = st.secrets['GEMINI_API_KEY'])
model = keras.models.load_model('keras_model.h5')
image = st.camera_input('Take a photo of waste properly.')

if image:
    img = Image.open(image).resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)

    with st.spinner('Analysing waste...'):
        prediction = model.predict(img_array)

    labels = ['Paper','Plastic','Organic','Hazardous']
    emojis = {
    'Paper': '📄',
    'Plastic': '🥤',
    'Organic': '🍃',
    'Hazardous': '☠️'}
    class_index = np.argmax(prediction)
    st.success(f'This looks like: {labels[class_index]} {emojis[labels[class_index]]}')

    with st.spinner('Thinking...'):
        response = client.models.generate_content(
            model = 'gemini-3-flash-preview',
            contents = f'Tell me which bin this type of waste belongs to in one sentence(red/green/blue):{labels[class_index]} also add bin colour and also give a 20 word environmental impact fact about the item'
        )
    st.write(response.text)