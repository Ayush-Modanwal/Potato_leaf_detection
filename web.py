import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

file_id="1THXw7f1GKhnmtLRAHrVbWEyvDrYuGMpx"
url='https://drive.google.com/uc?id=1THXw7f1GKhnmtLRAHrVbWEyvDrYuGMpx'
model_path="trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google drive...")
    gdown.download(url,model_path,quiet=False)

def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_plant_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)
    


st.sidebar.title("Plant Disease Detection For Sustainable Agriculture")
app_mode=st.sidebar.selectbox('select page',['HOME','DISEASE RECOGNITION'])


img=Image.open('Disease.jpg')
st.image(img, use_column_width=True)

if(app_mode=='HOME'):
    st.markdown("<h1 style= 'text-align:center;'>Plant Disease Detection For Sustainable Agriculture</h1>",unsafe_allow_html=True)
    
    
elif(app_mode=='DISEASE RECOGNITION'):
    st.header('Plant Disease Detection For Sustainable Agriculture')
    
    
test_image=st.file_uploader('choose an image:')
if(st.button('show image')):
    st.image(test_image,width=4,use_column_width=True)
    
if(st.button('Predict')):
    st.snow()
    st.write('our prediction') 
    result_index = model_prediction(test_image)
    
    class_name=['Potato__Early_blight','Potato__Late_blight','Potato__healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))