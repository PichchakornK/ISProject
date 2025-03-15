import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

st.title("🧠 Neural Network Model Demo for Image Classification")

uploaded_image = st.file_uploader("อัปโหลดไฟล์ภาพ (Dog or Cat)", type=["jpg", "png"])

if uploaded_image is not None:
    
    img = Image.open(uploaded_image)

    img = img.convert("RGB")

    img = img.resize((100, 100))

    img_array = np.array(img) / 255.0

    img_array = img_array.reshape((1, 100, 100, 3))
 
    st.image(img, caption="Uploaded Image", use_container_width=True)

    model_path = "C:/Users/User/OneDrive - kmutnb.ac.th/Documents/ISProject/ISProject/NN_Model/finalized_model.sav"
    model = pickle.load(open(model_path, 'rb'))

    if st.button("🔍 ทำนายผลลัพธ์"):
        predictions = model.predict(img_array) 
        prediction_class = "Dog" if predictions[0][0] < 0.5 else "Cat"  
        st.write(f"🔹 ผลลัพธ์การทำนาย: {prediction_class}")
st.sidebar.markdown(
    """
    <style>
    .sidebar-footer {
        position: absolute;
        width: 100%;
        text-align: center;
    }
    .sidebar-footer a {
        font-size: 16px;
        color: white;
        text-decoration: none;
    }
    .sidebar-footer img {
        vertical-align: middle;
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <a href="https://github.com/PichchakornK/ISProject.git" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20">
            6404062663215 Pichchakorn Kongmai
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer" style="margin-top: 20px; padding: 10px; background-color: rgba(0, 0, 0, 0.5); border-radius: 10px;">
        <h3 style="color:white; text-align:center; margin-bottom: 15px;">📚 References</h3>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Dataset for ML: 
                <a href="https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv" target="_blank" style="color: #3498db; text-decoration: none;">ML Dataset Link</a>
            </p>
        </div>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Dataset for NN: 
                <a href="https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD" target="_blank" style="color: #3498db; text-decoration: none;">NN Dataset Link</a>
            </p>
        </div>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Machine Learning Tutorial: 
                <a href="https://www.youtube.com/watch?v=T2yT5vt1NaQ&list=PLoTScYm9O0GH_3VrwwnQafwWQ6ibKnEtU&index=6" target="_blank" style="color: #3498db; text-decoration: none;">ML Video Link</a>
            </p>
        </div>
        <div style="margin-bottom: 10px;">
            <p style="color:white; font-size: 14px;">Neural Network Tutorial: 
                <a href="https://github.com/Coding-Lane/Image-Classification-CNN-Keras.git" target="_blank" style="color: #3498db; text-decoration: none;">NN GitHub Link</a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
