import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os

st.title("🧠 Neural Network Model Demo for Image Classification")

uploaded_image = st.file_uploader("อัปโหลดไฟล์ภาพ (Dog or Cat)", type=["jpg", "png"])

if uploaded_image is not None:
    
    img = Image.open(uploaded_image)
    img = img.convert("RGB")
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 100, 100, 3))
 
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ใช้ relative path เพื่อระบุเส้นทางของไฟล์โมเดล
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "..", "..", "NN_Model", "finalized_model.sav")

    # ตรวจสอบว่าไฟล์มีอยู่จริงก่อนโหลด
    try:
        if os.path.exists(model_path):
            model = pickle.load(open(model_path, 'rb'))
        else:
            st.error(f"ไม่พบไฟล์ที่ตำแหน่ง: {model_path}")
            model = None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        model = None

    if model is not None and st.button("🔍 ทำนายผลลัพธ์"):
        predictions = model.predict(img_array)
        prediction_class = "Dog" if predictions[0][0] < 0.5 else "Cat"
        st.write(f"🔹 ผลลัพธ์การทำนาย: {prediction_class}")
