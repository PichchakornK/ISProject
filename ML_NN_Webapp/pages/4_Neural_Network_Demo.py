import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

# ตั้งชื่อเว็บแอป
st.title("🧠 Neural Network Model Demo for Image Classification")

# อัปโหลดไฟล์ภาพจากเครื่อง
uploaded_image = st.file_uploader("อัปโหลดไฟล์ภาพ (Dog or Cat)", type=["jpg", "png"])

if uploaded_image is not None:
    # เปิดภาพที่อัปโหลด
    img = Image.open(uploaded_image)

    # แปลงภาพเป็น RGB (หากภาพเป็น grayscale หรือมีช่อง alpha)
    img = img.convert("RGB")

    # ปรับขนาดภาพให้ตรงกับขนาดที่โมเดลต้องการ (100x100)
    img = img.resize((100, 100))

    # แปลงภาพเป็น numpy array และ normalize ค่า
    img_array = np.array(img) / 255.0

    # ปรับรูปร่างของภาพให้เป็น (1, 100, 100, 3) เพื่อให้ตรงกับที่โมเดลต้องการ
    img_array = img_array.reshape((1, 100, 100, 3))

    # แสดงภาพที่อัปโหลดในเว็บแอป
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # โหลดโมเดล Neural Network ที่บันทึกไว้
    model_path = "C:/Users/User/OneDrive - kmutnb.ac.th/Documents/ISProject/ISProject/NN_Model/finalized_model.sav"
    model = pickle.load(open(model_path, 'rb'))

    # ทำนายผลลัพธ์จากโมเดล Neural Network
    if st.button("🔍 ทำนายผลลัพธ์"):
        predictions = model.predict(img_array)  # ทำนายผลลัพธ์
        prediction_class = "Dog" if predictions[0][0] < 0.5 else "Cat"  # สมมุติว่า 0 = Dog, 1 = Cat
        st.write(f"🔹 ผลลัพธ์การทำนาย: {prediction_class}")
