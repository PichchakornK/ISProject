import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler  
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt

# กำหนด path ที่ถูกต้อง
base_path = os.path.dirname(os.path.abspath(__file__))

# Path ของโมเดลและ dataset
model_path = os.path.abspath(os.path.join(base_path, '..', '..', 'ML_Model', 'linear_regression_model.sav'))
scaler_path = os.path.abspath(os.path.join(base_path, '..', '..', 'ML_Model', 'scaler.sav'))
kmeans_model_path = os.path.abspath(os.path.join(base_path, '..', '..', 'ML_Model', 'kmeans_model.sav'))
csv_file_path = os.path.abspath(os.path.join(base_path, '..', '..', 'ML_Model', 'DatasetML', 'msleep_sample.csv'))

# ตรวจสอบไฟล์ก่อนโหลด
def load_pickle_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error(f"❌ File not found: {path}")
        return None

# โหลดโมเดล
loaded_kmeans_model = load_pickle_model(kmeans_model_path)
loaded_lr_model = load_pickle_model(model_path)
loaded_scaler = load_pickle_model(scaler_path)

# อ่านไฟล์ CSV
if os.path.exists(csv_file_path):
    data = pd.read_csv(csv_file_path)
else:
    st.error(f"❌ File not found: {csv_file_path}")
    data = None

st.title("📝 Linear Regression and KMeans Clustering")

if data is not None:
    st.write("🔹 Dataset:")
    st.write(data.head())

    # แปลงคอลัมน์ 'vore' เป็น One-Hot Encoding (ถ้ามี)
    if 'vore' in data.columns:
        data = pd.get_dummies(data, columns=['vore'], drop_first=True)
        st.write("🔹 ข้อมูลหลัง One-Hot Encoding:")
        st.write(data.head())

    # แสดงตัวเลือกคอลัมน์ให้ผู้ใช้เลือก
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = num_cols + cat_cols

    selected_cols = st.multiselect("เลือกคอลัมน์:", all_cols)

    if selected_cols:
        features = data[selected_cols]
        st.write("🔹 ข้อมูลฟีเจอร์ที่เลือก:")
        st.write(features.head())

        # จัดการ Missing Values
        imputer = SimpleImputer(strategy="mean")
        features_imputed = imputer.fit_transform(features)

        # Standard Scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        # ทำนายผลด้วย Linear Regression
        if loaded_lr_model:
            predictions_lr = loaded_lr_model.predict(features_scaled)
            st.write("🔹 ผลการทำนายจากโมเดล Linear Regression:")
            st.write(predictions_lr)

        # ทำนายผลด้วย KMeans
        if loaded_kmeans_model:
            clusters = loaded_kmeans_model.predict(features_scaled)
            st.write("🔹 ผลการทำนายจากโมเดล KMeans Clustering:")
            st.write(clusters)

            # เพิ่มผลลัพธ์ลงใน DataFrame
            data['Linear Regression Prediction'] = predictions_lr if loaded_lr_model else None
            data['KMeans Cluster'] = clusters
            st.write("🔹 ข้อมูลพร้อมผลการทำนาย:")
            st.write(data)

            # แสดงกราฟ scatter plot ถ้ามีอย่างน้อย 2 ฟีเจอร์
            if len(selected_cols) >= 2:
                plt.figure(figsize=(8, 6))
                plt.scatter(data[selected_cols[0]], data[selected_cols[1]], c=data['KMeans Cluster'], cmap='viridis')
                plt.xlabel(selected_cols[0])
                plt.ylabel(selected_cols[1])
                plt.title('การกระจายข้อมูลในคลัสเตอร์')
                st.pyplot(plt)
            else:
                st.write("กรุณาเลือกอย่างน้อย 2 ฟีเจอร์เพื่อดูกราฟการกระจายข้อมูล.")
    else:
        st.write("กรุณาเลือกคอลัมน์เพื่อใช้ในการทำนาย")

