import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler  
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# โหลดโมเดลและ scaler
model_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\linear_regression_model.sav'
scaler_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\scaler.sav'
kmeans_model_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\kmeans_model.sav'

# โหลดโมเดล
loaded_kmeans_model = pickle.load(open(kmeans_model_path, 'rb'))
loaded_lr_model = pickle.load(open(model_path, 'rb'))
loaded_scaler = pickle.load(open(scaler_path, 'rb'))

# แทนที่การอัปโหลดไฟล์ CSV ด้วยการโหลดไฟล์ CSV โดยตรง
csv_file_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\DatasetML\msleep_sample.csv'  # ระบุ path ของไฟล์ CSV ที่ต้องการ
data = pd.read_csv(csv_file_path)

# Streamlit UI
st.title("📝 Linear Regression and KMeans Clustering")

# แสดงข้อมูลที่อัปโหลด
st.write("🔹 Dataset:")
st.write(data.head())

# ตรวจสอบว่ามีคอลัมน์ 'vore' หรือไม่
if 'vore' in data.columns:
    # ทำ One-Hot Encoding ให้กับคอลัมน์ 'vore' หากมี
    data = pd.get_dummies(data, columns=['vore'], drop_first=True)
    st.write("🔹 ข้อมูลหลัง One-Hot Encoding:")
    st.write(data.head())
else:
    st.write("🔹 คอลัมน์ 'vore' ไม่มีในข้อมูลที่อัปโหลด")

# ให้ผู้ใช้เลือกคอลัมน์ที่ต้องการใช้ในการทำนาย
st.write("🔹 เลือกคอลัมน์ที่ต้องการใช้ในการทำนาย (bodywt), (brainwt), (sleep_rem), (sleep_cycle), (awake):")

# คอลัมน์ตัวเลข (numeric columns)
num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# คอลัมน์ประเภทตัวอักษร (categorical columns) ถ้ามี
cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# รวมคอลัมน์ตัวเลขและคอลัมน์ประเภทตัวอักษร
all_cols = num_cols + cat_cols

# ให้ผู้ใช้เลือกคอลัมน์
selected_cols = st.multiselect("เลือกคอลัมน์:", all_cols)

if selected_cols:
    # ตรวจสอบและเลือกข้อมูลฟีเจอร์จากคอลัมน์ที่ผู้ใช้เลือก
    features = data[selected_cols]
    st.write("🔹 ข้อมูลฟีเจอร์ที่เลือก:")
    st.write(features.head())

    # เติมค่า Missing (ถ้ามี)
    imputer = SimpleImputer(strategy="mean")
    features = imputer.fit_transform(features)

    # แปลงข้อมูล (Scale) ด้วย Scaler
    scaler = StandardScaler()  # หรือใช้ scaler ที่คุณโหลดมา
    features_scaled = scaler.fit_transform(features)

    # ทำนายด้วยโมเดล Linear Regression
    predictions_lr = loaded_lr_model.predict(features_scaled)
    st.write("🔹 ผลการทำนายจากโมเดล Linear Regression:")
    st.write(predictions_lr)

    # ทำนายด้วยโมเดล KMeans Clustering
    clusters = loaded_kmeans_model.predict(features_scaled)
    st.write("🔹 ผลการทำนายจากโมเดล KMeans Clustering:")
    st.write(clusters)

    # แสดงผลลัพธ์ใน DataFrame
    data['Linear Regression Prediction'] = predictions_lr
    data['KMeans Cluster'] = clusters
    st.write("🔹 ข้อมูลพร้อมผลการทำนาย:")
    st.write(data)

    # แสดงการกระจายของข้อมูลในคลัสเตอร์
    st.write("🔹 การกระจายของข้อมูลในคลัสเตอร์:")
    plt.figure(figsize=(8, 6))

    # หากมี 2 ฟีเจอร์ที่เลือกมาแสดงในกราฟ
    if len(selected_cols) >= 2:
        plt.scatter(data[selected_cols[0]], data[selected_cols[1]], c=data['KMeans Cluster'], cmap='viridis')
        plt.xlabel(selected_cols[0])
        plt.ylabel(selected_cols[1])
        plt.title('การกระจายข้อมูลในคลัสเตอร์')
        st.pyplot(plt)
    else:
        st.write("กรุณาเลือกอย่างน้อย 2 ฟีเจอร์เพื่อดูกราฟการกระจายข้อมูล.")
    
else:
    st.write("กรุณาเลือกคอลัมน์เพื่อใช้ในการทำนาย")
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

# Add GitHub link at the bottom of the sidebar
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

# Reference section in the sidebar with improved styling
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
