import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler  
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

model_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\linear_regression_model.sav'
scaler_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\scaler.sav'
kmeans_model_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\kmeans_model.sav'

loaded_kmeans_model = pickle.load(open(kmeans_model_path, 'rb'))
loaded_lr_model = pickle.load(open(model_path, 'rb'))
loaded_scaler = pickle.load(open(scaler_path, 'rb'))

csv_file_path = r'C:\Users\User\OneDrive - kmutnb.ac.th\Documents\ISProject\ISProject\ML_Model\DatasetML\msleep_sample.csv'  # ระบุ path ของไฟล์ CSV ที่ต้องการ
data = pd.read_csv(csv_file_path)

st.title("📝 Linear Regression and KMeans Clustering")

st.write("🔹 Dataset:")
st.write(data.head())

if 'vore' in data.columns:
    data = pd.get_dummies(data, columns=['vore'], drop_first=True)
    st.write("🔹 ข้อมูลหลัง One-Hot Encoding:")
    st.write(data.head())
else:
    st.write("🔹 คอลัมน์ 'vore' ไม่มีในข้อมูลที่อัปโหลด")

st.write("🔹 เลือกคอลัมน์ที่ต้องการใช้ในการทำนาย (bodywt), (brainwt), (sleep_rem), (sleep_cycle), (awake):")

num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

all_cols = num_cols + cat_cols

selected_cols = st.multiselect("เลือกคอลัมน์:", all_cols)

if selected_cols:

    features = data[selected_cols]
    st.write("🔹 ข้อมูลฟีเจอร์ที่เลือก:")
    st.write(features.head())

    imputer = SimpleImputer(strategy="mean")
    features = imputer.fit_transform(features)

    scaler = StandardScaler()  
    features_scaled = scaler.fit_transform(features)

    predictions_lr = loaded_lr_model.predict(features_scaled)
    st.write("🔹 ผลการทำนายจากโมเดล Linear Regression:")
    st.write(predictions_lr)

    clusters = loaded_kmeans_model.predict(features_scaled)
    st.write("🔹 ผลการทำนายจากโมเดล KMeans Clustering:")
    st.write(clusters)

    data['Linear Regression Prediction'] = predictions_lr
    data['KMeans Cluster'] = clusters
    st.write("🔹 ข้อมูลพร้อมผลการทำนาย:")
    st.write(data)

    st.write("🔹 การกระจายของข้อมูลในคลัสเตอร์:")
    plt.figure(figsize=(8, 6))

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
