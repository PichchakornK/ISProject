import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML & NN Models Information", layout="wide")

# ตั้งชื่อหน้า
st.title("Model Development")
st.divider()
# ใช้ st.radio สำหรับเลือกหัวข้อ
option = st.radio(
    "🔍 **เลือกโมเดลที่ต้องการดูข้อมูล:**",
    ("🤖 Machine Learning (ML)", "🧠 Neural Network (NN)"),
    horizontal=True
)
st.divider()
if option == "🤖 Machine Learning (ML)":
        st.header("ขั้นตอนการพัฒนา Machine Learning (ML)")
        st.write("1.ติดตั้งและเรียกใช้ไลบรารีที่จำเป็น")
        st.code("""
        import import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from google.colab import files, drive
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import pickle
        import cloudpickle as cp
            """, language='python')
        st.write("2.เชื่อมต่อ Google Drive กับ Google Colab เพื่อให้สามารถเข้าถึงไฟล์ใน Drive จาก Colab ได้")
        st.code("""from google.colab import drive
drive.mount('/content/drive')""")
        st.write("3.อัปโหลด Dataset มาจาก GitHub, สุ่มเลือก 50 แถว, แล้วบันทึกไฟล์ CSV ไปยัง Google Drive ที่ตำแหน่งที่กำหนด")
        st.code("""df = pd.read_csv('https://github.com/prasertcbs/tutorial/raw/master/msleep.csv')
df = df.sample(50, random_state=123)

# กำหนด path สำหรับบันทึกไฟล์ใน Google Drive
file_path = '/content/drive/MyDrive/IS_Final_Project/ML_Model/DatasetML/msleep_sample.csv'

# บันทึก DataFrame เป็นไฟล์ CSV ใน Google Drive
df.to_csv(file_path, index=False)

print(f"File saved to: {file_path}")""")
        st.write("4.ตรวจสอบ missing values ใน DataFrame")
        st.code("""print("Missing Values: ", df.isnull().sum())""")
        st.write("5.เติมค่า missing values ในคอลัมน์ที่เลือก (num_cols) ด้วยค่าเฉลี่ย (Mean) โดยใช้ SimpleImputer จากไลบรารี sklearn.impute")
        st.code("""num_cols = ['bodywt', 'brainwt', 'sleep_total', 'sleep_rem', 'sleep_cycle', 'awake']
imputer = SimpleImputer(strategy="mean")
df[num_cols] = imputer.fit_transform(df[num_cols])""")
        st.write("6.แปลงคอลัมน์เชิงหมวดหมู่ (categorical) ที่ชื่อว่า vore ให้เป็นค่าตัวเลขด้วยการใช้ One-Hot Encoding โดยใช้ pd.get_dummies()")
        st.code("""cat_cols = ['vore'] if 'vore' in df.columns else []
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)""")
        st.write("7.แสดงความสัมพันธ์ระหว่างคุณลักษณะต่างๆ ด้วย Heatmap ซึ่งช่วยให้เห็นว่าคุณลักษณะใดมีความสัมพันธ์กับเป้าหมาย")
        st.code("""plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show())""")
        st.write("8.แยกข้อมูลออกเป็นตัวแปร X (Features) และ y (Target variable) โดยเลือกคอลัมน์ที่เกี่ยวข้องในการคำนวณ sleep_total")
        st.code("""X = df[['bodywt', 'brainwt', 'sleep_rem', 'sleep_cycle', 'awake']]
y = df['sleep_total']""")
        st.write("9.Scaling ข้อมูล ใช้ StandardScaler เพื่อปรับขนาดข้อมูล (ทำให้ข้อมูลมีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1) ก่อนที่จะฝึกโมเดล")
        st.code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)""")
        st.write("10.แบ่งข้อมูลออกเป็นชุดฝึก (train) และชุดทดสอบ (test) โดยใช้ train_test_split จาก sklearn")
        st.code("""X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)""")
        st.write("11.สร้างโมเดล Linear Regression และฝึกโมเดลด้วยข้อมูลชุดฝึก (X_train และ y_train)")
        st.code("""lr_model = LinearRegression()
lr_model.fit(X_train, y_train)""")
        st.write("12.ใช้ชุดข้อมูลทดสอบ (X_test และ y_test) เพื่อทำนายและประเมินโมเดลด้วย Mean Squared Error (MSE) และ R² Score เพื่อดูความแม่นยำของโมเดล")
        st.code("""#  Evaluate Model
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Squared Error (MSE): {mse:.4f}")
print(f"📈 R² Score (Test Data): {r2:.4f}")

#  ตรวจสอบ Overfitting
train_r2 = lr_model.score(X_train, y_train)
test_r2 = lr_model.score(X_test, y_test)

print(f"🟢 Train R²: {train_r2:.4f}")
print(f"🔵 Test R²: {test_r2:.4f}")
""")
        st.write("13.พล็อตกราฟ ค่าจริง vs ค่าทำนาย โดยใช้ scatter plot แสดงความสัมพันธ์ระหว่างค่าจริง (y_test) กับค่าทำนาย (y_pred) และมีเส้น ideal line เพื่อเปรียบเทียบผลลัพธ์ของโมเดล")
        st.code("""#  พล็อตกราฟค่าจริงและค่าทำนาย
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()""")
        st.write("14.บันทึกและโหลดโมเดลและตัวแปร scaler โดยใช้ pickle เพื่อจัดเก็บและเรียกใช้งานภายหลัง")
        st.code("""
    model_path = '/content/drive/MyDrive/IS_Final_Project/ML_Model/linear_regression_model.sav'
    scaler_path = '/content/drive/MyDrive/IS_Final_Project/ML_Model/scaler.sav'
    pickle.dump(lr_model, open(model_path, 'wb'))
    pickle.dump(scaler, open(scaler_path, 'wb'))

    # Load using pickle
    loaded_lr_model = pickle.load(open(model_path, 'rb'))
    loaded_scaler = pickle.load(open(scaler_path, 'rb'))""")
        st.write("15.ทดสอบโมเดลที่โหลดมาจากไฟล์และคำนวณ ค่า R² (R-squared) ซึ่งใช้วัดความแม่นยำของโมเดล")
        st.code("""result = loaded_lr_model.score(X_test, y_test)
    print(f"Model Accuracy (R² score): {result:.4f}")""")
        st.write("16.ใช้ Elbow Method เพื่อหาค่าที่เหมาะสมที่สุดของจำนวนกลุ่ม (K) สำหรับ K-Means clustering โดยการวิเคราะห์ WCSS")
        st.code("""wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal K")
    plt.show()""")
        st.write("17.สร้างโมเดล K-Means Clustering และฝึกโมเดลด้วยข้อมูลที่ถูกปรับขนาด (scaled data) โดยเลือกจำนวนคลัสเตอร์ที่ดีที่สุด (K=3)")
        st.code("""optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)""")
        st.write("18.บันทึก K-Means Model ด้วย pickle")
        st.code("""kmeans_model_path = '/content/drive/MyDrive/IS_Final_Project/ML_Model/kmeans_model.sav'  # Use the Google Drive path here
pickle.dump(kmeans, open(kmeans_model_path, 'wb'))""")
        st.write("19.โหลด K-Means Model จาก Google Drive")
        st.code("""kmeans_model_path = '/content/drive/MyDrive/IS_Final_Project/ML_Model/kmeans_model.sav'
loaded_kmeans_model = cp.load(open(kmeans_model_path, 'rb'))""")
        st.write("20.PCA ลดมิติเป็น 2D เพื่อวาดกราฟ Clustering")
        st.code("""pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", edgecolors="k")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering Visualization")
plt.colorbar(label="Cluster")
plt.show()""")
        st.write("21.ดาวน์โหลดไฟล์โมเดลที่บันทึกไว้")
        st.code("""files.download("/content/drive/MyDrive/IS_Final_Project/ML_Model/linear_regression_model.sav")
files.download("/content/drive/MyDrive/IS_Final_Project/ML_Model/scaler.sav") 
files.download("/content/drive/MyDrive/IS_Final_Project/ML_Model/kmeans_model.sav") """)

elif option == "🧠 Neural Network (NN)":
        st.header("ขั้นตอนการพัฒนา Neural Network (NN)")
        st.write("1.ติดตั้งเข้าไลบรารีที่จำเป็น")
        st.code("""import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
""")
        st.write("2.เชื่อมต่อ Google Drive เพื่อให้สามารถเข้าถึงไฟล์ใน Drive จาก Colab ได้")
        st.code("""from google.colab import drive
drive.mount('/content/drive')""")
        st.write("3.กำหนด Path ที่อยู่ของไฟล์ Dataset และใช้ np.loadtxt() โหลดข้อมูลจาก CSV เพื่อเตรียม X_train, Y_train, X_test, Y_test สำหรับการเทรนและทดสอบโมเดล")
        st.code("""#  2. กำหนดพาธของไฟล์ 
input_path = "/content/drive/MyDrive/IS_Final_Project/NN_Model/DatasetNN/input.csv"
labels_path = "/content/drive/MyDrive/IS_Final_Project/NN_Model/DatasetNN/labels.csv"
input_test_path = "/content/drive/MyDrive/IS_Final_Project/NN_Model/DatasetNN/input_test.csv"
labels_test_path = "/content/drive/MyDrive/IS_Final_Project/NN_Model/DatasetNN/labels_test.csv"

#  3. โหลดข้อมูล (ตรวจสอบว่า CSV ไม่มี header หรือ string)
X_train = np.loadtxt(input_path, delimiter=',')
Y_train = np.loadtxt(labels_path, delimiter=',')

X_test = np.loadtxt(input_test_path, delimiter=',')
Y_test = np.loadtxt(labels_test_path, delimiter=',')

""")
        st.write("4.Reshape ข้อมูล ปรับ X_train และ X_test ให้อยู่ในรูปภาพ 100×100×3 (RGB) เพื่อใช้กับ CNN และ Normalize แปลงค่าพิกเซลให้อยู่ในช่วง [0,1] โดยหารด้วย 255.0 เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น ")
        st.code("""#  Reshape ข้อมูลให้ตรงกับ CNN Input (100x100x3)
X_train = X_train.reshape(-1, 100, 100, 3)
Y_train = Y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 100, 100, 3)
Y_test = Y_test.reshape(-1, 1)

#  Normalize ค่าให้อยู่ในช่วง [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0""")
        st.write("5.แสดงขนาดของข้อมูล")
        st.code("""print("Shape of X_train ", X_train.shape)
print("Shape of Y_train ", Y_train.shape)
print("Shape of X_test ", X_test.shape)
print("Shape of Y_test ", Y_test.shape)""")
        st.write("6.แสดงภาพตัวอย่างจาก Dataset โดยเลือกภาพแบบสุ่มจาก X_train แล้วแสดงผลด้วย plt.imshow() พร้อม label ของภาพ")
        st.code("""idx = random.randint(0, len(X_train) - 1)
plt.imshow(X_train[idx])
plt.title(f"Example Image - Label: {Y_train[idx][0]}")
plt.show()""")
        st.write("7.สร้างโมเดล CNN ด้วย Sequential() สำหรับ Binary Classification (จำแนกภาพ หมา vs แมว)")
        with st.expander("📌 **โครงสร้างโมเดล**"):
            st.write("""
            - **Conv2D(32, 3×3) + ReLU**  ดึงฟีเจอร์จากภาพ
            - **MaxPooling2D(2×2)** ลดขนาดฟีเจอร์แมพ
            - **Conv2D(64, 3×3) + ReLU** เพิ่มความลึกของฟีเจอร์
            - **MaxPooling2D(2×2)** ลดขนาดฟีเจอร์แมพอีกครั้ง
            - **Flatten()** แปลงเป็นเวกเตอร์
            - **Dense(128) + ReLU** Fully Connected Layer
            - **Dense(1) + Sigmoid** เอาต์พุตแบบ 0 หรือ 1 (Binary Classification) 
            """)
        st.code("""model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary Classification (Dog vs Cat)
])""")
        st.write("8.คอมไพล์ โมเดล CNN สำหรับ Binary Classification โดยใช้ Conv2D และ MaxPooling2D เพื่อดึงฟีเจอร์จากภาพ จากนั้นใช้ Dense สำหรับการจำแนกและผลลัพธ์จะออกมาเป็น 0 หรือ 1 (หมา หรือ แมว) โดยใช้ binary_crossentropy เป็น loss function และ adam optimizer พร้อมวัด accuracy ในการฝึกโมเดล")
        st.code("""model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])""")
        st.write("9.ฝึกโมเดล ด้วย model.fit() โมเดลจะถูกฝึกและเก็บประวัติการเรียนรู้ในตัวแปร history เพื่อใช้ในการวิเคราะห์ภายหลัง")
        st.code("""history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))""")
        st.write("10.บันทึกโมเดลที่ฝึกเสร็จแล้วในไฟล์ .sav โดยใช้ pickle.dump() และเก็บไว้ใน Google Drive ที่pathที่กำหนด")
        st.code("""model_path = "/content/drive/MyDrive/IS_Final_Project/NN_Model/finalized_model.sav"
pickle.dump(model, open(model_path, 'wb'))
print(f"✅ Model saved at: {model_path}")""")
        st.write("11.โหลดโมเดลที่บันทึกไว้ก่อนหน้านี้จากไฟล์ .sav ด้วย pickle.load()")
        st.code("""loaded_model = pickle.load(open(model_path, 'rb'))
print("✅ Model Loaded Successfully!")""")
        st.write("12.ประเมินโมเดล ที่โหลดมาโดยใช้ evaluate() บนชุดข้อมูลทดสอบ (X_test, Y_test) และแสดง ค่า Loss และ Accuracy ของโมเดลในการทดสอบ")
        st.code("""loss, acc = loaded_model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")""")
        st.write("13.แสดงภาพทดสอบสุ่มจาก X_test และใช้โมเดลที่โหลดมาเพื่อทำนายว่าเป็น หมา หรือ แมว โดยแสดงผลการทำนายดังกล่าวว่าโมเดลทำนายเป็น Dog หรือ Cat")
        st.code("""#  Making Predictions
idx2 = random.randint(0, len(Y_test) - 1)
plt.imshow(X_test[idx2])
plt.title("Random Test Image")
plt.show()

y_pred = loaded_model.predict(X_test[idx2].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

# Print ค่าทำนาย
pred_class = "Dog" if y_pred[0][0] == 0 else "Cat"
print(f"Our model predicts it is a: {pred_class}")""")
        st.write("14.ทำนายผลทั้งหมดจาก X_test ด้วยโมเดลที่โหลดมา จากนั้นแปลงผลลัพธ์เป็น 0 (หมา) หรือ 1 (แมว) และใช้ classification_report() เพื่อแสดงผลการทำนายพร้อม precision, recall, f1-score สำหรับแต่ละคลาส")
        st.code(""""y_pred_all = loaded_model.predict(X_test)
y_pred_classes = (y_pred_all > 0.5).astype(int)
print(classification_report(Y_test, y_pred_classes, target_names=['Dog', 'Cat']))""")
        st.write("15.คำนวณ Confusion Matrix จากผลลัพธ์ที่ทำนายและแสดงผลเป็น heatmap โดยใช้ sns.heatmap() เพื่อแสดงความสัมพันธ์ระหว่างค่าทำนาย (Predicted) และค่าจริง (Actual) สำหรับ หมา และ แมว")
        st.code("""cm = confusion_matrix(Y_test, y_pred_classes)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()""")
        st.write("16.แสดงกราฟการฝึกโมเดล")
        with st.expander("📌 **กราฟการฝึกโมเดล**"):
            st.write("""
            - **กราฟ Accuracy**  เปรียบเทียบความแม่นยำระหว่างการฝึกและการทดสอบ (validation)
            - **กราฟ Loss** เปรียบเทียบการสูญเสียระหว่างการฝึกและการทดสอบ (validation) โดยใช้ plt.subplot() และ history เพื่อแสดงผลทั้งสองกราฟในหน้าต่างเดียวกัน.
            """)
        st.code("""plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()""")
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
