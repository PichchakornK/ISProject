import streamlit as st

st.set_page_config(page_title="ML & NN Models Information", layout="wide")

st.title("ML & NN Models Information")

st.divider()
option = st.radio(
    "🔍 **เลือกโมเดลที่ต้องการดูข้อมูล:**",
    ("🤖 Machine Learning (ML)", "🧠 Neural Network (NN)"),
    horizontal=True
)

st.divider()

if option == "🤖 Machine Learning (ML)":
    st.title("🤖 Machine Learning (ML)")
    st.write("**Machine Learning (ML)**  คือ การสอนคอมพิวเตอร์ให้สามารถเรียนรู้จากข้อมูล และทำการตัดสินใจหรือทำนายผลโดยไม่ต้องโปรแกรมกำหนดขั้นตอนทุกครั้ง โดยใช้อัลกอริธึมที่สามารถพัฒนาตัวเองจากประสบการณ์ (ข้อมูล) ที่ได้รับ")

    with st.expander("📌 **อัลกอริธึมที่ใช้ในโปรเจค**"):
        st.markdown("""
        - **Linear Regression** → ทำนายค่าต่อเนื่อง เช่น เวลานอนทั้งหมด (sleep_total)
        - **Clustering** → จัดกลุ่มสัตว์ตามลักษณะพฤติกรรม เช่น พฤติกรรมการนอน
        """)
    with st.expander("📌 **Data Preparation**"):
        st.write("""
        **Dataset📝** 
        - **ข้อมูลการนอนของสัตว์เลี้ยงลูกด้วยนม**
        - ชุดข้อมูลนี้ประกอบด้วยข้อมูลเกี่ยวกับสัตว์ชนิดต่างๆ และมีคุณสมบัติดังนี้:
            1.	name: ชื่อสามัญของสัตว์
            2.	genus: ชื่อสกุลของสัตว์ (ลำดับทางชีววิทยา)
            3.	vore: ประเภทของอาหารที่สัตว์บริโภค (เช่น, สัตว์กินเนื้อ, สัตว์กินพืช, สัตว์กินทั้งพืชและสัตว์)
            4.	order: ลำดับทางชีววิทยาของสัตว์ (เช่น, Carnivora, Primates, Rodentia)
            5.	conservation: สถานะการอนุรักษ์ของสัตว์ เช่น "lc" (Least Concern) หมายถึงสถานะที่ไม่ถูกคุกคาม, "nt" (Near Threatened) ใกล้จะถูกคุกคาม, "vu" (Vulnerable) เสี่ยงต่อการสูญพันธุ์, หรือ "en" (Endangered) ใกล้จะสูญพันธุ์
            6.	sleep_total: จำนวนชั่วโมงนอนรวมต่อวันของสัตว์
            7.	sleep_rem: จำนวนชั่วโมงนอน REM (Rapid Eye Movement) ต่อวันของสัตว์ (ถ้ามีข้อมูล)
            8.	sleep_cycle: ระยะเวลาของวงจรการนอนในหนึ่งรอบ (ถ้ามีข้อมูล)
            9.	awake: จำนวนชั่วโมงที่สัตว์ตื่นอยู่ต่อวัน
            10.	brainwt: น้ำหนักสมองของสัตว์ในหน่วยกิโลกรัม (ถ้ามีข้อมูล)
            11.	bodywt: น้ำหนักตัวของสัตว์ในหน่วยกิโลกรัม
            **""")
        st.write("""
        **แหล่งที่มาของDataset** 
        - **https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv**""")
        st.write("""
        **Data Cleaning & Preprocessing** 
        - **ตรวจสอบและจัดการค่า missing ในข้อมูล**
        - **เติมค่าที่หายไปในคอลัมน์ตัวเลขด้วย Mean Imputation**
        - **แปลงข้อมูลหมวดหมู่ (categorical) ให้เป็นข้อมูลตัวเลขด้วย One-Hot Encoding**
        - **ปรับสเกลข้อมูลให้เหมาะสม**""")
        st.write("""
        **การจัดการค่า Missing Values** 
        - **ใช้ Mean Imputation สำหรับคอลัมน์ที่เป็นตัวเลข เพื่อเติมค่าหายไปด้วยค่าเฉลี่ยของคอลัมน์นั้น**""")
        
    with st.expander("📌 **วิธีการเทรนและวัดผลโมเดล**"):
        st.markdown("""
        - **Train/Test Split** → แบ่งข้อมูลออกเป็น Training Set และ Test Set
        - **Loss Function** → ใช้ประเมินข้อผิดพลาดของโมเดล
        - **Evaluation Metrics** → ใช้ Accuracy, Precision, Recall, F1 Score
        """)

    st.title("📉 Linear Regression")
    st.write("""
    **Linear Regression** คืออัลกอริธึมที่ใช้ในการทำนายค่าต่อเนื่อง (continuous value) โดยการหาความสัมพันธ์เชิงเส้นระหว่างตัวแปรอิสระ (independent variables) และตัวแปรตาม (dependent variable)
    """)
    st.write("""
    **วัตถุประสงค์:**
    - **เพื่อทำนาย sleep_total (ชั่วโมงการนอนรวม) โดยใช้คุณลักษณะต่างๆ เช่น น้ำหนักร่างกาย (bodywt), น้ำหนักสมอง (brainwt), การนอนหลับ REM (sleep_rem), จำนวนรอบการนอน (sleep_cycle), และเวลาในการตื่น (awake)**""")

    st.write("สมการของ Linear Regression")
    st.latex(r"y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n")

    st.write("โดยที่:")
    st.write("""
    - \\( y \\) คือค่าที่ทำนาย (ในที่นี้คือ **sleep_total**)
    - \\( X_1, X_2, ..., X_n \\) คือคุณลักษณะต่างๆ เช่น **bodywt, brainwt, sleep_rem, sleep_cycle, awake**
    - \\( \\beta_0 \\) คือค่าคงที่ (intercept)
    - \\( \\beta_1, \\beta_2, ..., \\beta_n \\) คือค่าสัมประสิทธิ์ที่โมเดลจะเรียนรู้
    """)
    
    st.write("""
    ในกรณีศึกษานี้, **ตัวแปรตาม** คือ **sleep_total** (ชั่วโมงการนอนรวม) และ **ตัวแปรอิสระ** ที่ใช้ในการทำนายประกอบด้วย:
    - **bodywt** (น้ำหนักร่างกาย)
    - **brainwt** (น้ำหนักสมอง)
    - **sleep_rem** (การนอนหลับ REM)
    - **sleep_cycle** (จำนวนรอบการนอน)
    - **awake** (เวลาในการตื่น)
    """)

    with st.expander("📌 **Mean Squared Error (MSE)**"):
        st.write("ใช้ในการประเมินค่าผิดพลาดระหว่างค่าทำนายกับค่าจริง ยิ่ง MSE น้อยแสดงว่าโมเดลทำนายได้แม่นยำมากขึ้น")
        st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2")

        st.write("โดยที่:")
        st.write("""
        - \\( y_i \\) คือค่าจริง  
        - \\( \hat{y}_i \\) คือค่าทำนาย  
        - \\( n \\) คือจำนวนตัวอย่าง
        """)


    with st.expander("📌 **R² Score**"):
        st.write("ใช้ในการประเมินว่าโมเดลสามารถอธิบายความแปรปรวนของข้อมูลได้มากแค่ไหน ค่านี้จะอยู่ในช่วง 0 ถึง 1 ถ้า R² ใกล้ 1 หมายถึงโมเดลสามารถอธิบายข้อมูลได้ดีมาก")
        st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")
        st.write("โดยที่:")
        st.write("""
        - \\( R^2 \\) ใช้วัดว่าโมเดลสามารถอธิบายความแปรปรวนของข้อมูลได้มากแค่ไหน (ค่าใกล้ 1 หมายถึงโมเดลดี)
        - \\( y_i \\) คือค่าจริง  
        - \\( \hat{y}_i \\) คือค่าทำนาย  
        - \\( n \\) คือจำนวนตัวอย่าง  
        - \\( \bar{y} \\) คือค่าเฉลี่ยของ \\( y \\)
        """)
    st.write("""
    **สรุปข้อมูลที่ได้:**
    - **ให้ผลลัพธ์ที่ใช้ทำนาย ชั่วโมงการนอน โดยพิจารณาจากลักษณะต่างๆ ของสัตว์ เช่น น้ำหนักร่างกาย, น้ำหนักสมอง, และอื่นๆ**""")

    st.title("📊 K-Means Clustering")
    st.write("""
    **K-Means Clustering** เป็นอัลกอริธึมที่ใช้ในการแบ่งกลุ่มข้อมูล (clustering) โดยการหาค่ากลุ่ม (clusters) ที่เหมือนกันหรือมีลักษณะคล้ายคลึงกันในเชิงลักษณะ (features)""")
    st.write("""
    **วัตถุประสงค์:**
    - **เพื่อแบ่งกลุ่มสัตว์ในข้อมูล โดยอิงจากคุณลักษณะต่างๆ เช่น น้ำหนักร่างกาย(bodywt), น้ำหนักสมอง(brainwt), การนอนหลับ REM(sleep_rem) และอื่นๆ เพื่อหา pattern หรือกลุ่มที่คล้ายคลึงกันในข้อมูล**""")
    st.write("""
    ในกรณีศึกษานี้, **K-Means Clustering** ถูกใช้ในการแบ่งกลุ่มสัตว์ตามพฤติกรรมการนอนหลับและลักษณะทางชีววิทยา เช่น **bodywt** และ **brainwt** โดยการใช้ **Elbow Method** เพื่อตัดสินใจเลือกจำนวนกลุ่ม (K) ที่เหมาะสม
    """)
    st.write("""
    **K-Means Clustering** คืออัลกอริธึมที่ใช้ในการแบ่งกลุ่มข้อมูล (unsupervised learning) โดยไม่ต้องมีการป้ายกำกับข้อมูลล่วงหน้า
    ขั้นตอนหลัก:
    1. เลือกจำนวนกลุ่ม (K) ที่ต้องการ
    2. กำหนดจุดศูนย์กลาง (centroid) ของแต่ละกลุ่ม
    3. แบ่งข้อมูลทุกจุดไปยังกลุ่มที่มีศูนย์กลางใกล้ที่สุด
    4. ปรับตำแหน่งศูนย์กลางให้ตรงกับค่าเฉลี่ยของจุดในกลุ่ม
    5. ทำซ้ำขั้นตอนจนผลลัพธ์คงที่
    """)
    with st.expander("📌 **Elbow Method**"):
        st.write("ใช้ในการหาค่าที่เหมาะสมของจำนวนกลุ่ม (K) โดยการคำนวณ **WCSS** (Within-Cluster Sum of Squares) สำหรับจำนวน K ที่ต่างกัน และดูว่าค่า WCSS ลดลงอย่างรวดเร็วจนถึงจุดที่ไม่ลดลงอีกแล้ว จุดนี้คือจำนวนกลุ่มที่ดีที่สุด")
        st.latex(r"WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} (x - \mu_i)^2")
        st.write("โดยที่:")
        st.write("""
        - **𝐶𝑖** คือกลุ่มที่ i,
        - **𝜇𝑖** คือค่าเฉลี่ยของจุดในกลุ่ม **𝐶𝑖**
        - **𝑥** คือจุดข้อมูล
        """)
    with st.expander("📌 **PCA (Principal Component Analysis)**"):
        st.write("ใช้ลดมิติของข้อมูลที่มีหลายมิติ (เช่น ข้อมูลที่มีหลายคุณลักษณะ) ลงเป็น 2D เพื่อให้สามารถแสดงผลกราฟได้ โดยจะทำการลดมิติข้อมูลที่ไม่จำเป็นออกไปและเก็บข้อมูลที่มีความแปรปรวนสูงที่สุด")
        st.write("""การใช้ **PCA** จะช่วยให้การแสดงผลการแบ่งกลุ่ม (clustering) ดูชัดเจนมากขึ้นบนกราฟ 2D.
    """)
    st.write("""
    **สรุปข้อมูลที่ได้:**
    - **แบ่งข้อมูลสัตว์ออกเป็นกลุ่มต่างๆ ตามคุณลักษณะที่คล้ายคลึงกัน เช่น การนอนหลับ, น้ำหนักสมอง และอื่นๆ ซึ่งช่วยให้เห็นความสัมพันธ์ในข้อมูลและสามารถใช้ข้อมูลที่มีในการวิเคราะห์หรือตัดสินใจเกี่ยวกับสัตว์เหล่านั้นได้**""")

    st.subheader("ขั้นตอนการสร้างและประเมินโมเดลการทำนายข้อมูลสัตว์")

    st.write("""
        ในโปรเจคนี้เราจะใช้โมเดล Linear Regression เพื่อทำนายข้อมูลเกี่ยวกับสัตว์ เช่น ขนาดร่างกาย, ขนาดสมอง, 
        และพฤติกรรมการนอนหลับ โดยการเริ่มต้นจากการเตรียมข้อมูล การสร้างโมเดล และการประเมินผลของโมเดล
    """)

    st.subheader("📂 ข้อมูลและแหล่งที่มา")
    st.write("""
        ข้อมูลที่ใช้ในโปรเจคนี้ประกอบด้วยข้อมูลเกี่ยวกับสัตว์ เช่น ขนาดร่างกาย, ขนาดสมอง, และพฤติกรรมการนอนหลับ 
        ซึ่งจะมีข้อมูลที่มีการขาดหายไปในบางฟีเจอร์ เราจะทำการเติมค่าที่หายไปและทำการแปลงข้อมูลให้พร้อมสำหรับการฝึกโมเดล
        คุณสามารถดาวน์โหลดข้อมูลได้จาก [Link](https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv)
    """)

    st.subheader("🔄 การโหลดและเตรียมข้อมูล")
    st.write("""
        ข้อมูลจะถูกโหลดจากไฟล์ CSV และเราจะเลือกข้อมูลที่มีความจำเป็นสำหรับการฝึกโมเดล เช่น ขนาดร่างกาย, ขนาดสมอง,
        และพฤติกรรมการนอนหลับ. ข้อมูลจะถูกตรวจสอบค่าที่หายไปและใช้เทคนิคการ Imputation เพื่อเติมค่าที่ขาดหายไป.
    """)

    st.subheader("🔧 การจัดการข้อมูล")
    st.write("""
        - **Imputation**: เราจะใช้ `SimpleImputer` เพื่อเติมค่าที่หายไปในฟีเจอร์ที่เป็นตัวเลข.
        - **Normalization**: ใช้ `StandardScaler` เพื่อทำการปรับขนาดข้อมูลให้อยู่ในช่วงที่โมเดลสามารถเรียนรู้ได้ง่าย.
        - **One-Hot Encoding**: สำหรับฟีเจอร์ที่เป็น categorical (เช่น `vore`), จะถูกแปลงเป็นข้อมูลตัวเลขโดยใช้ `get_dummies`.
    """)

    st.subheader("🧠 การพัฒนาโมเดล Linear Regression")
    st.write("""
        โมเดลที่ใช้ในการทำนายข้อมูลจะเป็น **Linear Regression** ซึ่งเป็นโมเดลที่เหมาะสมสำหรับการทำนายค่าต่อเนื่อง (regression).
        โมเดลนี้จะเรียนรู้จากฟีเจอร์ เช่น ขนาดร่างกาย, ขนาดสมอง, และพฤติกรรมการนอนหลับ เพื่อทำนายผลลัพธ์ที่เป็นค่าต่อเนื่อง
    """)

    st.subheader("🛠️ การคอมไพล์โมเดล")
    st.write("""
        โมเดลจะถูกคอมไพล์ด้วย:
        - **Loss function**: `mean_squared_error` ซึ่งใช้สำหรับการประเมินความผิดพลาดของโมเดล
        - **Optimizer**: `Adam` เพื่อปรับปรุงค่าพารามิเตอร์ในโมเดล
        - **Metrics**: ใช้ `r2_score` เพื่อวัดความสามารถในการทำนายของโมเดล
    """)

    st.subheader("📊 การฝึกโมเดล")
    st.write("""
        การฝึกโมเดลจะใช้ **training set** และ **test set** โดยแบ่งข้อมูลออกเป็น 70% สำหรับการฝึกและ 30% สำหรับการทดสอบ.
        โมเดลจะถูกฝึกด้วยข้อมูลที่มี และจะทำนายค่าผลลัพธ์จากชุดทดสอบเพื่อประเมินความแม่นยำของโมเดล
    """)

    st.subheader("🔍 การทดสอบและประเมินผล")
    st.write("""
        หลังจากฝึกโมเดลเสร็จแล้ว เราจะใช้ชุดข้อมูลทดสอบเพื่อประเมินประสิทธิภาพของโมเดล.
        เราจะใช้ **Mean Squared Error (MSE)** และ **R² score** เพื่อดูว่าค่าผลลัพธ์ที่ทำนายตรงกับค่าจริงมากแค่ไหน.
        นอกจากนี้ยังมีการตรวจสอบการโอเวอร์ฟิตของโมเดลโดยเปรียบเทียบค่า R² บนชุดฝึกและชุดทดสอบ.
    """)

    st.subheader("💾 การบันทึกและโหลดโมเดล")
    st.write("""
        หลังจากการฝึกโมเดลเสร็จสิ้น เราสามารถบันทึกโมเดลและ Scaler ลงในไฟล์ `.sav` โดยใช้ `pickle` เพื่อให้สามารถนำโมเดลไปใช้งานในอนาคตได้.
        โมเดลที่บันทึกสามารถโหลดกลับมาใช้งานและทำนายผลกับข้อมูลใหม่ ๆ ได้
    """)

elif option == "🧠 Neural Network (NN)":
    st.title("🧠 Neural Network (NN)")
    st.write("Neural Network (NN) เป็นโมเดลที่เลียนแบบการทำงานของสมอง โดยมีชั้นต่าง ๆ เช่น **Input Layer, Hidden Layers และ Output Layer**")
    with st.expander("📌 **Neural Network ที่ใช้ในโปรเจค**"):
        st.markdown("""
        - **CNN (Convolutional Neural Network)** → โครงข่ายประสาทเทียม (Neural Network) ที่ถูกออกแบบมาเพื่อทำงานเกี่ยวกับข้อมูลภาพ (Image Data) โดยเฉพาะ ซึ่งสามารถใช้ในการจำแนกประเภทของภาพ, การแยกแยะวัตถุ, การตรวจจับและรู้จำลักษณะต่างๆ ในภาพได้
        ซึ่งในโปรเจคนี้ใช้จำแนกภาพของ หมา กับ แมว""")
    with st.expander("📌 **Data Preparation**"):
        st.write("""
        **Dataset📝** 
        - **ข้อมูลภาพที่ประกอบด้วยหมาและแมว โดยแบ่งเป็น training set และ test set**""")
        st.write("""
        **แหล่งที่มาของDataset** 
        - **https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD**""")
        st.write("""
        **การโหลดข้อมูล** 
        - **ใช้ฟังก์ชัน np.loadtxt() เพื่อโหลดไฟล์ CSV ที่เก็บข้อมูลภาพและ labels ที่ไม่มี header หรือ string**""")
        st.write("""
        **การจัดการข้อมูล** 
        - **Reshape: เปลี่ยนรูปข้อมูลภาพจาก 1D เป็น 3D โดยการปรับขนาดภาพเป็น 100x100x3 (RGB) เพื่อให้สอดคล้องกับอินพุตของ CNN**)
        - **Normalization: แปลงค่าพิกเซลในช่วง [0, 255] ให้เป็นช่วง [0, 1] โดยการหารด้วย 255.0 เพื่อให้โมเดลเรียนรู้ได้ง่ายขึ้น**""")
    with st.expander("📌 **องค์ประกอบของ Neural Network**"):
         st.markdown(""" 
            - **Input Layer** → รับข้อมูลเข้า เช่น รูปภาพที่แปลงเป็นเวกเตอร์ 3D (เช่น 100x100x3 สำหรับภาพ RGB)  
            - **Hidden Layer** → ประมวลผลข้อมูลด้วย activation functions เช่น ReLU, มีทั้ง Dense Layer และ Convolutional Layer  
            - **Output Layer** → ให้ผลลัพธ์การทำนาย เช่น class (หมา/แมว), ใช้ Sigmoid สำหรับการจำแนก 2 คลาส 
            - **Activation Function** → ReLU ใช้ใน hidden layers เพื่อเพิ่มความไม่เป็นเชิงเส้น, Sigmoid ใช้ใน output layer สำหรับการจำแนก 2 คลาส
            - **Loss Function** → ใช้ binary_crossentropy สำหรับปัญหาการจำแนก 2 คลาส.  
            - **Optimizer** → ใช้ Adam หรือ SGD เพื่ออัปเดตค่าพารามิเตอร์ 
            - **Metrics** → ใช้ accuracy เพื่อตรวจสอบความแม่นยำในการทำนาย
            """)
    st.title("CNN (Convolutional Neural Network)")
    st.write("""
    **CNN** คือ โครงข่ายประสาทเทียม (Neural Network) ที่ถูกออกแบบมาเพื่อทำงานเกี่ยวกับข้อมูลภาพ (Image Data) โดยเฉพาะ ซึ่งสามารถใช้ในการจำแนกประเภทของภาพ, การแยกแยะวัตถุ, การตรวจจับและรู้จำลักษณะต่างๆ ในภาพได้ ซึ่งในโปรเจคนี้ใช้จำแนกภาพของ หมา กับ แมว""")
    st.subheader("🐶 ขั้นตอนการสร้างโมเดลการจำแนกหมา vs แมว")

    st.write("""
        ในโปรเจคนี้เราจะใช้โมเดล Convolutional Neural Network (CNN) เพื่อจำแนกภาพของหมาและแมวจากข้อมูลภาพ
        ซึ่งจะเริ่มจากการเตรียมข้อมูล การสร้างโมเดล และขั้นตอนต่างๆ ที่เกี่ยวข้อง
    """)


    st.subheader("📂 ข้อมูลและแหล่งที่มา")
    st.write("""
        ข้อมูลที่ใช้ในโปรเจคนี้จะประกอบด้วยภาพของหมาและแมว โดยการเตรียมข้อมูลจะถูกแบ่งเป็นชุดข้อมูล
        Training Set และ Test Set ซึ่งเราจะใช้ข้อมูลนี้ในการฝึกโมเดลและทดสอบประสิทธิภาพของโมเดล
        คุณสามารถดาวน์โหลดได้จาก [Google Drive](https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD)
    """)


    st.subheader("🔄 การโหลดข้อมูล")
    st.write("""
        ข้อมูลภาพจะถูกโหลดจากไฟล์ CSV โดยใช้ฟังก์ชันที่เหมาะสมในการดึงข้อมูลจากไฟล์ที่ไม่มี header หรือ string
        ซึ่งจะเป็นการแปลงข้อมูลภาพให้พร้อมใช้งานในการฝึกโมเดล
    """)


    st.subheader("🔧 การจัดการข้อมูล")
    st.write("""
        - **Reshape**: การเปลี่ยนแปลงรูปภาพจาก 1D (ภาพที่มีขนาดต่าง ๆ) เป็น 3D ขนาด 100x100x3 (RGB) เพื่อให้เหมาะสมกับอินพุตของ CNN
        - **Normalization**: การปรับค่าพิกเซลในภาพจากช่วง [0, 255] เป็น [0, 1] เพื่อให้โมเดลสามารถเรียนรู้ได้ง่ายขึ้น
    """)


    st.subheader("🧠 การพัฒนาโมเดล CNN")
    st.write("""
        การจำแนกหมา vs แมว เราจะใช้ **Convolutional Neural Network (CNN)** ซึ่งเป็นโมเดลที่เหมาะสมสำหรับการทำงานกับข้อมูลภาพ (Image Data)
        โดยจะประกอบไปด้วยหลายเลเยอร์ที่สำคัญ ได้แก่:
        - **Conv2D Layer**: ใช้ฟิลเตอร์ขนาด 3x3 เพื่อลดขนาดข้อมูลและดึงฟีเจอร์จากภาพ
        - **MaxPooling2D Layer**: ลดขนาดของข้อมูลเพื่อเพิ่มความเร็วในการคำนวณและลดการโอเวอร์ฟิต
        - **Flatten**: แปลงข้อมูล 2D ให้เป็น 1D เพื่อให้สามารถเชื่อมต่อกับเลเยอร์ Dense
        - **Dense Layer**: เลเยอร์ที่ใช้ทำการเรียนรู้จากฟีเจอร์ที่ถูกดึงออกมา โดยใช้ ReLU เป็น Activation Function
        - **Output Layer**: ใช้ Sigmoid Activation Function เพื่อให้ผลลัพธ์เป็น 0 หรือ 1 (จำแนกเป็นหมา หรือ แมว)
    """)


    st.subheader("🛠️ การคอมไพล์โมเดล")
    st.write("""
        โมเดลนี้จะถูกคอมไพล์โดยใช้:
        - **Loss function**: `binary_crossentropy` ซึ่งเหมาะสำหรับปัญหาการจำแนก 2 คลาส
        - **Optimizer**: `Adam` ซึ่งเป็นอัลกอริธึมที่นิยมในการฝึกโมเดล
        - **Metrics**: ใช้ `accuracy` ในการวัดความแม่นยำในการจำแนก
    """)


    st.subheader("📊 การฝึกโมเดล")
    st.write("""
        การฝึกโมเดลจะใช้ **10 epochs** และ **batch size 64** เพื่อให้โมเดลเรียนรู้จากข้อมูลที่มี
        โดยการฝึกจะมีการแบ่งข้อมูลออกเป็น Training Set และ Validation Set เพื่อตรวจสอบประสิทธิภาพของโมเดลในระหว่างการฝึก
    """)


    st.subheader("🔍 การทดสอบและประเมินผล")
    st.write("""
        หลังจากฝึกโมเดลเสร็จแล้ว เราจะใช้ชุดข้อมูลทดสอบเพื่อประเมินประสิทธิภาพของโมเดล
        โดยการแสดงผลผ่าน **Confusion Matrix** เพื่อดูความถูกต้องในการจำแนกหมาและแมว
        และ **Classification Report** เพื่อแสดงค่า Precision, Recall, และ F1-Score สำหรับทั้งสองคลาส (หมาและแมว)
    """)

    st.subheader("💾 การบันทึกและโหลดโมเดล")
    st.write("""
        หลังจากการฝึกโมเดลเสร็จสิ้น เราสามารถบันทึกโมเดลลงในไฟล์ `.sav` โดยใช้ `pickle` เพื่อให้สามารถนำโมเดลไปใช้งานได้ในอนาคต
        โมเดลที่บันทึกสามารถโหลดกลับมาใช้งานและทำนายผลกับข้อมูลใหม่ ๆ ได้
    """)
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
    <div style="margin-top: 20px; padding: 10px; background-color: rgba(0, 0, 0, 0.5); border-radius: 10px; text-align: center;">
        <h3 style="color:white;">📚 References</h3>
        <p style="color:white; font-size: 14px;">Dataset for ML: 
            <a href="https://raw.githubusercontent.com/prasertcbs/tutorial/master/msleep.csv" target="_blank" style="color: #3498db; text-decoration: none;">ML Dataset Link</a>
        </p>
        <p style="color:white; font-size: 14px;">Dataset for NN: 
            <a href="https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD" target="_blank" style="color: #3498db; text-decoration: none;">NN Dataset Link</a>
        </p>
        <p style="color:white; font-size: 14px;">Machine Learning Tutorial: 
            <a href="https://www.youtube.com/watch?v=T2yT5vt1NaQ&list=PLoTScYm9O0GH_3VrwwnQafwWQ6ibKnEtU&index=6" target="_blank" style="color: #3498db; text-decoration: none;">ML Video Link</a>
        </p>
        <p style="color:white; font-size: 14px;">Neural Network Tutorial: 
            <a href="https://www.youtube.com/watch?v=J1jhfAw5Uvo&t=629s" target="_blank" style="color: #3498db; text-decoration: none;">NN Video Link</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
