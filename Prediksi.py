import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

# ==========================================
# PENGATURAN HALAMAN
# ==========================================
st.set_page_config(page_title="Proyek ML - Prediksi Harga Rumah", page_icon="🏠", layout="wide")

@st.cache_data 
def load_data():
    return pd.read_csv('dataset_rumah.csv', sep=';')

df = load_data()

# ==========================================
# MENU NAVIGASI (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Menu Tahapan")
menu = st.sidebar.radio("Pilih Tahapan Proses:", 
                        ("1. Pemahaman & Eksplorasi Data", 
                         "2. Preprocessing Data", 
                         "3. Pemodelan & Evaluasi", 
                         "4. Simulasi Prediksi"))

st.sidebar.markdown("---")
st.sidebar.info("Proyek Akhir Machine Learning: Supervised Learning - Prediksi Harga Rumah.")

# ==========================================
# TAHAP 1: EKSPLORASI DATA
# ==========================================
if menu == "1. Pemahaman & Eksplorasi Data":
    st.title("Pemahaman & Eksplorasi Data")
    st.write("Dataset ini berisi informasi spesifikasi rumah beserta harganya.")
    
    st.dataframe(df.head(10))
    st.write(f"**Total Data:** {df.shape[0]} baris dan {df.shape[1]} kolom.")
    
    st.subheader("Distribusi Lokasi Rumah")
    lokasi_count = df['lokasi'].value_counts()
    st.bar_chart(lokasi_count)

# ==========================================
# TAHAP 2: PREPROCESSING DATA
# ==========================================
elif menu == "2. Preprocessing Data":
    st.title("Preprocessing Data")
    st.write("Mengubah teks menjadi angka menggunakan **One-Hot Encoding**.")

    if st.button("Mulai Preprocessing"):
        x = df.drop(columns=['harga'])
        y = df['harga']

        x_encoded = pd.get_dummies(x, columns=['lokasi'])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sebelum (Teks)")
            st.dataframe(x[['lokasi']].head())
        with col2:
            st.subheader("Sesudah (Angka)")
            st.dataframe(x_encoded.head())

        st.subheader("Pembagian Data (Train & Test Split)")
        x_train, x_test, y_train, y_test = train_test_split(
            x_encoded, y, test_size=0.2, random_state=42
        )
        st.success(f"Data dibagi: {x_train.shape[0]} Training (Belajar) dan {x_test.shape[0]} Testing (Ujian).")

# ==========================================
# TAHAP 3: PEMODELAN & EVALUASI
# ==========================================
elif menu == "3. Pemodelan & Evaluasi":
    st.title("Pemodelan & Evaluasi")
    st.write("Mari kita lihat proses bagaimana mesin belajar!")
    
    if st.button("🚀 Mulai Proses Pelatihan Mesin", type="primary"):
        
        # 1. Menyiapkan elemen visual untuk loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Preprocessing senyap
        X = pd.get_dummies(df.drop(columns=['harga']), columns=['lokasi'])
        y = df['harga']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- PROSES 1 ---
        status_text.info("🔄 Langkah 1: Membaca dan menyiapkan data training...")
        time.sleep(1) # Jeda waktu agar terlihat sedang memproses
        progress_bar.progress(25)
        
        # --- PROSES 2 ---
        status_text.info("⚙️ Langkah 2: Melatih algoritma Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        time.sleep(1.5)
        progress_bar.progress(50)
        
        # --- PROSES 3 ---
        status_text.info("🌲 Langkah 3: Melatih algoritma Random Forest (Membangun pohon keputusan)...")
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        time.sleep(2)
        progress_bar.progress(80)
        
        # --- PROSES 4 ---
        status_text.info("📊 Langkah 4: Menghitung Akurasi dan Error kedua model...")
        time.sleep(1)
        progress_bar.progress(100)
        
        # Selesai
        status_text.success("✅ Pelatihan Selesai! Berikut adalah hasil ujian dari kedua algoritma:")
        
        eval_data = {
            "Algoritma": ["Linear Regression", "Random Forest"],
            "R2 Score (Akurasi)": [r2_score(y_test, lr_pred), r2_score(y_test, rf_pred)],
            "Mean Absolute Error (Rp)": [mean_absolute_error(y_test, lr_pred), mean_absolute_error(y_test, rf_pred)]
        }
        
        eval_df = pd.DataFrame(eval_data)
        st.dataframe(eval_df.style.highlight_max(subset=['R2 Score (Akurasi)'], color='lightgreen'))
        
        st.info("Berdasarkan tabel di atas, **Random Forest** lebih akurat. Kita akan menggunakan Random Forest di tahap Simulasi Prediksi.")
    else:
        st.warning("Klik tombol di atas untuk melihat proses mesin belajar.")

# ==========================================
# TAHAP 4: SIMULASI PREDIKSI
# ==========================================
elif menu == "4. Simulasi Prediksi":
    st.title("Simulasi Prediksi Harga Baru")
    
    # Preprocessing & Model Setup Senyap
    X = pd.get_dummies(df.drop(columns=['harga']), columns=['lokasi'])
    y = df['harga']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = RandomForestRegressor(random_state=42)
    best_model.fit(X_train, y_train)
    
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            luas_bangunan = st.number_input("Luas Bangunan (m2)", 30, 1000, 120)
            luas_tanah = st.number_input("Luas Tanah (m2)", 30, 2000, 150)
            jumlah_kamar = st.number_input("Jumlah Kamar", 1, 10, 3)
            jumlah_kamar_mandi = st.number_input("Jumlah Kamar Mandi", 1, 10, 2)
        with col2:
            garasi = st.selectbox("Kapasitas Garasi (Mobil)", [0, 1, 2, 3])
            tahun_bangun = st.number_input("Tahun Bangun", 1980, 2026, 2015)
            kualitas = st.slider("Skor Kualitas (1 = Buruk, 5 = Mewah)", 1, 5, 3)
            lokasi = st.selectbox("Lokasi Kota", df['lokasi'].unique())
            
        submit_button = st.form_submit_button("Hitung Prediksi Harga", type="primary")
        
    if submit_button:
        # Simulasi proses loading pendek saat memprediksi
        with st.spinner("Sedang menghitung estimasi harga..."):
            time.sleep(1)
            
            input_data = pd.DataFrame({
                'luas_bangunan': [luas_bangunan], 'luas_tanah': [luas_tanah],
                'jumlah_kamar': [jumlah_kamar], 'jumlah_kamar_mandi': [jumlah_kamar_mandi],
                'garasi': [garasi], 'tahun_bangun': [tahun_bangun],
                'kualitas': [kualitas], 'lokasi': [lokasi]
            })
            
            input_data = pd.get_dummies(input_data, columns=['lokasi'])
            for col in X_train.columns:
                if col not in input_data.columns:
                    input_data[col] = 0 
            input_data = input_data[X_train.columns] 
            
            prediksi = best_model.predict(input_data)[0]
            
            st.success(f"### 💰 Estimasi Harga Rumah: Rp {prediksi:,.0f}")
            