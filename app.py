import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="Spor Salonu Veri Analizi", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("gym_members_exercise_tracking.csv")
        return df
    except FileNotFoundError:
        st.error("CSV dosyası bulunamadı.")
        return None

def main():
    st.sidebar.title("YBS Proje Menüsü")
    menu = st.sidebar.radio("Bölümler", ["Ana Sayfa", "Veri Analizi", "Görselleştirme", "Görüntü İşleme"])

    df = load_data()

    if df is not None:
        
        if menu == "Ana Sayfa":
            st.title("Spor Salonu Üyeleri Analizi")
            st.markdown("""
            Bu proje kapsamında spor salonu üyelerinin verileri incelenmiştir.
            
            **Kullanılan Yöntemler:**
            * Pandas ile veri manipülasyonu
            * Seaborn ve Matplotlib ile görselleştirme
            * NumPy ile görüntü işleme teknikleri
            """)
            
        elif menu == "Veri Analizi":
            st.header("Veri İnceleme")
            
            st.subheader("İlk 5 Satır")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            col1.write(f"Satır Sayısı: {df.shape[0]}")
            col1.write(f"Sütun Sayısı: {df.shape[1]}")
            
            st.subheader("İstatistiksel Veriler")
            st.write(df.describe())
            
            st.subheader("Filtreleme")
            gender_option = st.selectbox("Cinsiyet Seç:", df['Gender'].unique())
            filtered_df = df[df['Gender'] == gender_option]
            st.dataframe(filtered_df.head())
            
            st.subheader("Ortalama Kalori Analizi")
            groupby_df = df.groupby("Workout_Type")["Calories_Burned"].mean().reset_index()
            st.dataframe(groupby_df)

        elif menu == "Görselleştirme":
            st.header("Grafikler")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Yaş Dağılımı")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.histplot(df['Age'], bins=20, kde=True, ax=ax1)
                st.pyplot(fig1)
                
            with col2:
                st.subheader("Egzersiz Türleri")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.countplot(x='Workout_Type', data=df, ax=ax2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)
                
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Süre ve Kalori İlişkisi")
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x='Session_Duration (hours)', y='Calories_Burned', hue='Workout_Type', data=df, ax=ax3)
                st.pyplot(fig3)
                
            with col4:
                st.subheader("Korelasyon Matrisi")
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax4)
                st.pyplot(fig4)

        elif menu == "Görüntü İşleme":
            st.header("NumPy Görüntü İşleme")
            
            uploaded_file = st.file_uploader("Resim Yükle", type=['jpg', 'png', 'jpeg'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                st.image(image, caption="Orijinal", use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Gri Tonlama**")
                    if len(img_array.shape) == 3:
                        gray_img = np.mean(img_array, axis=2).astype(np.uint8)
                    else:
                        gray_img = img_array
                    st.image(gray_img, caption="Mean Method", clamp=True, channels='GRAY')
                
                with col2:
                    st.markdown("**Negatif Görüntü**")
                    negative_img = 255 - img_array
                    st.image(negative_img, caption="Inverted")

if __name__ == "__main__":
    main()