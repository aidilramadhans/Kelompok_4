import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # Untuk VideoFrame

# --- Konfigurasi Halaman Streamlit (untuk tampilan) ---
st.set_page_config(
    page_title="Deteksi Emosi Wajah AI",
    page_icon=":smiley:", # Ikon untuk tab browser
    layout="wide", # Layout lebar
)

# --- CSS Kustom untuk mempercantik tampilan ---
st.markdown(
    """
    <style>
    /* Mengatur gaya untuk judul utama */
    .title-font {
        font-size: 2.8em !important;
        color: #4CAF50; /* Warna hijau */
        text-align: center;
        margin-bottom: 0.5em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); /* Efek bayangan teks */
    }
    /* Mengatur gaya untuk sub-judul atau teks pengantar */
    .intro-text {
        font-size: 1.2em !important;
        color: #BBBBBB; /* Warna abu-abu terang */
        text-align: center;
        margin-bottom: 2em;
    }
    /* Mengatur tata letak radio button agar di tengah */
    .stRadio > div {
        justify-content: center;
    }
    /* Mengatur warna pesan sukses/warning */
    .stAlert > div {
        font-weight: bold;
    }
    .stButton > button {
        background-color: #2196F3; /* Warna biru untuk tombol */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 1.1em;
    }
    /* Gaya untuk probabilitas emosi */
    .emotion-probability {
        font-size: 1.1em;
        color: #EEEEEE;
        margin-bottom: 0.5em;
    }
    /* Menyembunyikan watermark Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 1. Muat Model dan Konfigurasi ---
@st.cache_resource # Cache model agar tidak dimuat berulang kali
def load_emotion_model():
    try:
        model = load_model('model_emosi_fer2013.h5')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}. Pastikan 'model_emosi_fer2013.h5' ada di folder yang sama.")
        st.stop()

@st.cache_resource # Cache cascade agar tidak dimuat berulang kali
def load_face_cascade():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("Gagal memuat haarcascade_frontalface_default.xml. Pastikan file ada dan path benar.")
            st.stop()
        return face_cascade
    except Exception as e:
        st.error(f"Gagal memuat face cascade: {e}. Pastikan 'haarcascade_frontalface_default.xml' ada di folder yang sama.")
        st.stop()

model = load_emotion_model()
face_cascade = load_face_cascade()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- 2. Fungsi Deteksi dan Prediksi Emosi ---
# Fungsi ini sekarang akan menerima NumPy array (RGB) sebagai input
# dan mengembalikan gambar yang sudah diproses, serta daftar probabilitas emosi
def detect_and_predict_emotion(img_array_rgb):
    gray_img = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    
    img_with_boxes = img_array_rgb.copy() 
    
    # List untuk menyimpan (label_emosi, probabilitas) untuk setiap wajah
    all_detected_emotions_with_prob = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2) # Gambar kotak merah

        roi_gray = gray_img[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)[0] # Dapatkan array probabilitas
        predicted_emotion_index = np.argmax(prediction)
        predicted_emotion_label = emotion_labels[predicted_emotion_index]
        predicted_probability = prediction[predicted_emotion_index] * 100 # Konversi ke persentase

        all_detected_emotions_with_prob.append((predicted_emotion_label, predicted_probability))

        # Tampilkan label emosi DAN persentase di atas kotak wajah untuk live stream
        text_to_display = f"{predicted_emotion_label} {predicted_probability:.0f}%"
        cv2.putText(img_with_boxes, text_to_display, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return img_with_boxes, all_detected_emotions_with_prob, len(faces)


# --- 3. Kelas VideoTransformer untuk Live Stream (menggunakan streamlit-webrtc) ---
class FaceEmotionTransformer(VideoTransformerBase):
    def __init__(self, model, face_cascade, emotion_labels):
        self.model = model
        self.face_cascade = face_cascade
        self.emotion_labels = emotion_labels

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Turun ukuran demi kinerja lebih cepat
        img_rgb = cv2.resize(img_rgb, (480, 360))  

        processed_frame_rgb, _, _ = detect_and_predict_emotion(img_rgb)

        processed_frame_bgr = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)

        return av.VideoFrame.from_ndarray(processed_frame_bgr, format="bgr24")


# --- 4. Antarmuka Pengguna Streamlit ---

# Menggunakan markdown dengan class untuk judul
st.markdown("<h1 class='title-font'>Aplikasi Deteksi Emosi Wajah AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='intro-text'>Unggah gambar, ambil foto, atau gunakan kamera langsung untuk deteksi emosi wajah!</p>", unsafe_allow_html=True)


# Pilihan metode input
input_method = st.radio(
    "Pilih metode input:",
    ("Unggah Gambar", "Ambil Foto (Satu Kali)", "Kamera Langsung (Live)"),
    index=2 # Default ke Kamera Langsung
)

# Placeholder untuk pesan hasil (agar tidak berkedip)
result_message_placeholder = st.empty()

# Logika berdasarkan pilihan input
if input_method == "Unggah Gambar":
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
        result_message_placeholder.info("Menganalisis gambar...")

        img_array_rgb = np.array(image.convert('RGB'))
        processed_image, emotions_with_prob, num_faces = detect_and_predict_emotion(img_array_rgb)

        if num_faces > 0:
            result_message_placeholder.success(f"Ditemukan {num_faces} wajah.")
            st.image(processed_image, caption='Hasil Deteksi Emosi', use_column_width=True, channels="RGB")
            st.markdown("<h3 class='subheader'>Probabilitas Emosi:</h3>", unsafe_allow_html=True)
            for emotion, probability in emotions_with_prob:
                st.markdown(f"<p class='emotion-probability'>- <b>{emotion}:</b> {probability:.2f}%</p>", unsafe_allow_html=True)
        else:
            result_message_placeholder.warning("Tidak ada wajah terdeteksi dalam gambar ini.")

elif input_method == "Ambil Foto (Satu Kali)":
    camera_image = st.camera_input("Ambil Foto untuk Deteksi Emosi")

    if camera_image is not None:
        image = Image.open(camera_image)
        result_message_placeholder.info("Menganalisis gambar...")

        img_array_rgb = np.array(image.convert('RGB'))
        processed_image, emotions_with_prob, num_faces = detect_and_predict_emotion(img_array_rgb)

        if num_faces > 0:
            result_message_placeholder.success(f"Ditemukan {num_faces} wajah.")
            st.image(processed_image, caption='Hasil Deteksi Emosi', use_column_width=True, channels="RGB")
            st.markdown("<h3 class='subheader'>Probabilitas Emosi:</h3>", unsafe_allow_html=True)
            for emotion, probability in emotions_with_prob:
                st.markdown(f"<p class='emotion-probability'>- <b>{emotion}:</b> {probability:.2f}%</p>", unsafe_allow_html=True)
        else:
            result_message_placeholder.warning("Tidak ada wajah terdeteksi dalam gambar ini.")

elif input_method == "Kamera Langsung (Live)":
    st.info("Klik 'Mulai' untuk mengaktifkan kamera dan melihat deteksi emosi secara langsung.")
    
    webrtc_ctx = webrtc_streamer(
        key="emotion_detector_live", # Key unik untuk live stream
        video_transformer_factory=lambda: FaceEmotionTransformer(model, face_cascade, emotion_labels),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False}
    )

    if webrtc_ctx.video_transformer:
        st.write("Menganalisis video langsung... (Probabilitas akan muncul di layar video)")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #777;'>Dibuat dengan Streamlit dan Model Deep Learning.</p>", unsafe_allow_html=True)