import os
import tensorflow as tf
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from PIL import Image
import gdown
from tqdm import tqdm
import io
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Set page configuration
st.set_page_config(
    page_title="Sistem Deteksi Makanan dan Minuman",
    page_icon="🍽️",
    layout="wide"
)

MODEL_ID = '1catZKB9HcjHt4FC40bRyeVvE-P24Ev_Q' 
MODEL_PATH = 'FINAL_MODEL.h5'

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #9ad5fc;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("🍽️ Teknologi Cerdas untuk Konsumsi Bijak dan Berkelanjutan")
st.write("Sistem ini dapat mengklasifikasikan makanan dan minuman melalui upload gambar atau secara real-time")

# Define class names and descriptions
class_names = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']
class_descriptions = {
    'buah': """
        🍎 Kategori Buah
        - Sumber vitamin dan mineral alami
        - Mengandung serat tinggi
        - Baik untuk sistem pencernaan
    """,
    'karbohidrat': """
        🍚 Kategori Karbohidrat
        - Sumber energi utama tubuh
        - Termasuk nasi, roti, dan umbi-umbian
        - Penting untuk aktivitas sehari-hari
    """,
    'minuman': """
        🥤 Kategori Minuman
        - Membantu hidrasi tubuh
        - Beragam jenis minuman sehat
        - Penting untuk metabolisme
    """,
    'protein': """
        🍖 Kategori Protein
        - Penting untuk pertumbuhan
        - Sumber protein hewani dan nabati
        - Membantu pembentukan otot
    """,
    'sayur': """
        🥬 Kategori Sayuran
        - Kaya akan vitamin dan mineral
        - Sumber serat yang baik
        - Mendukung sistem imun
    """
}

# Load model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model from Google Drive...")
            url = f'https://drive.google.com/uc?id={MODEL_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
        image = Image.fromarray(image)
    else:
        original = np.array(image)
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array, original

def predict_image(image):
    """
    Make prediction with multiple object detection and draw flexible bounding boxes
    """
    try:
        processed_image, original_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        
        height, width = original_image.shape[:2]
        output_image = original_image.copy()
        
        # Lower threshold for better detection
        CONFIDENCE_THRESHOLD = 0.2
        
        detected_objects = []
        
        # Use sliding window approach for multiple detections
        window_sizes = [(224, 224), (112, 112), (168, 168)]
        stride_ratio = 0.5
        
        for window_size in window_sizes:
            stride = (int(window_size[0] * stride_ratio), int(window_size[1] * stride_ratio))
            
            for y in range(0, height - window_size[1] + 1, stride[1]):
                for x in range(0, width - window_size[0] + 1, stride[0]):
                    # Extract window
                    window = original_image[y:y + window_size[1], x:x + window_size[0]]
                    
                    # Process window
                    window_processed = cv2.resize(window, (224, 224))
                    window_processed = window_processed / 255.0
                    window_processed = np.expand_dims(window_processed, 0)
                    
                    # Predict
                    window_predictions = model.predict(window_processed, verbose=0)
                    
                    # Check for detections
                    for class_idx, confidence in enumerate(window_predictions[0]):
                        if confidence > CONFIDENCE_THRESHOLD:
                            detected_objects.append({
                                'class': class_names[class_idx],
                                'confidence': confidence * 100,
                                'bbox': (x, y, x + window_size[0], y + window_size[1])
                            })
        
        # Apply non-maximum suppression
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            return intersection / float(area1 + area2 - intersection)
        
        # Sort by confidence
        detected_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS
        filtered_objects = []
        while detected_objects:
            current = detected_objects.pop(0)
            filtered_objects.append(current)
            
            detected_objects = [
                obj for obj in detected_objects
                if calculate_iou(current['bbox'], obj['bbox']) < 0.3
            ]
        
        # Draw detections
        for obj in filtered_objects:
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            
            x1, y1, x2, y2 = map(int, obj['bbox'])
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{obj['class']}: {obj['confidence']:.2f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Ensure label background doesn't go outside image
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(output_image, 
                        (x1, label_y - label_size[1] - 10),
                        (x1 + label_size[0], label_y),
                        color, -1)
            
            # Draw label text
            cv2.putText(output_image, label,
                       (x1, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate overall probabilities
        all_probabilities = {}
        for obj in filtered_objects:
            class_name = obj['class']
            if class_name not in all_probabilities:
                all_probabilities[class_name] = obj['confidence']
            else:
                all_probabilities[class_name] = max(all_probabilities[class_name], obj['confidence'])
        
        # Fill in missing classes with zero probability
        for class_name in class_names:
            if class_name not in all_probabilities:
                all_probabilities[class_name] = 0.0
        
        return {
            'detected_objects': filtered_objects,
            'all_probabilities': all_probabilities,
            'output_image': output_image
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.last_prediction_time = time.time()
        self.prediction_interval = 0.5  # Predict every 0.5 seconds
        self.current_prediction = None
        self.frame_count = 0
        self.buffer_size = 3
        self.detection_buffer = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        current_time = time.time()
        if current_time - self.last_prediction_time >= self.prediction_interval:
            try:
                result = predict_image(img)
                if result:
                    self.current_prediction = result
                    self.detection_buffer.append(result)
                    if len(self.detection_buffer) > self.buffer_size:
                        self.detection_buffer.pop(0)
                self.last_prediction_time = current_time
            except Exception as e:
                st.error(f"Error in video transform: {str(e)}")
        
        if self.current_prediction:
            img = self.current_prediction['output_image']
            
            # Add detection information overlay
            height, width = img.shape[:2]
            y_offset = 30
            if self.detection_buffer:
                latest_detection = self.detection_buffer[-1]
                for obj in latest_detection['detected_objects'][:5]:  # Show top 5 detections
                    text = f"{obj['class']}: {obj['confidence']:.1f}%"
                    cv2.putText(img, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
        
        return img

def main():
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Home", "Upload Gambar", "Real-time Detection"])
    
    with tab1:
        st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
        st.write("Selamat datang di Sistem Andromeda! Sistem ini membantu anda menganalisis komposisi makanan sesuai panduan gizi Indonesia '4 Sehat 5 Sempurna'.")
        st.write("Andromeda telah mengembangkan aplikasi ini sebagai hasil penugasan Final project pada track Artificial Intelligence, Startup Campus.")

        # Informasi tentang aplikasi
        st.markdown(""" 
        ### Tentang Aplikasi
        Aplikasi ini bertujuan untuk mengembangkan **Automated Nutritional Analysis**, yaitu deteksi objek untuk evaluasi makanan sehat berdasarkan prinsip *4 Sehat 5 Sempurna*. 
        Dengan memanfaatkan teknologi berbasis **Convolutional Neural Networks (CNN)**, aplikasi ini mampu:
        - Menganalisis komposisi makanan.
        - Mengevaluasi keseimbangan gizi secara otomatis.
        - Memberikan visualisasi interaktif melalui anotasi objek pada gambar makanan.      

        ### Fitur Utama
        - Deteksi dan klasifikasi makanan menggunakan **CNN**.
        - Evaluasi otomatis keseimbangan nutrisi.
        - Tampilan interaktif dengan anotasi visual.
        - Mendukung edukasi masyarakat tentang gizi berbasis teknologi.

        ### Teknologi yang Digunakan
        1. **TensorFlow/Keras** untuk model CNN.
        2. **OpenCV** untuk pemrosesan gambar.
        3. Dataset untuk makanan dan minuman yang dihubungkan dengan Drive.
                    
        ### Prinsip 4 Sehat 5 Sempurna
        - 🍚 **Carbohydrates (Karbohidrat)**
        - 🥩 **Proteins (Protein)**
        - 🥕 **Vegetables (Sayur)**
        - 🍎 **Fruits (Buah)**
        - 🥛 **Beverages (Minuman)**
        """)

        if st.button("Kepo sama dataset lengkap nya??"):
            kaggle_url = "https://www.kaggle.com/datasets/andromedagroup05/data-4-sehat-5-sempurna/data"
            st.warning("Kamu akan diarahkan ke halaman dataset di Kaggle.")
            st.markdown(
                f'<a href="{kaggle_url}" target="_blank" style="text-decoration:none;">'
                '<button style="background-color:#51baff; color:white; padding:10px 20px; border:none; cursor:pointer;">'
                '**Klik di sini yah!**</button></a>',
                unsafe_allow_html=True
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Yuk Analisis Hidanganmu!")
            uploaded_file = st.file_uploader("Pilih file gambar...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang diunggah", use_column_width=True)
                
                if st.button("Analisis Gambar"):
                    with st.spinner("Sedang menganalisis..."):
                        result = predict_image(image)
                        
                        if result:
                            st.image(result['output_image'], caption="Hasil Deteksi", use_column_width=True)
                            
                            st.write("### Hasil Deteksi:")
                            for obj in result['detected_objects']:
                                st.markdown(f"""
                                <div class="detection-box">
                                    <h4>{obj['class'].title()}</h4>
                                    <p>Confidence: {obj['confidence']:.2f}%</p>
                                    {class_descriptions[obj['class']]}
                                </div>
                                """, unsafe_allow_html=True)
        
        with col2:
            if uploaded_file is not None and result:
                st.write("### Probabilitas per Kategori:")
                for class_name, prob in result['all_probabilities'].items():
                    st.progress(prob / 100)
                    st.write(f"{class_name.title()}: {prob:.2f}%")

    with tab3:
        st.write("### Real-time Food Detection")
        st.write("Gunakan kamera untuk deteksi makanan secara real-time")
        
        rtc_configuration = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]}
        )

        webrtc_ctx = webrtc_streamer(
            key="food-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_transformer_factory=VideoTransformer,
            async_transform=True,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            }
        )

if __name__ == "__main__":
    main()