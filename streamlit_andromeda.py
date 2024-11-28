import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import gdown  # Import gdown untuk mengunduh dari Google Drive
import os
import requests
from tqdm import tqdm

# Page configuration
st.set_page_config(
    page_title="Andromeda - Food Detection",
    page_icon="🍽️",
    layout="wide"
)

def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        with open(destination, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress bar
                    if total_size:
                        progress = (downloaded / total_size)
                        progress_bar.progress(progress)
                        progress_text.text(f"Downloaded: {downloaded}/{total_size} bytes ({progress:.1%})")
        
        progress_bar.empty()
        progress_text.empty()

    url = f"https://drive.google.com/uc?id={file_id}"
    session = requests.Session()

    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)

MODEL_ID = '1ogRogs-V0Sq-yeQi0fHrw1SmnkCZv9Nu' 
MODEL_PATH = 'model.h5'

# Color scheme for different food categories
DETECTION_COLORS = {
    "karbohidrat": (255, 0, 0),    # Blue
    "protein": (0, 255, 0),        # Green
    "buah": (0, 0, 255),          # Red
    "sayur": (255, 255, 0),       # Cyan
    "minuman": (255, 0, 255)      # Magenta
}

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Main Menu", 
        ["Home", "Upload Image", "Live Camera"],
        icons=["house", "upload", "camera"],
        menu_icon="cast",
        default_index=0

    )
    # Nambahin confidence threshold di sidebar
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05

    )

# Cache the model loading
@st.cache_resource
def load_detection_model():
    try:
        # Buat folder model jika belum ada
        os.makedirs('model', exist_ok=True)
        
        # Download model jika belum ada
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model from Google Drive...")
            try:
                download_file_from_google_drive(MODEL_ID, MODEL_PATH)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                return None
        
        # Verifikasi file size
        if os.path.getsize(MODEL_PATH) < 1000:  # Minimal size in bytes
            st.error("Downloaded file seems corrupted. Retrying download...")
            os.remove(MODEL_PATH)
            return load_detection_model()
        
        # Load model
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

    # try:
    #     # Unduh model jika belum ada
    #     if not os.path.exists(MODEL_PATH):
    #         url = f'https://drive.google.com/uc?id=1ogRogs-V0Sq-yeQi0fHrw1SmnkCZv9Nu'
    #         st.info("Downloading model from Google Drive")
    #         gdown.download(url, MODEL_PATH, quiet=False)

    #     model = load_model(MODEL_PATH)
    #     return model
    # except Exception as e:
    #     st.error(f"Error loading model: {str(e)}")
    #     return None

# Fungsi untuk mendeteksi kamera yang tersedia
def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        # cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Backend DirectShow
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Global variables
CLASS_NAMES = ["karbohidrat", "protein", "buah", "sayur", "minuman"]
# CLASS_NAMES = ["buah", "karbohidrat", "minuman", "protein", "sayur"]
# CONFIDENCE_THRESHOLD = 0.5

def preprocess_frame(frame, target_size=(224, 224)):
    """Preprocess frame for model input"""
    processed_frame = cv2.resize(frame, target_size)
    processed_frame = processed_frame.astype("float32") / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame

def get_region_proposals(image):
    """Get region proposals using contour detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Minimum size threshold
            regions.append((x, y, w, h))
    return regions

def draw_detection_boxes(image, detections):
    """Draw detection boxes with labels and confidence scores"""
    image_with_boxes = image.copy()
    
    for det in detections:
        x, y, w, h = det['box']
        class_name = det['class']
        confidence = det['confidence']
        color = DETECTION_COLORS.get(class_name, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        
        # Create label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        
        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw label background
        cv2.rectangle(image_with_boxes, 
                     (x, y - label_h - 10), 
                     (x + label_w, y),
                     color, 
                     -1)
        
        # Draw label text
        cv2.putText(image_with_boxes, 
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)
    
    return image_with_boxes

# def detect_objects(image, model):
#     """Detect objects in the image"""
#     height, width = image.shape[:2]
#     regions = get_region_proposals(image)
#     detections = []
    
#     for x, y, w, h in regions:
#         # Extract region
#         region = image[y:y+h, x:x+w]
        
#         # Preprocess region
#         processed_region = preprocess_frame(region)
        
#         # Get predictions
#         predictions = model.predict(processed_region, verbose=0)[0]
#         confidence = float(np.max(predictions))
        
#         if confidence >= CONFIDENCE_THRESHOLD:
#             class_idx = np.argmax(predictions)
#             detections.append({
#                 'box': (x, y, w, h),
#                 'confidence': confidence,
#                 'class': CLASS_NAMES[class_idx]
#             })
    
#     return detections

def detect_objects(image, model, threshold=0.5):
    height, width = image.shape[:2]
    regions = get_region_proposals(image)
    detections = []

    for x, y, w, h in regions:
        # Extract region
        region = image[y:y+h, x:x+w]

        # Preprocess region
        processed_region = preprocess_frame(region)

        # Get predictions
        predictions = model.predict(processed_region, verbose=0)[0]
        confidence = float(np.max(predictions))
        if confidence >= threshold:
            class_idx = np.argmax(predictions)
            detections.append({
                'box': (x, y, w, h),
                'confidence': confidence,
                'class': CLASS_NAMES[class_idx]
            })

    return detections

# Home page
if selected == "Home":
    st.title("Andromeda")
    st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
    
    # Add more detailed information about the system
    st.markdown("""
    ### About the System
    This system helps you analyze food items according to the Indonesian healthy eating guide "4 Sehat 5 Sempurna".
    
    The system can detect:
    - 🍚 Carbohydrates (Karbohidrat)
    - 🥩 Proteins (Protein)
    - 🥕 Vegetables (Sayur)
    - 🍎 Fruits (Buah)
    - 🥛 Beverages (Minuman)
    """)

# Upload Image page
# elif selected == "Upload Image":
#     st.title("Food Image Detection")
    
#     model = load_detection_model()
#     if model is None:
#         st.error("Failed to load model. Please check the model file.")
#     else:
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
#         if uploaded_file is not None:
#             # Load and display original image
#             image = Image.open(uploaded_file)
#             image_np = np.array(image)
#             image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", use_column_width=True)
            
#             # Process image
#             if st.button("Detect Food Items"):
#                 with st.spinner("Processing..."):
#                     # Detect objects
#                     detections = detect_objects(image_cv, model)
                    
#                     # Draw detection boxes
#                     result_image = draw_detection_boxes(image_cv, detections)
#                     result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
#                     # Display results
#                     with col2:
#                         st.image(result_image, caption="Detection Result", use_column_width=True)
                    
#                     # Display detection details
#                     st.subheader("Detection Details:")
#                     for det in detections:
#                         st.write(f"- Found {det['class']} with {det['confidence']:.2f} confidence")

elif selected == "Upload Image":
    st.title("Food Image Detection")

    model = load_detection_model()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            if st.button("Detect Food Items"):
                with st.spinner("Processing..."):

                    detections = detect_objects(image_cv, model, threshold=confidence_threshold)

                    # Draw detection boxes
                    result_image = draw_detection_boxes(image_cv, detections)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                    # Display results
                    with col2:
                        st.image(result_image, caption="Detection Result", use_container_width=True)

                    # Display detection details
                    st.subheader("Detection Details:")
                    for det in detections:
                        st.write(f"- Found {det['class']} with {det['confidence']:.2f} confidence")


# Live Camera page
# elif selected == "Live Camera":
#     st.title("Live Camera Detection")
    
#     model = load_detection_model()
#     if model is None:
#         st.error("Failed to load model. Please check the model file.")
#     else:
#         # Add camera controls
#         run = st.checkbox('Start Camera')
#         FRAME_WINDOW = st.image([])
        
#         camera = cv2.VideoCapture(0)
        
#         while run:
#             ret, frame = camera.read()
#             if not ret:
#                 st.error("Failed to access camera")
#                 break
            
#             # Detect objects
#             detections = detect_objects(frame, model)
            
#             # Draw detection boxes
#             frame_with_detections = draw_detection_boxes(frame, detections)
            
#             # Convert BGR to RGB for display
#             frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
#             FRAME_WINDOW.image(frame_rgb)
            
#             # Add small delay to reduce CPU usage
#             time.sleep(0.1)
            
#         camera.release()

elif selected == "Live Camera":
    st.title("Live Camera Detection")

    model = load_detection_model()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
    else:
        # List available cameras
        cameras = list_available_cameras()
        if not cameras:
            st.error("No cameras found. Please connect a camera and refresh.")
        else:
            camera_index = st.selectbox("Select Camera", cameras, format_func=lambda x: f"Camera {x}")
            run = st.checkbox("Start Camera")
            FRAME_WINDOW = st.image([])
            
            if run:
                # Open the selected camera
                camera = cv2.VideoCapture(camera_index)
                if not camera.isOpened():
                    st.error(f"Failed to open camera {camera_index}.")
                else:
                    try:
                            while run:
                                ret, frame = camera.read()
                                if not ret:
                                    st.error(f"Failed to read from camera {camera_index}.")
                                    break

                                # Detect objects
                                detections = detect_objects(frame, model, threshold=confidence_threshold)
                    
                                # Draw detection boxes
                                frame_with_detections = draw_detection_boxes(frame, detections)
                    
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(frame_rgb, channels="RGB")
                    
                                # Add small delay to reduce CPU usage
                                time.sleep(0.1)
                    finally:
                        camera.release()
                        cv2.destroyAllWindows()


# import streamlit as st
# from streamlit_option_menu import option_menu
# import tensorflow as tf
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from PIL import Image
# import time
# import gdown  # Import gdown untuk mengunduh dari Google Drive
# import os

# # Page configuration
# st.set_page_config(
#     page_title="Andromeda - Food Detection",
#     page_icon="🍽️",
#     layout="wide"
# )

# MODEL_ID = '1ogRogs-V0Sq-yeQi0fHrw1SmnkCZv9Nu' 
# MODEL_PATH = 'model.h5'

# # Color scheme for different food categories
# DETECTION_COLORS = {
#     "karbohidrat": (255, 0, 0),    # Blue
#     "protein": (0, 255, 0),        # Green
#     "buah": (0, 0, 255),          # Red
#     "sayur": (255, 255, 0),       # Cyan
#     "minuman": (255, 0, 255)      # Magenta
# }

# # Sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         "Main Menu", 
#         ["Home", "Upload Image", "Live Camera"],
#         icons=["house", "upload", "camera"],
#         menu_icon="cast",
#         default_index=0
#     )

# # Cache the model loading
# @st.cache_resource
# def load_detection_model():
#     try:
#         # Unduh model jika belum ada
#         if not os.path.exists(MODEL_PATH):
#             url = f'https://drive.google.com/uc?id=1ogRogs-V0Sq-yeQi0fHrw1SmnkCZv9Nu'
#             st.info("Downloading model from Google Drive...")
#             gdown.download(url, MODEL_PATH, quiet=False)

#         model = load_model(MODEL_PATH)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

#     # try:
#     #     model = load_model('./model/model.h5')
#     #     return model
#     # except Exception as e:
#     #     st.error(f"Error loading model: {str(e)}")
#     #     return None

# # Global variables
# CLASS_NAMES = ["karbohidrat", "protein", "buah", "sayur", "minuman"]
# CONFIDENCE_THRESHOLD = 0.5

# def preprocess_frame(frame, target_size=(224, 224)):
#     """Preprocess frame for model input"""
#     processed_frame = cv2.resize(frame, target_size)
#     processed_frame = processed_frame.astype("float32") / 255.0
#     processed_frame = np.expand_dims(processed_frame, axis=0)
#     return processed_frame

# def get_region_proposals(image):
#     """Get region proposals using contour detection"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     regions = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 50 and h > 50:  # Minimum size threshold
#             regions.append((x, y, w, h))
#     return regions

# def draw_detection_boxes(image, detections):
#     """Draw detection boxes with labels and confidence scores"""
#     image_with_boxes = image.copy()
    
#     for det in detections:
#         x, y, w, h = det['box']
#         class_name = det['class']
#         confidence = det['confidence']
#         color = DETECTION_COLORS.get(class_name, (0, 255, 0))
        
#         # Draw bounding box
#         cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        
#         # Create label with class name and confidence
#         label = f"{class_name}: {confidence:.2f}"
        
#         # Get label size
#         (label_w, label_h), baseline = cv2.getTextSize(
#             label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
#         # Draw label background
#         cv2.rectangle(image_with_boxes, 
#                      (x, y - label_h - 10), 
#                      (x + label_w, y),
#                      color, 
#                      -1)
        
#         # Draw label text
#         cv2.putText(image_with_boxes, 
#                     label,
#                     (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     2)
    
#     return image_with_boxes

# def detect_objects(image, model):
#     """Detect objects in the image"""
#     height, width = image.shape[:2]
#     regions = get_region_proposals(image)
#     detections = []
    
#     for x, y, w, h in regions:
#         # Extract region
#         region = image[y:y+h, x:x+w]
        
#         # Preprocess region
#         processed_region = preprocess_frame(region)
        
#         # Get predictions
#         predictions = model.predict(processed_region, verbose=0)[0]
#         confidence = float(np.max(predictions))
        
#         if confidence >= CONFIDENCE_THRESHOLD:
#             class_idx = np.argmax(predictions)
#             detections.append({
#                 'box': (x, y, w, h),
#                 'confidence': confidence,
#                 'class': CLASS_NAMES[class_idx]
#             })
    
#     return detections

# # Home page
# if selected == "Home":
#     st.title("Andromeda")
#     st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
    
#     # Add more detailed information about the system
#     st.markdown("""
#     ### About the System
#     This system helps you analyze food items according to the Indonesian healthy eating guide "4 Sehat 5 Sempurna".
    
#     The system can detect:
#     - 🍚 Carbohydrates (Karbohidrat)
#     - 🥩 Proteins (Protein)
#     - 🥕 Vegetables (Sayur)
#     - 🍎 Fruits (Buah)
#     - 🥛 Beverages (Minuman)
#     """)

# # Upload Image page
# elif selected == "Upload Image":
#     st.title("Food Image Detection")
    
#     model = load_detection_model()
#     if model is None:
#         st.error("Failed to load model. Please check the model file.")
#     else:
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
#         if uploaded_file is not None:
#             # Load and display original image
#             image = Image.open(uploaded_file)
#             image_np = np.array(image)
#             image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", use_column_width=True)
            
#             # Process image
#             if st.button("Detect Food Items"):
#                 with st.spinner("Processing..."):
#                     # Detect objects
#                     detections = detect_objects(image_cv, model)
                    
#                     # Draw detection boxes
#                     result_image = draw_detection_boxes(image_cv, detections)
#                     result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
#                     # Display results
#                     with col2:
#                         st.image(result_image, caption="Detection Result", use_column_width=True)
                    
#                     # Display detection details
#                     st.subheader("Detection Details:")
#                     for det in detections:
#                         st.write(f"- Found {det['class']} with {det['confidence']:.2f} confidence")

# # Live Camera page
# elif selected == "Live Camera":
#     st.title("Live Camera Detection")
    
#     model = load_detection_model()
#     if model is None:
#         st.error("Failed to load model. Please check the model file.")
#     else:
#         # Add camera controls
#         run = st.checkbox('Start Camera')
#         FRAME_WINDOW = st.image([])
        
#         camera = cv2.VideoCapture(0)
        
#         while run:
#             ret, frame = camera.read()
#             if not ret:
#                 st.error("Failed to access camera")
#                 break
            
#             # Detect objects
#             detections = detect_objects(frame, model)
            
#             # Draw detection boxes
#             frame_with_detections = draw_detection_boxes(frame, detections)
            
#             # Convert BGR to RGB for display
#             frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
#             FRAME_WINDOW.image(frame_rgb)
            
#             # Add small delay to reduce CPU usage
#             time.sleep(0.1)
            
#         camera.release()
