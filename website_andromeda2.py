import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Andromeda - Food Detection",
    page_icon="🍽️",
    layout="wide"
)

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

# Cache the model loading
@st.cache_resource
def load_detection_model():
    try:
        model = load_model('./model/best_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Global variables
CLASS_NAMES = ["karbohidrat", "protein", "buah", "sayur", "minuman"]
CONFIDENCE_THRESHOLD = 0.5

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

def detect_objects(image, model):
    """Detect objects in the image"""
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
        
        if confidence >= CONFIDENCE_THRESHOLD:
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
elif selected == "Upload Image":
    st.title("Food Image Detection")
    
    model = load_detection_model()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Process image
            if st.button("Detect Food Items"):
                with st.spinner("Processing..."):
                    # Detect objects
                    detections = detect_objects(image_cv, model)
                    
                    # Draw detection boxes
                    result_image = draw_detection_boxes(image_cv, detections)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # Display results
                    with col2:
                        st.image(result_image, caption="Detection Result", use_column_width=True)
                    
                    # Display detection details
                    st.subheader("Detection Details:")
                    for det in detections:
                        st.write(f"- Found {det['class']} with {det['confidence']:.2f} confidence")

# Live Camera page
elif selected == "Live Camera":
    st.title("Live Camera Detection")
    
    model = load_detection_model()
    if model is None:
        st.error("Failed to load model. Please check the model file.")
    else:
        # Add camera controls
        run = st.checkbox('Start Camera')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera")
                break
            
            # Detect objects
            detections = detect_objects(frame, model)
            
            # Draw detection boxes
            frame_with_detections = draw_detection_boxes(frame, detections)
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            # Add small delay to reduce CPU usage
            time.sleep(0.1)
            
        camera.release()



# from tkinter import Image
# import streamlit as st  
# from streamlit_option_menu import option_menu  
# import tensorflow as tf
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# from time import time
# from PIL import Image



# with st.sidebar:
#     selected = option_menu("Main Menu", 
#                            ["Home", "Upload Image", "Live Camera"], 
#                            icons=["house", "upload", "camera"], 
#                            menu_icon="cast", 
#                            default_index=0)

# if selected == "Home":
#     st.title("Andromeda")
#     st.header("Automated Nutritional Analysis: Object Detection for Balanced Meal Evaluation According to 4 Sehat 5 Sempurna")
#     st.write("Hello ges howareyu")

# # Menu Upload image   
# elif selected == "Upload Image":

#     st.title("Klasifikasi Gambar Makanan")
#     st.write("Unggah gambar makanan untuk diklasifikasikan.")

#     # Load model
# @st.cache_resource
# def load_detection_model():
#     return load_model('./model/object_detection_model_2.h5')

# model = load_detection_model()

# # Define class names
# class_names = ['karbohidrat', 'protein', 'sayur', 'buah', 'minuman']

# # Preprocess image for model input
# def preprocess_image(image):
#     image = cv2.resize(image, (224, 224))
#     # image = cv2.resize(image, (150, 150))
#     image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#     return np.expand_dims(image, axis=0)

# # Detect objects in the image
# def detect_objects(image):
#     # Convert PIL image to OpenCV format
#     image_np = np.array(image)
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     # Get region proposals
#     gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     detections = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 50 and h > 50:
#             region = image_np[y:y+h, x:x+w]
#             processed_region = preprocess_image(region)
#             predictions = model.predict(processed_region, verbose=0)
#             confidence = float(np.max(predictions[0]))
#             if confidence >= 0.5:
#                 class_idx = np.argmax(predictions[0])
#                 detections.append({
#                     'box': (x, y, w, h),
#                     'confidence': confidence,
#                     'class': class_names[class_idx]
#                 })
#     return detections

# # Draw detection boxes on the image
# def draw_detections(image, detections):
#     image_np = np.array(image)
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     colors = {
#         'karbohidrat': (255, 0, 0),    # Blue
#         'protein': (0, 255, 0),       # Green
#         'sayur': (0, 0, 255),         # Red
#         'buah': (255, 255, 0),        # Cyan
#         'minuman': (255, 0, 255)      # Magenta
#     }

#     for det in detections:
#         x, y, w, h = det['box']
#         class_name = det['class']
#         confidence = det['confidence']
#         color = colors.get(class_name, (0, 255, 0))

#         # Draw bounding box
#         cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)

#         # Draw label
#         label = f"{class_name}: {confidence:.2f}"
#         cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

# # Streamlit App
# def main():
#     st.title("Food Image Detection")
#     st.write("Upload an image to classify food items in it.")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Load and display the uploaded image
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Detect objects
#         with st.spinner("Processing..."):
#             detections = detect_objects(image)
#             result_image = draw_detections(image, detections)

#         # Display results
#         st.image(result_image, caption="Detection Results", use_column_width=True)
#         st.write("Detections:")
#         for det in detections:
#             st.write(f"- {det['class']} with confidence {det['confidence']:.2f}")

# if __name__ == "__main__":
#     main()


# # Menu live camera 
# elif selected == "Live Camera":
#     st.title("Live Camera")

# # Load model
#     model = load_model('./model/model_lama.h5')

# # Define class names
#     class_names = ['karbohidrat', 'protein', 'sayur', 'buah', 'minuman']

#     def preprocess_image(image):
#         image = cv2.resize(image, (224, 224))
#         # image = cv2.resize(image, (150, 150))
#         image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#         return np.expand_dims(image, axis=0)

#     def get_region_proposals(image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         regions = [
#             (x, y, w, h) for x, y, w, h in [cv2.boundingRect(c) for c in contours]
#             if w > 50 and h > 50
#         ]
#         return regions

#     def detect_objects_in_region(image, box, min_confidence=0.5):
#         x, y, w, h = box
#         region = image[y:y+h, x:x+w]
#         if region.shape[0] < 32 or region.shape[1] < 32:
#             return None
#         processed_region = preprocess_image(region)
#         predictions = model.predict(processed_region, verbose=0)
#         confidence = float(np.max(predictions[0]))
#         if confidence >= min_confidence:
#             class_idx = np.argmax(predictions[0])
#             return {'box': box, 'confidence': confidence, 'class': class_names[class_idx]}
#         return None

#     def apply_nms(detections, nms_threshold=0.3):
#         if not detections:
#             return []
#         boxes = [d['box'] for d in detections]
#         scores = [d['confidence'] for d in detections]
#         boxes_nms = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
#         indices = cv2.dnn.NMSBoxes(boxes_nms, scores, 0.5, nms_threshold)
#         return [detections[i] for i in indices.flatten()]

#     def draw_detection(image, detection):
#         colors = {
#             'karbohidrat': (255, 0, 0), 'protein': (0, 255, 0),
#             'sayur': (0, 0, 255), 'buah': (255, 255, 0), 'minuman': (255, 0, 255)
#         }
#         x, y, w, h = detection['box']
#         class_name = detection['class']
#         confidence = detection['confidence']
#         color = colors.get(class_name, (0, 255, 0))
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#         label = f'{class_name}: {confidence:.2f}'
#         (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(image, (x, y - label_h - 5), (x + label_w, y), color, -1)
#         cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     def process_frame(frame, max_regions=10):
#         regions = get_region_proposals(frame)
#         regions = regions[:max_regions]
#         detections = [detect_objects_in_region(frame, box) for box in regions]
#         detections = [d for d in detections if d is not None]
#         detections = apply_nms(detections)
#         for det in detections:
#             draw_detection(frame, det)
#         return frame

#     def realtime_detection():
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("Error: Could not open webcam.")
#             return

#     # Set frame resolution
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#         fps_time = time()
#         frame_count = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Error: Failed to capture frame.")
#                 break

#             processed_frame = process_frame(frame)

#             frame_count += 1
#             if frame_count >= 30:
#                 fps = frame_count / (time() - fps_time)
#                 cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 frame_count = 0
#                 fps_time = time()

#             cv2.imshow('Real-time Food Detection', processed_frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     if __name__ == "__main__":
#         realtime_detection()
