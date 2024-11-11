import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from io import BytesIO
from scipy import ndimage
from PIL import Image
import tempfile

# Define the image preprocessing class
class ImagePreprocessor:
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def adjust_contrast(image, alpha=1.5, beta=0):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def denoise_image(image, strength=10):
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

    @staticmethod
    def sharpen_image(image):
        gaussian_blur = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 2, gaussian_blur, -1, 0)

    @staticmethod
    def clahe_equalization(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def enhance_edges(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced_image = cv2.addWeighted(image, 0.8, edges_bgr, 0.5, 0)
        return enhanced_image

def sliding_window_detection(image, model, window_size=(640, 640), overlap=0.2):
    """Perform object detection using sliding windows"""
    height, width = image.shape[:2]
    stride_x = int(window_size[0] * (1 - overlap))
    stride_y = int(window_size[1] * (1 - overlap))
    all_detections = []

    for y in range(0, height - window_size[1] + stride_y, stride_y):
        for x in range(0, width - window_size[0] + stride_x, stride_x):
            end_x = min(x + window_size[0], width)
            end_y = min(y + window_size[1], height)
            x = max(0, end_x - window_size[0])
            y = max(0, end_y - window_size[1])

            window = image[y:end_y, x:end_x]
            results = model.predict(window, conf=0.1)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    adjusted_box = [
                        r[0] + x, r[1] + y, r[2] + x, r[3] + y,
                        float(box.conf[0]), int(box.cls[0])
                    ]
                    all_detections.append(adjusted_box)

    return all_detections

def non_max_suppression(boxes, overlap_thresh=0.4):
    """Apply non-maximum suppression to avoid duplicate detections"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    return boxes[keep]

# Set page config and dark mode styling
st.set_page_config(layout="wide", page_title="Barchan Detection System")

st.markdown("""
    <style>
    /* Dark mode colors */
    :root {
        --background-color: #1E1E1E;
        --card-background: #2D2D2D;
        --text-color: #E0E0E0;
        --border-color: #404040;
        --accent-color: #4B9FE1;
        --hover-color: #5DADEA;
    }
    
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--background-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: var(--card-background);
        border-radius: 4px;
        color: var(--text-color);
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: var(--accent-color);
        color: white;
    }
    
    div[data-testid="stToolbar"] {
        display: none;
    }
    
    .upload-box {
        border: 2px dashed var(--border-color);
        border-radius: 5px;
        padding: 20px;
        text-align: center;
        background-color: var(--card-background);
    }
    
    .settings-card {
        background-color: var(--card-background);
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Additional dark mode styling */
    .stDataFrame {
        background-color: var(--card-background) !important;
    }
    
    .stDataFrame [data-testid="stTable"] {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
    }
    
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_result' not in st.session_state:
    st.session_state.processed_result = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Header Section
st.title("🎯 Barchan Detection System")
st.markdown("---")

# Main Layout
tabs = st.tabs(["📸 Image Processing", "📊 Results", "ℹ️ Help"])

with tabs[0]:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Image Input")
        with st.container():
            st.markdown('<div class="upload-box">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Drop your image here or click to upload", 
                                           type=["jpg", "jpeg", "png"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.session_state.original_image = np.array(image)
                st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)
        st.subheader("Detection Settings")
        
        with st.expander("📏 Measurement Settings", expanded=True):
            pix_to_meter = st.number_input(
                "Pixel-to-meter conversion rate:",
                min_value=0.0,
                value=0.01,
                format="%.4f"
            )
        
        with st.expander("🔧 Image Enhancement", expanded=True):
            denoise = st.checkbox("Denoise Image", True)
            sharpen = st.checkbox("Sharpen Image", True)
            contrast = st.checkbox("Adjust Contrast", True)
            clahe = st.checkbox("CLAHE Equalization", True)
            gamma = st.slider("Gamma Adjustment", 0.5, 3.0, 1.98, 0.01)
            enhance_edges = st.checkbox("Enhance Edges", True)
        
        if st.button("🔍 Process and Detect", 
                    disabled=uploaded_file is None,
                    type="primary"):
            if uploaded_file:
                with st.spinner("Processing image..."):
                    try:
                        # Process the image
                        preprocessor = ImagePreprocessor()
                        processed_img = st.session_state.original_image.copy()
                        
                        if denoise:
                            processed_img = preprocessor.denoise_image(processed_img)
                        if sharpen:
                            processed_img = preprocessor.sharpen_image(processed_img)
                        if contrast:
                            processed_img = preprocessor.adjust_contrast(processed_img)
                        if clahe:
                            processed_img = preprocessor.clahe_equalization(processed_img)
                        if gamma:
                            processed_img = preprocessor.adjust_gamma(processed_img, gamma)
                        if enhance_edges:
                            processed_img = preprocessor.enhance_edges(processed_img)

                        # Perform detection
                        detections = sliding_window_detection(processed_img, model)
                        final_detections = non_max_suppression(detections, overlap_thresh=0.4)

                        # Process results
                        bounding_box_data = []
                        result_img = st.session_state.original_image.copy()

                        # Sort detections by area
                        valid_detections = []
                        for det in final_detections:
                            x1, y1, x2, y2, conf, cls_id = det
                            if model.names[int(cls_id)] == "Barchan" and conf > 0.10:
                                area_pixels = (x2 - x1) * (y2 - y1)
                                if area_pixels > 600:
                                    valid_detections.append((det, area_pixels))
                        
                        valid_detections.sort(key=lambda x: x[1], reverse=True)

                        # Draw detections
                        for idx, (det, area_pixels) in enumerate(valid_detections, 1):
                            x1, y1, x2, y2, conf, cls_id = det
                            area_meters = area_pixels * (pix_to_meter ** 2)
                            
                            bounding_box_data.append({
                                "Detection #": idx,
                                "Confidence": conf,
                                "Area (px²)": area_pixels,
                                "Area (m²)": area_meters
                            })
                            
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 204), 2)
                            label = f"#{idx} ({conf:.2f}), Area: {area_meters:.2f} m²"
                            cv2.putText(result_img, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 250, 250), 2)

                        # Convert and store results
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        result_bytes = BytesIO()
                        Image.fromarray(result_img_rgb).save(result_bytes, format='PNG')
                        
                        st.session_state.processed_result = {
                            'result_img': result_img_rgb,
                            'df_boxes': pd.DataFrame(bounding_box_data) if bounding_box_data else None,
                            'result_bytes': result_bytes.getvalue()
                        }
                        
                        st.success("Processing complete! Switch to Results tab to view detections.")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        
        if uploaded_file is None:
            st.info("👆 Please upload an image to begin")
            
        st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    # Results Tab
    if st.session_state.processed_result is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Detection Visualization")
            st.image(st.session_state.processed_result['result_img'], 
                    caption="Detection Result",
                    use_column_width=True)
            
            # Download buttons
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "📥 Download Result Image",
                    data=st.session_state.processed_result['result_bytes'],
                    file_name="detection_result.png",
                    mime="image/png"
                )
            with col_b:
                if st.session_state.processed_result['df_boxes'] is not None:
                    st.download_button(
                        "📥 Download Detection Data",
                        data=st.session_state.processed_result['df_boxes'].to_csv(index=False).encode('utf-8'),
                        file_name="detections.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.subheader("Detection Details")
            if st.session_state.processed_result['df_boxes'] is not None:
                st.dataframe(
                    st.session_state.processed_result['df_boxes'],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Statistics
                st.markdown("### Summary Statistics")
                df = st.session_state.processed_result['df_boxes']
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Total Detections", len(df))
                    st.metric("Avg Confidence", f"{df['Confidence'].mean():.2%}")
                with stats_col2:
                    st.metric("Total Area (m²)", f"{df['Area (m²)'].sum():.2f}")
                    st.metric("Avg Area (m²)", f"{df['Area (m²)'].mean():.2f}")
            else:
                st.info("No detections found in the image.")
    else:
        st.info("👈 Process an image first to see results here")

with tabs[2]:
    # Help Tab
    st.subheader("How to Use")
    st.markdown("""
    1. **Upload Image**
        * Click the upload box or drag and drop your image
        * Supported formats: JPG, JPEG, PNG
        
    2. **Adjust Settings**
        * Set the pixel-to-meter conversion rate
        * Configure image enhancement options
        * Hover over each option to see more details
        
    3. **Process Image**
        * Click the "Process and Detect" button
        * Wait for processing to complete
        
    4. **View Results**
        * Switch to the Results tab
        * View detections and statistics
        * Download results as needed
        
    5. **Tips**
        * Higher resolution images may take longer to process
        * Adjust enhancement settings for better detection
        * Use the statistics to analyze results
    """)