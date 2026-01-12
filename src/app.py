import sys
from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Third-party imports for Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Ensure local imports work
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from model import PneumoniaResNet
from transforms import get_transforms

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Pneumonia AI Diagnostic Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
TRANSFORM_VALID = get_transforms(stage='valid')

# --- Helper Functions ---

@st.cache_resource
def load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = PneumoniaResNet(num_classes=2, pretrained=False)
    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found at: {checkpoint_path}")
        st.stop()
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Robust Dictionary Loading
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict(model: torch.nn.Module, image: Image.Image) -> tuple[int, float, torch.Tensor]:
    """
    Returns: (predicted_class, probability_pneumonia, preprocessed_tensor)
    """
    img_tensor = TRANSFORM_VALID(image.convert('RGB'))
    input_batch = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_batch)
        probs = F.softmax(logits, dim=1)
        
    pneumonia_prob = probs[0, 1].item()
    pred_class = torch.argmax(probs, dim=1).item()
    
    return pred_class, pneumonia_prob, input_batch

def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor, target_layer):
    """
    Generates Grad-CAM heatmap.
    """
    # Create GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # We want to see what parts of the image contribute to the 'Pneumonia' class (Class 1)
    # If the model predicts Normal, we can still ask "Why did you think it might be Pneumonia?" or "Why Normal?"
    # Here we default to the predicted class or force Class 1 (Pneumonia) to see signs of disease.
    # Let's visualize the target class '1' (Pneumonia) to see disease indicators.
    targets = [ClassifierOutputTarget(1)]

    # Generate CAM
    # You can pass use_cuda=True if on GPU
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    # In this package, result is [Batch, H, W]
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

# --- Layout ---

def sidebar_analytics():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Clinic Daily Stats")
    
    # Dummy Data
    labels = ['Normal', 'Pneumonia']
    sizes = [3, 2]
    colors = ['#66b3ff', '#ff9999']

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.axis('equal')  
    
    # Make background transparent for streamlit
    fig.patch.set_alpha(0)
    st.sidebar.pyplot(fig)
    st.sidebar.caption("Total Scans Today: 5")

def main():
    st.title("🩺 AI Diagnostic Dashboard")
    st.markdown("Professional Grade Pneumonia Detection System")

    # --- Sidebar ---
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05)
    show_heatmap = st.sidebar.checkbox("Show AI Attention (Grad-CAM)", value=False)
    
    sidebar_analytics()

    # --- Main Content ---
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Patient X-Ray (DICOM/JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    model = load_model(CHECKPOINT_PATH)
    
    # Identify the target layer for Grad-CAM (usually the last convolutional layer)
    # For ResNet18, it's typically model.backbone.layer4[-1]
    target_layer = model.backbone.layer4[-1]

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Patient Scan")
            image_pil = Image.open(uploaded_file).convert('RGB')
            st.image(image_pil, caption="Original X-Ray", use_column_width=True)

        # Run Analysis
        with st.spinner("Processing image..."):
            pred_class, prob, img_tensor = predict(model, image_pil)

        with col2:
            st.subheader("Analysis Results")
            
            # 1. Visualization of Probability
            st.write("**Probability Distribution**")
            
            # Create a progress bar style visualization
            # Pneumonia is red, Normal is green
            
            # Probability of Pneumonia
            p_score = prob
            
            bar_color = "red" if p_score > confidence_threshold else "green"
            
            st.progress(p_score)
            st.caption(f"Pneumonia Probability: {p_score:.1%}")
            
            # 2. Verdict Box
            st.markdown("---")
            if p_score > confidence_threshold:
                st.error(f"### ⚠️ PNEUMONIA DETECTED")
                st.markdown(f"The model is **{p_score:.1%}** confident that this scan shows signs of Pneumonia.")
            else:
                st.success(f"### ✅ NORMAL")
                st.markdown(f"The model is **{1-p_score:.1%}** confident that this scan is Normal.")

            # 3. Grad-CAM Visualization
            if show_heatmap:
                st.markdown("---")
                st.subheader("AI Attention Map")
                
                # Generate CAM
                cam_mask = generate_gradcam(model, img_tensor, target_layer)
                
                # Prepare image for overlay
                # Resize original image to 224x224 (training size) to match CAM or resize CAM to original
                # Easier: Resize image to match model input resolution for visualization
                img_resized = np.array(image_pil.resize((224, 224)))
                img_normalized = np.float32(img_resized) / 255.0
                
                # Create visualization
                visualization = show_cam_on_image(img_normalized, cam_mask, use_rgb=True)
                
                st.image(visualization, caption="Red regions indicate areas of high interest", use_container_width=True)

if __name__ == "__main__":
    main()
