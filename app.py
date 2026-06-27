import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🩺 Breast Cancer Detection System from Mammogram")
st.markdown("---")


# ======================== LOAD MODEL ========================
@st.cache_resource
def load_models():
    try:
        # Dapatkan direktori tempat script dijalankan
        script_dir = os.path.dirname(os.path.abspath(__file__))
        unet_path = os.path.join(script_dir, "models", "unet_best_final.keras")
        cae_path  = os.path.join(script_dir, "models", "cae_best_final.keras")
        
        if not os.path.exists(unet_path) or not os.path.exists(cae_path):
            st.error("Model files not found in the specified folder!")
            st.info(f"Check path: {script_dir}")
            return None, None

        unet_model = tf.keras.models.load_model(unet_path, compile=False)
        cae_model  = tf.keras.models.load_model(cae_path, compile=False)

        st.success("✅ Models loaded successfully!")
        return unet_model, cae_model

    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None


def normalize_to_grayscale_2d(arr):
    """
    Mengonversi output model menjadi grayscale 2D (H, W) 
    untuk ditampilkan dengan st.image().
    """
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[-1] in [3, 4]:
            arr = np.mean(arr[:, :, :3], axis=-1)
        else:
            arr = np.mean(arr, axis=-1)
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")

    return np.clip(arr, 0, 1).astype(np.float32)


unet_model, cae_model = load_models()

# ======================== THRESHOLD (dari hasil evaluasi notebook) ========================
# Nilai ini didapat dari cell evaluasi: mean + std anomaly score pada data test
CAE_THRESHOLD = 1.263   # dalam persen (%)


# ======================== MODEL SELECTION ========================
model_choice = st.radio(
    "Select Model:",
    ["U-Net", "CAE", "Compare Both"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload Mammogram Image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None and unet_model is not None and cae_model is not None:

    image = Image.open(uploaded_file).convert('L')
    
    # ── FIX 1: Ukuran diubah dari 256 menjadi 128 ──────────────────
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_display = (img_array * 255).astype(np.uint8)  # Untuk visualisasi display
    
    # ── FIX 2: Dimensi tensor input disesuaikan menjadi (1, 128, 128, 1) ──
    input_tensor = np.expand_dims(img_array, axis=(0, -1))  

    # Menampilkan preview gambar yang diunggah
    st.subheader("Image Preview")
    col_prev, _ = st.columns([1, 2])
    with col_prev:
        st.image(img_display, caption="Mammogram (128x128 grayscale)", use_container_width=True)

    if st.button("🔍 Analyze Image", type="primary"):
        st.subheader("Analysis Results")

        # ======================== U-NET (SUDAH DIINDENTASI KEMBALI) ========================
        if model_choice in ["U-Net", "Compare Both"]:
            st.markdown("### 🔷 U-Net — Lesion Segmentation")
            with st.spinner("Processing with U-Net..."):
                pred_raw = unet_model.predict(input_tensor, verbose=0)[0]
                pred   = normalize_to_grayscale_2d(pred_raw)   
                
                # OPTIMASI 1: Lebih selektif mendeteksi lesi sejati
                binary = (pred > 0.75).astype(np.uint8)         

                overlay = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                contour_count = 0
                for cnt in contours:
                    # OPTIMASI 2: Blokir noise kotak kecil kosong di latar belakang hitam
                    if cv2.contourArea(cnt) > 100: 
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        cv2.putText(overlay, "Lesion", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        contour_count += 1

                col1, col2 = st.columns(2)
                with col1:
                    st.image(overlay, caption="U-Net + Bounding Box", use_container_width=True)
                with col2:
                    if contour_count > 0:
                        st.success("✅ **CANCER DETECTED**")
                        confidence = float(np.mean(pred) * 100)
                        st.metric("U-Net Confidence", f"{confidence:.2f}%")
                        st.info(f"📍 Number of Lesions Detected: {contour_count}")
                    else:
                        st.info("❌ **No Cancer Detected**")
                        st.success("No significant lesions found.")

        # ======================== CAE ========================
        if model_choice in ["CAE", "Compare Both"]:
            st.markdown("### 🔶 CAE — Anomaly Detection")
            with st.spinner("Processing with CAE..."):
                recon_raw = cae_model.predict(input_tensor, verbose=0)[0]
                recon     = normalize_to_grayscale_2d(recon_raw)   

                error         = np.abs(img_array - recon)          
                anomaly_score = float(np.mean(error) * 100)

                # Tampilkan skor anomali dan batas threshold
                col_score1, col_score2 = st.columns(2)
                with col_score1:
                    st.metric("CAE Anomaly Score", f"{anomaly_score:.4f}%")
                with col_score2:
                    st.metric("Threshold", f"{CAE_THRESHOLD:.4f}%",
                              help="Derived from evaluation: mean + std of test anomaly scores")

                # Progress bar kedekatan skor anomali ke threshold
                ratio = min(anomaly_score / CAE_THRESHOLD, 1.5)
                st.progress(min(ratio, 1.0), text=f"Anomaly level: {anomaly_score:.4f}% / {CAE_THRESHOLD:.4f}%")

                # Visualisasi tiga perbandingan gambar
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image((img_array * 255).astype(np.uint8), caption="Original", use_container_width=True)
                with col2:
                    st.image((recon * 255).astype(np.uint8), caption="CAE Reconstruction", use_container_width=True)
                with col3:
                    error_display = error / (error.max() + 1e-8)
                    st.image((error_display * 255).astype(np.uint8), caption="Error Map", use_container_width=True)

                # Hasil deteksi dari model CAE
                if anomaly_score > CAE_THRESHOLD:
                    st.success("✅ **ANOMALY DETECTED** (Possible Cancer)")
                    st.warning("⚠️ Anomaly score exceeds threshold — significant deviation from normal patterns detected.")
                else:
                    st.info("❌ **Normal**")
                    st.success("Anomaly score is below threshold — image matches normal patterns.")

        st.markdown("---")
        st.caption("⚠️ This result is intended as a diagnostic aid only. Always consult a specialist physician.")

elif uploaded_file is not None:
    st.warning("Models failed to load. Make sure TensorFlow and Keras are installed correctly.")
    st.code(
        "pip install tensorflow-cpu==2.16.1 numpy==1.26.4 keras==3.3.3",
        language="bash"
    )