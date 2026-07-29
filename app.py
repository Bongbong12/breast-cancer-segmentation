import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import time
import tensorflow as tf

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🩺 Mammogram Image Analysis Using Convolutional Autoencoder and U-Net")
st.markdown("---")


# ======================== LOAD MODEL ========================
@st.cache_resource
def load_models():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        unet_path = os.path.join(script_dir, "models", "unet_best_final.keras")
        cae_path = os.path.join(script_dir, "models", "cae_best_final.keras")

        if not os.path.exists(unet_path) or not os.path.exists(cae_path):
            st.error("Model files not found in the specified folder!")
            st.info(f"Check path: {script_dir}")
            return None, None

        unet_model = tf.keras.models.load_model(unet_path, compile=False)
        cae_model = tf.keras.models.load_model(cae_path, compile=False)

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


def run_unet(unet_model, img_array, input_tensor):
    st.markdown("### 🔷 U-Net — Lesion Segmentation")
    with st.spinner("Processing with U-Net..."):
        pred_raw = unet_model.predict(input_tensor, verbose=0)[0]
        pred = normalize_to_grayscale_2d(pred_raw)

        # Lebih selektif mendeteksi lesi sejati
        binary = (pred > 0.85).astype(np.uint8)

        overlay = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_count = 0
        for cnt in contours:
            # Blokir noise kotak kecil kosong di latar belakang hitam
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(overlay, "Lesion", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                contour_count += 1

        col1, col2 = st.columns(2)
        with col1:
            st.image(overlay, caption="U-Net + Bounding Box", use_container_width=True)
        with col2:
            if contour_count > 0:
                st.success("✅ **Suspicious Lesion Detected**")
                prediction_score = float(np.max(pred) * 100)
                st.metric("Prediction Score", f"{prediction_score:.2f}%")
                st.info(f"📍 Number of Lesions Detected: {contour_count}")
            else:
                st.info("❌ **No Lesion Region Detected**")
                st.success("No significant lesions found.")

    st.markdown("---")
    st.subheader("📊 Performance Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dice Score", "0.4186")
    with col2:
        st.metric("IoU", "0.2647")
    with col3:
        st.metric("Precision", "0.5254")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Recall", "0.3479")
    with col5:
        st.metric("F1 Score", "0.4186")
    with col6:
        st.metric("Accuracy", "0.8955")

    with st.expander("ℹ️ Model Details"):
        st.write("**Architecture** : U-Net")
        st.write("**Input Size** : 128 × 128")
        st.write("**Dataset** : CBIS-DDSM")
        st.write("**Loss Function** : Binary Crossentropy")
        st.write("**Optimizer** : Adam")
        st.write("**Output** : Segmentation Mask")

    st.markdown("---")
    st.subheader("📝 Analysis Summary")
    if contour_count > 0:
        st.success(
            "The uploaded mammogram image contains suspicious lesion regions detected by the "
            "U-Net model. The segmentation result is visualized using the predicted lesion mask "
            "and bounding box."
        )
    else:
        st.info("No suspicious lesion region was detected in the uploaded mammogram image.")


def run_cae(cae_model, img_array, input_tensor, threshold):
    st.markdown("### 🔶 CAE — Anomaly Detection")
    with st.spinner("Processing with CAE..."):
        recon_raw = cae_model.predict(input_tensor, verbose=0)[0]
        recon = normalize_to_grayscale_2d(recon_raw)

        error = np.abs(img_array - recon)
        anomaly_score = float(np.mean(error) * 100)

        col_score1, col_score2 = st.columns(2)
        with col_score1:
            st.metric("CAE Anomaly Score", f"{anomaly_score:.4f}%")
        with col_score2:
            st.metric(
                "Threshold",
                f"{threshold:.4f}%",
                help="Derived from evaluation: mean + std of test anomaly scores",
            )

        ratio = min(anomaly_score / threshold, 1.5)
        st.progress(min(ratio, 1.0), text=f"Anomaly level: {anomaly_score:.4f}% / {threshold:.4f}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image((img_array * 255).astype(np.uint8), caption="Original", use_container_width=True)
        with col2:
            st.image((recon * 255).astype(np.uint8), caption="CAE Reconstruction", use_container_width=True)
        with col3:
            error_display = error / (error.max() + 1e-8)
            st.image((error_display * 255).astype(np.uint8), caption="Error Map", use_container_width=True)
 is_above_threshold = anomaly_score > threshold

        if is_above_threshold:
             st.info("📌 **Reconstruction Difference Above Threshold**")
             st.warning(
        "The reconstruction score is above the predefined threshold, "
        "indicating a larger reconstruction difference between the original "
        "and reconstructed mammogram image."
            )
        else:
            st.success("✅ **Reconstruction Difference Within Threshold**")
            st.info(
        "The reconstruction score is within the predefined threshold, "
        "indicating that the reconstructed image is consistent with the "
        "reconstruction pattern learned by the model."
       )

        with st.expander("ℹ️ Model Details"):
            st.write("**Architecture** : Convolutional Autoencoder (CAE)")
            st.write("**Input Size** : 128 × 128")
            st.write("**Dataset** : CBIS-DDSM")
            st.write("**Loss Function** : Mean Squared Error (MSE)")
            st.write("**Optimizer** : Adam")
            st.write("**Output** : Reconstructed Image")

        st.markdown("---")
        st.subheader("📊 Performance Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Anomaly Score", "1.4779%")

        with col2:
            st.metric("Threshold", "1.7578%")

        with col3:
            st.metric("Evaluation", "Reconstruction-Based Analysis")

        if is_above_threshold:
            st.warning(
        "The reconstruction score exceeds the predefined threshold, "
        "indicating greater reconstruction differences in the analyzed "
        "mammogram image."
             )
        else:
            st.success(
        "The reconstruction score is below the predefined threshold, "
        "indicating that the reconstructed image follows the reconstruction "
        "pattern learned by the model."
    )

# ======================== MAIN APP ========================
unet_model, cae_model = load_models()

# Nilai ini didapat dari cell evaluasi: mean + std anomaly score pada data test
CAE_THRESHOLD = 1.7578  # dalam persen (%)

model_choice = st.radio(
    "Select Model:",
    ["U-Net", "CAE"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload Mammogram Image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    input_tensor = np.expand_dims(img_array, axis=(0, -1))

    if st.button("🔍 Analyze Image"):

        if unet_model is None or cae_model is None:
            st.warning("Models failed to load. Make sure TensorFlow and Keras are installed correctly.")
            st.code(
                "pip install tensorflow-cpu==2.16.1 numpy==1.26.4 keras==3.3.3",
                language="bash"
            )
        else:
            progress = st.progress(0, text="Initializing analysis...")
            time.sleep(0.2)
            progress.progress(20, text="Loading model...")
            time.sleep(0.2)
            progress.progress(40, text="Preprocessing image...")
            time.sleep(0.2)
            progress.progress(70, text="Running AI prediction...")
            time.sleep(0.2)
            progress.progress(90, text="Generating visualization...")
            time.sleep(0.2)
            progress.progress(100, text="Analysis completed!")

            st.success("✅ Analysis completed successfully!")
            st.subheader("Analysis Results")

            if model_choice == "U-Net":
                run_unet(unet_model, img_array, input_tensor)
            else:
                run_cae(cae_model, img_array, input_tensor, CAE_THRESHOLD)

st.markdown("---")
st.caption(
    "⚠️ This application is developed as a research prototype. "
    "The analysis results are generated by the trained deep learning models "
    "and should be used for research and educational purposes only. "
    "They are not intended to support clinical diagnosis or medical decision-making."
)

