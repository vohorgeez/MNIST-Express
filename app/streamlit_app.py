import os
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas

from mnist_express.config import Settings
from mnist_express.preprocessing import preprocess_user_drawing, transform_inference
from mnist_express.inference import predict_knn, extract_prediction_summary

MODEL_PATH = "artifacts/models/model_knn_best.joblib"
PREPROCESSOR_PATH = "artifacts/models/preprocessor.joblib"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)

    preprocessor = None
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor

def plot_preprocessed_image(img_28x28: np.ndarray):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_28x28, cmap="gray")
    ax.set_title("Dessin prétraité (28x28)")
    ax.axis("off")
    return fig

def is_canvas_empty(img: np.ndarray) -> bool:
    return img is None or np.max(img) == 0

def extract_grayscale_from_canvas(canvas_image: np.ndarray) -> np.ndarray:
    """
    canvas_image shape expected: (H, W, 4) RGBA
    We keep only one channel because the drawing is white on black.
    """
    if canvas_image is None:
        return np.zeros((280, 280), dtype=np.uint8)
    
    if canvas_image.ndim != 3 or canvas_image.shape[2] < 3:
        raise ValueError("Canvas image must be RGBA or RGB.")
    
    grayscale = canvas_image[:, :, 0]
    return grayscale.astype(np.uint8)

def main():
    st.set_page_config(page_title="MNIST Express", layout="wide")
    st.title("MNIST Express v4")
    st.write("Dessine un chiffre, puis lance la prédiction.")

    settings = Settings()
    model, preprocessor = load_artifacts()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Canvas")
        stroke_width = st.slider("Epaisseur du trait", min_value=8, max_value=30, value=18)

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF"
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_canvas"
        )

        predict_clicked = st.button("Prédire", type="primary")

    with col_right:
        st.subheader("Résultat")

        if predict_clicked:
            if canvas_result.image_data is None:
                st.warning("Le canvas est vide.")
                return
            
            gray = extract_grayscale_from_canvas(canvas_result.image_data)

            if is_canvas_empty(gray):
                st.warning("Le canvas est vide.")
                return
            
            preprocessed_img = preprocess_user_drawing(gray)

            if preprocessor is not None:
                X_input = transform_inference(preprocessor, preprocessed_img)
            else:
                X_input = preprocessed_img.reshape(1, 784).astype(np.float64)

            predictions, probabilities, predict_duration = predict_knn(
                model=model,
                X=X_input,
                settings=settings,
            )

            summary = extract_prediction_summary(
                model=model,
                predictions=predictions,
                probabilities=probabilities,
                sample_index=0,
                top_k=3,
            )

            st.metric("Prédiction", str(summary["predicted_class"]))
            st.metric("Confiance k-NN", f"{summary['confidence'] * 100:.1f}%")
            st.metric("Temps d'inférence", f"{predict_duration * 1000:.2f} ms")

            st.write("Top 3")
            for digit, score in summary["top_k"]:
                st.write(f"- {digit} : {score * 100:.1f}%")

            fig = plot_preprocessed_image(preprocessed_img)
            st.pyplot(fig)

            st.write("Vecteur d'entrée")
            st.caption(f"Shape envoyée au modèle : {X_input.shape}")

if __name__ == "__main__":
    main()