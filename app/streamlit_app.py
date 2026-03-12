import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from mnist_express.config import Settings
from mnist_express.inference import extract_prediction_summary, predict_knn
from mnist_express.monitoring import (
    load_usage_stats,
    record_prediction,
    record_user_feedback,
)
from mnist_express.preprocessing import preprocess_user_drawing

MODEL_PATH = "artifacts/models/model_knn_best.joblib"

logger = logging.getLogger(__name__)

def setup_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("model_loaded | path=%s", MODEL_PATH)
    return model

def extract_grayscale_from_canvas(canvas_image: np.ndarray) -> np.ndarray:
    """
    Convert RGBA canvas image to grayscale-like 2D image.
    The drawing is white on black, so one RGB channel is enough.
    """
    if canvas_image is None:
        return np.zeros((280, 280), dtype=np.uint8)
    
    if canvas_image.ndim != 3 or canvas_image.shape[2] < 3:
        raise ValueError("Canvas image must be RGB/RGBA")
    
    gray = canvas_image[:, :, 0]
    return gray.astype(np.uint8)

def is_canvas_empty(img: np.ndarray) -> bool:
    return img is None or np.max(img) == 0

def plot_preprocessed_image(img_28x28: np.ndarray):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_28x28, cmap="gray")
    ax.set_title("Dessin prétraité (28x28)")
    ax.axis("off")
    return fig

def get_expected_n_features(model) -> int | None:
    if hasattr(model, "named_steps"):
        if "knn" in model.named_steps and hasattr(model.named_steps["knn"], "n_features_in_"):
            return model.named_steps["knn"].n_feaatures_in_
        
        for _, step, in reversed(model.steps):
            if hasattr(step, "n_features_in_"):
                return step.n_features_in_
            
    if hasattr(model, "n_features_in_"):
        return model.n_feautures_in_
    
    return None

def init_session_state() -> None:
    if "last_prediction_done" not in st.session_state:
        st.session_state.last_prediction_done = False

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    if "last_summary" not in st.session_state:
        st.session_state.last_summary = None

    if "last_preprocessed_img" not in st.session_state:
        st.session_state.last_preprocessed_img = None

    if "last_predict_duration_ms" not in st.session_state:
        st.session_state.last_predict_duration_ms = None

    if "last_input_shape" not in st.session_state:
        st.session_state.last_input_shape = None

def render_monitoring_metrics(settings: Settings) -> None:
    stats = load_usage_stats(settings)

    col1, col2, col3 = st.columns(3)
    col1.metric("Usages totaux", stats.total_predictions)
    col2.metric("Feedbacks reçus", stats.total_feedback)
    col3.metric("Taux d'erreur utilisateur", f"{stats.user_error_rate * 100:.1f}%")

    st.caption(
        f"Temps moyen d'inférence: {stats.avg_inference_time_ms:.2f} ms"
        + (
            f" | Dernière prédiction: {stats.last_prediction_at}"
            if stats.last_prediction_at
            else ""
        )
    )

def handle_prediction(model, settings: Settings, canvas_result) -> None:
    if canvas_result.image_data is None:
        logger.warning("prediction_skipped | reason=no_canvas_data")
        st.warning("Le canvas est vide.")
        return
    
    gray = extract_grayscale_from_canvas(canvas_result.image_data)

    if is_canvas_empty(gray):
        logger.warning("prediction_skipped | reason=empty_canvas")
        st.warning("Le canvas est vide.")
        return
    
    preprocessed_img = preprocess_user_drawing(gray)
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

    predict_duration_ms = predict_duration * 1000.0

    if settings.enable_monitoring:
        record_prediction(settings=settings, inference_time_ms=predict_duration_ms)
    
    st.session_state.last_prediction_done = True
    st.session_state.feedback_submitted = False
    st.session_state.last_summary = summary
    st.session_state.last_preprocessed_img = preprocessed_img
    st.session_state.last_predict_duration_ms = predict_duration_ms
    st.session_state.last_input_shape = X_input.shape

    logger.info(
        "prediction_completed | predicted_class=%d | confidence=%.4f | inference_time_ms=%.2f",
        summary["predicted_class"],
        summary["confidence"],
        predict_duration_ms,
    )

def render_prediction_result(settings: Settings) -> None:
    summary = st.session_state.last_summary
    preprocessed_img = st.session_state.last_preprocessed_img
    predict_duration_ms = st.session_state.last_predict_duration_ms
    input_shape = st.session_state.last_input_shape

    if summary is None or preprocessed_img is None or predict_duration_ms is None:
        return
    
    st.metric("Prédiction", str(summary["predicted_class"]))
    st.metric("Confiance k-NN", f"{summary['confidence'] * 100:.1f}%")
    st.metric("Temps d'inférence", f"{predict_duration_ms:.2f} ms")

    st.write("Top 3")
    for digit, score in summary["top_k"]:
        st.write(f"- {digit} : {score * 100:.1f}%")

    fig = plot_preprocessed_image(preprocessed_img)
    st.pyplot(fig)

    st.caption(f"Shape envoyée au modèle: {input_shape}")

    if settings.enable_monitoring and settings.enable_user_feedback:
        st.divider()
        st.write("Ce résultat était-il correct ?")

        col_ok, col_ko = st.columns(2)

        with col_ok:
            ok_clicked = st.button(
                "Prédiction correcte",
                disabled=st.session_state.feedback_submitted,
                use_container_width=True,
            )

        with col_ko:
            ko_clicked = st.button(
                "Prédiction incorrecte",
                disabled=st.session_state.feedback_submitted,
                use_container_width=True,
            )

        if ok_clicked:
            record_user_feedback(settings=settings, is_error=False)
            st.session_state.feedback_submitted = True
            logger.info("feedback_recorded | is_error=False")
            st.success("Feedback enregistré : prédiction correcte.")

        if ko_clicked:
            record_user_feedback(settings=settings, is_error=True)
            st.session_state.feedback_submitted = True
            logger.info("feedback_recorded | is_error=True")
            st.error("Feedback enregistré : prédiction incorrecte.")

def main():
    st.set_page_config(page_title="MNIST Express", layout="wide")

    settings = Settings()
    setup_logging(settings)
    init_session_state()

    logger.info("app_started")

    st.title("MNIST Express v4")
    st.write("Dessine un chiffre, puis lance la prédiction.")

    model = load_model()
    expected_n_features = get_expected_n_features(model)

    st.caption(f"Modèle chargé : {expected_n_features} features")

    with st.expander("Instrumentation", expanded=True):
        render_monitoring_metrics(settings)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Canvas")

        stroke_width = st.slider(
            "Epaisseur du trait",
            min_value=8,
            max_value=30,
            value=18,
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_canvas",
        )

        predict_clicked = st.button("Prédire", type="primary")

    with col_right:
        st.subheader("Résultat")

        if predict_clicked:
            handle_prediction(model=model, settings=settings, canvas_result=canvas_result)

        render_prediction_result(settings)

if __name__ == "__main__":
    main()