"""Adaptateur primaire Streamlit.

Ce fichier ne fait que de l'interface : il traduit le canvas en image en niveaux de gris, appelle les cas d'usage fournis par `bootstrap.build_services` et affiche les résultats. Il ne charge aucun modèle et n'écrit aucun fichier.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import logging
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from mnist_express.bootstrap import AppServices, build_services
from mnist_express.config import Settings
from mnist_express.core.model import EmptyDrawingError, Prediction

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
def get_services() -> AppServices:
    return build_services(Settings())

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

def plot_preprocessed_image(img_28x28: np.ndarray):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_28x28, cmap="gray")
    ax.set_title("Dessin prétraité (28x28)")
    ax.axis("off")
    return fig

def init_session_state() -> None:
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

def render_monitoring_metrics(services: AppServices) -> None:
    stats = services.get_usage_stats.execute()

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

def handle_prediction(services: AppServices, canvas_result) -> None:
    if canvas_result.image_data is None:
        logger.warning("prediction_skipped | reason=no_canvas_data")
        st.warning("Le canvas est vide.")
        return
    
    gray = extract_grayscale_from_canvas(canvas_result.image_data)

    try:
        prediction = services.predict_digit.execute(gray)
    except EmptyDrawingError:
        st.warning("Le canvas est vide.")
        return
    st.session_state.last_prediction = prediction
    st.session_state.feedback_submitted = False

def render_prediction_result(services: AppServices) -> None:
    prediction: Prediction | None = st.session_state.last_prediction
    if prediction is None:
        return
    
    settings = services.settings

    st.metric("Prédiction", str(prediction.predicted_class))
    st.metric("Confiance k-NN", f"{prediction.confidence * 100:.1f}%")
    st.metric("Temps d'inférence", f"{prediction.inference_time_ms:.2f} ms")

    st.write("Top 3")
    for digit, score in prediction.top_k:
        st.write(f"- {digit} : {score * 100:.1f}%")

    fig = plot_preprocessed_image(prediction.preprocessed_image)
    st.pyplot(fig)

    st.caption(f"Shape envoyée au modèle: {prediction.input_shape}")

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
            services.record_feedback.execute(is_error=False)
            st.session_state.feedback_submitted = True
            st.success("Feedback enregistré : prédiction correcte.")

        if ko_clicked:
            services.record_feedback.execute(is_error=True)
            st.session_state.feedback_submitted = True
            st.error("Feedback enregistré : prédiction incorrecte.")

def main():
    st.set_page_config(page_title="MNIST Express", layout="wide")

    setup_logging(Settings())
    services = get_services()
    init_session_state()

    logger.info("app_started")

    st.title("MNIST Express v4")
    st.write("Dessine un chiffre, puis lance la prédiction.")

    st.caption(f"Modèle chargé : {services.model_n_features} features")

    with st.expander("Instrumentation", expanded=True):
        render_monitoring_metrics(services)

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
            handle_prediction(services=services, canvas_result=canvas_result)

        render_prediction_result(services)

if __name__ == "__main__":
    main()