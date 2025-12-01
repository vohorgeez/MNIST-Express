import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from streamlit_drawable_canvas import st_canvas

def preprocess_canvas_image(image):
    """
    image : array H x W x 4 (RGBA) venant du canvas
    retourne : array 1 x 64 prêt pour scaler.transform(...)
    """
    # 1. passer en niveaux de gris (on prend un canal)
    img = image[:, :, 0]

    # 2. inversion car dans load_digits, traits = clair, fond = sombre
    img = 255 - img

    # 3. resize vers 8x8 avec PIL
    pil = Image.fromarray(img.astype(np.uint8)).resize((8, 8), Image.BICUBIC)

    # 4. numpy + float
    arr = np.array(pil).astype(np.float32)

    # 5. mise à l'échelle 0-16 comme load_digits
    arr = arr * (16.0 / 255.0)

    # 6. flatten en vecteur 1 x 64
    arr = arr.reshape(1, -1)

    return arr

@st.cache_resource
def train_model(k: int, metric: str, weights: str):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    clf.fit(X_train, y_train)
    return clf, scaler, digits

st.set_page_config(page_title="MNIST Express", page_icon="🧠")

st.title("MNIST Express - k-NN digits classifier")
k = st.slider("Number of neighbors", min_value=1, max_value=15, value=5, step=2)

metric = st.selectbox(
    "Distance",
    options=["euclidean", "manhattan", "chebyshev"],
    index=0,
)

weights = st.selectbox(
    "Poids",
    options=["uniform", "distance"],
    index=1, # par défaut "distance"
)

clf, scaler, digits = train_model(k, metric, weights)

st.subheader("Data sample")
cols = st.columns(6)
for col, (image, label) in zip(cols, zip(digits.images[:6], digits.target[:6])):
    col.image(image, clamp=True, caption=f"Label {label}", width=80)

st.markdown("---")
st.subheader("Dessine un chiffre")
canvas = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=196,
    width=196,
    drawing_mode="freedraw",
    key="canvas",
)

top_k = st.slider("Afficher les top-k classes", min_value=1, max_value=10, value=3)

if canvas.image_data is not None:
    sample = preprocess_canvas_image(canvas.image_data)
    sample = scaler.transform(sample)

    probas = clf.predict_proba(sample)[0]
    pred = int(np.argmax(probas))

    st.write(f"**Prédiction : {pred}**")

    top_indices = np.argsort(probas)[::-1][:top_k]
    top_values = probas[top_indices]

    for cls, p in zip(top_indices, top_values):
        st.write(f"Classe {cls} : {p:.3f}")

    if st.checkbox("Voir la version 8x8 utilisée par le modèle"):
        fig, ax = plt.subplots()
        ax.imshow(sample.reshape(8, 8), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)