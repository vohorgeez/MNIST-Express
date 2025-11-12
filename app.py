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

@st.cache_resource
def train_model(k: int):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(X_train, y_train)
    return clf, scaler, digits

st.set_page_config(page_title="MNIST Express", page_icon="🧠")

st.title("MNIST Express - k-NN digits classifier")
k = st.slider("Number of neighbors", min_value=1, max_value=15, value=5, step=2)

clf, scaler, digits = train_model(k)

st.subheader("Datasimple sample")
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

if canvas.image_data is not None:
    pil_img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8)).resize((8,8), Image.BICUBIC)
    sample = np.array(pil_img).reshape(1, -1)
    sample = scaler.transform(sample)
    pred = clf.predict(sample)[0]

    st.write(f"**Prédiction : {pred}***")

    if st.checkbox("Voir la version 8×8 utilisée par le modèle"):
        fig, ax = plt.subplots()
        ax.imshow(sample.reshape(8,8), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)