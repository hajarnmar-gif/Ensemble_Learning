import streamlit as st
import numpy as np
import joblib
import cv2

from PIL import Image
from skimage import color
from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

# =====================================
# 1. CONFIGURATION DE L'APPLICATION
# =====================================

st.set_page_config(
    page_title="Classification Voiture / Bateau",
    page_icon=" ",
    layout="centered"
)

st.title("  Classification des images")
st.subheader("Reconnaissance des voitures et des bateaux")
st.write(
    "Cette application utilise le modèle Boosting "
    "entraîné pour reconnaître les images de voitures et de bateaux."
)

# =====================================
# 2. CHARGEMENT DU MODELE
# =====================================

@st.cache_resource
def load_model():
    return joblib.load("boosting_model.joblib")

try:
    model = load_model()
    st.success("Modèle Boosting chargé avec succès.")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()


# =====================================
# 3. PREPARATION DE L'IMAGE
# =====================================

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((32, 32))

    img = np.array(image, dtype=np.float32) / 255.0

    return img


# =====================================
# 4. EXTRACTION DES CARACTERISTIQUES RGB
# =====================================

def extract_rgb_features(img):
    features = []

    for channel in range(3):
        channel_data = img[:, :, channel]

        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))

    return np.array(features, dtype=np.float32)


# =====================================
# 5. EXTRACTION DES CARACTERISTIQUES HSV
# =====================================

def extract_hsv_features(img):
    hsv_img = color.rgb2hsv(img)

    hist, _ = np.histogramdd(
        hsv_img.reshape(-1, 3),
        bins=(8, 2, 2),
        range=((0, 1), (0, 1), (0, 1))
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum()

    return hist.flatten()


# =====================================
# 6. EXTRACTION DES CARACTERISTIQUES GLCM
# =====================================

def extract_glcm_features(img):
    gray = color.rgb2gray(img)

    gray_uint8 = (gray * 255).astype(np.uint8)

    glcm = graycomatrix(
        gray_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ]

    return np.array(features, dtype=np.float32)


# =====================================
# 7. EXTRACTION DES CARACTERISTIQUES LBP
# =====================================

def extract_lbp_features(img, P=8, R=1):
    gray = color.rgb2gray(img)

    gray_uint8 = (gray * 255).astype(np.uint8)

    lbp = local_binary_pattern(
        gray_uint8,
        P,
        R,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, P + 3),
        range=(0, P + 2)
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum()

    return hist


# =====================================
# 8. EXTRACTION DES CARACTERISTIQUES HU
# =====================================

def extract_hu_features(img):
    gray = color.rgb2gray(img)

    gray_uint8 = (gray * 255).astype(np.uint8)

    moments = cv2.moments(gray_uint8)

    hu = cv2.HuMoments(moments).flatten()

    hu = -np.sign(hu) * np.log10(
        np.abs(hu) + 1e-30
    )

    return hu.astype(np.float32)


# =====================================
# 9. FUSION DES CINQ DESCRIPTEURS
# =====================================

def extract_all_features(img):

    rgb = extract_rgb_features(img)
    hsv = extract_hsv_features(img)
    glcm = extract_glcm_features(img)
    lbp = extract_lbp_features(img)
    hu = extract_hu_features(img)

    features = np.concatenate([
        rgb,
        hsv,
        glcm,
        lbp,
        hu
    ])

    return features.reshape(1, -1)


# =====================================
# 10. INTERFACE DE PREDICTION
# =====================================

st.divider()

st.header("Importer une image")

uploaded_file = st.file_uploader(
    "Choisir une image de voiture ou de bateau",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Image importée",
        use_container_width=True
    )

    if st.button("🔍 Prédire", type="primary"):

        with st.spinner("Analyse de l'image en cours..."):

            try:
                # Préparation de l'image
                img = prepare_image(image)

                # Extraction des caractéristiques
                features = extract_all_features(img)

                # Vérification du nombre de caractéristiques
                if features.shape[1] != 59:
                    st.error(
                        f"Nombre de caractéristiques incorrect : "
                        f"{features.shape[1]} au lieu de 59."
                    )
                    st.stop()

                # Prédiction
                prediction = int(model.predict(features)[0])

                class_names = {
                    0: "Voiture 🚗",
                    1: "Bateau 🚢"
                }

                st.divider()

                st.subheader("Résultat de la classification")

                if prediction == 0:
                    st.success(
                        f"Classe prédite : {class_names[prediction]}"
                    )
                else:
                    st.success(
                        f"Classe prédite : {class_names[prediction]}"
                    )

                # Probabilités si disponibles
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(features)[0]

                    st.write("Probabilités estimées :")

                    for class_id, probability in zip(
                        model.classes_,
                        probabilities
                    ):
                        class_name = class_names[int(class_id)]

                        st.write(
                            f"{class_name} : {probability * 100:.2f} %"
                        )

                        st.progress(float(probability))

            except Exception as e:
                st.error(f"Erreur pendant la prédiction : {e}")

st.divider()

st.caption(
    "Projet de classification supervisée : "
    "Voitures et Bateaux | Méthode : Boosting"
)