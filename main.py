import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# Desactivar GPU en TensorFlow
tf.config.set_visible_devices([], 'GPU')

# --- Cargar el modelo ---
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# --- Clases (nombres de los integrantes de Talacha TecNM) ---
CLASSES = ['alejandro', 'fernando_de_jesus', 'jeizer_oswaldo', 'jothan_kaleb', 'rodrigo']

# --- Función para preprocesar la imagen ---
def preprocess_image(image):
    # Verificamos el tamaño esperado por el modelo
    input_shape = model.input_shape[1:3]
    image = image.resize(input_shape)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Configuración de la página ---
st.set_page_config(layout="wide")
st.title("Reconocimiento Facial - Talacha TecNM")
st.divider()

# --- División en columnas ---
col1, col2 = st.columns([1, 2], gap="large")

# --- Información y método de captura ---
with col1:
    st.write("Bienvenido integrante de Talacha TecNM, por favor elige un método para procesar tu imagen:")

    # Lista desplegable para elegir el método
    metodo = st.selectbox("Selecciona el método de captura:", ["Subir una imagen", "Tomar una foto con la cámara"])

    # Dependiendo del método seleccionado
    if metodo == "Subir una imagen":
        imagen = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    elif metodo == "Tomar una foto con la cámara":
        imagen = st.camera_input("Toma una imagen")

# --- Proceso de predicción ---
with col2:
    if imagen:
        st.image(imagen, caption="Imagen capturada", use_container_width=True)
        st.divider()

        with st.spinner("Procesando imagen..."):
            img = Image.open(imagen)
            img = preprocess_image(img)

            # Hacemos la predicción
            prediccion = model.predict(img)
            indice = np.argmax(prediccion)
            nombre_identificado = CLASSES[indice]
            confianza = prediccion[0][indice] * 100

            # Debugging: visualización de las predicciones
            st.write("**Predicciones detalladas:**")
            for i, clase in enumerate(CLASSES):
                st.write(f"{clase}: {prediccion[0][i] * 100:.2f}%")

            time.sleep(1)
        
        # Mostramos el resultado final
        st.success(f"¡Persona identificada: {nombre_identificado}!")
