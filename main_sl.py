import os
import math

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st 

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Activation,
    Dense,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Reshape,
    LeakyReLU,
    BatchNormalization,
)
# Titulo
st.set_page_config(layout="centered", page_title="ACGAN MNIST Generator")
st.title("Generador de Dígitos MNIST con ACGAN")
st.write("Selecciona un dígito para generar imagenes.")

@st.cache_resource # Cargar el modelo solo una vez y cachéarlo
def load_acgan_model():
    model_path = "acgan_mnist.h5"
    if not os.path.exists(model_path):
        st.error(f"Error: El archivo del modelo '{model_path}' no se encontró. Revisar que esté en la misma carpeta que el main.")
        st.stop() # Parar la app si no se encuentra el archivo
    return load_model(model_path)

trained_generator = load_acgan_model()
latent_size = 100
num_classes = 10

# --- Streamlit Sidebar -----
st.sidebar.header("Opciones de Generación")
selected_label = st.sidebar.slider(
    "Selecciona el dígito a generar:",
    min_value=0,
    max_value=9,
    value=0, # Valor por defecto
    step=1
)

# --- Funcion plot adaptada para Streamlit ---
def generate_and_plot_acgan(generator, latent_size, num_classes, class_label):
    st.subheader(f"Generando imágenes para el dígito: {class_label}")

    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_class_labels = np.ones(16, dtype='int32') * class_label

    # Generator expects labels with shape (batch_size, 1)
    noise_class_input = noise_class_labels.reshape(-1, 1)
    
    # Predict images -  st.spinner, es un spinner para indicar que se están generando las imágenes
    with st.spinner("Generando imágenes..."):
        images = generator.predict([noise_input, noise_class_input], verbose=0) # verbose=0 tpara no mostrar la salida de keras

    fig, ax = plt.subplots(figsize=(8, 8))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(num_images))
    cols = int(math.ceil(num_images / rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {noise_class_labels[i]}")
        plt.axis('off')

    plt.tight_layout() # Ajustar espaciado
    
    st.pyplot(fig) # Mostrar la figura en Streamlit
    plt.close(fig) 

# Correr el modelo cuando se cargue la app o cambie el valor del slider
if st.sidebar.button("Generar Imágenes") or st.session_state.get('initial_run', True):
    generate_and_plot_acgan(trained_generator, latent_size, num_classes, selected_label)
    st.session_state['initial_run'] = False # Set to False after first run

st.markdown("---")
st.write("Esta aplicación utiliza un modelo ACGAN pre-entrenado para generar dígitos escritos a mano basados en un número de entrada.")
