import numpy as np
import math
from numpy.linalg import eig
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st


def SVD(A):
    ATA = np.dot(A.T, A)

    eigenvalues, V = np.linalg.eig(ATA)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    eigenvalues[eigenvalues < 0] = 0

    Sigma = np.sqrt(np.diag(eigenvalues))

    U = np.zeros((A.shape[0], len(eigenvalues)))
    for i in range(len(eigenvalues)):
        if eigenvalues[i] != 0:
            U[:, i] = (1 / np.sqrt(eigenvalues[i])) * np.dot(A, V[:, i])

    return U, Sigma, V

def singular_values(A):
    
    ATA = np.dot(A.T, A)

    eigenvalues, V = np.linalg.eig(ATA)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvalues[eigenvalues < 0] = 0

    singular = np.sqrt(eigenvalues)
    return singular

def rank_k_approximation(A, k):
    U, Sigma, V = SVD(A)

    approx_U = U[:, :k]
    approx_Sigma = Sigma[:k, :k]
    approx_Vt = V.T[:k, :]

    return approx_U,approx_Sigma,approx_Vt


def compress_image(image_array, k):
    U, Sigma, Vt = rank_k_approximation(image_array, k)

    compressed_image = np.dot(U, np.dot(Sigma, Vt))

    return compressed_image

def compress_color_image(image_array, k):
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    red_compressed = compress_image(red_channel, k)
    green_compressed = compress_image(green_channel, k)
    blue_compressed = compress_image(blue_channel, k)

    compressed_image = np.stack((red_compressed, green_compressed, blue_compressed), axis=-1)

    return compressed_image


def main():
    st.title('Image Compression with SVD')
    image_width=200
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        image_array = np.array(image)

        image_array = image_array - np.min(image_array)
        image_array = image_array / np.max(image_array) * 255

        k = st.slider('Select compression factor (k)', min_value=1, max_value=max(image_array.shape), value=50)

        compressed_image = compress_color_image(image_array, k)

        compressed_image = Image.fromarray(compressed_image.astype('uint8'))

        with col1:
            st.image(image, caption='Original Image', width=image_width)

        with col2:
            st.image(compressed_image, caption='Compressed Image', width=image_width)

        red_channel = image_array[:, :, 0]
        green_channel = image_array[:, :, 1]
        blue_channel = image_array[:, :, 2]

        red_sing_vals = singular_values(red_channel)
        green_sing_vals = singular_values(green_channel)
        blue_sing_vals = singular_values(blue_channel)

        red_max=max(red_sing_vals)
        green_max=max(green_sing_vals)
        blue_max=max(blue_sing_vals)

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        axes[0].bar(np.arange(len(red_sing_vals)), red_sing_vals, color='r', label='Singular Values')
        axes[0].axvline(x=k, ymin=0, ymax=max(red_sing_vals), color='k', linestyle='--')
        axes[0].text(k + 1, red_max / 2, f'k = {k}', color='k', fontsize=10, rotation=90, ha='left', va='center')
        axes[0].set_title('Red Channel Singular Values')

        axes[1].bar(np.arange(len(green_sing_vals)), green_sing_vals, color='g', label='Singular Values')
        axes[1].axvline(x=k, ymin=0, ymax=max(red_sing_vals), color='k', linestyle='--')
        axes[1].text(k + 1, green_max / 2, f'k = {k}', color='k', fontsize=10, rotation=90, ha='left', va='center')
        axes[1].set_title('Green Channel Singular Values')

        axes[2].bar(np.arange(len(blue_sing_vals)), blue_sing_vals, color='b', label='Singular Values')
        axes[2].axvline(x=k, ymin=0, ymax=max(red_sing_vals), color='k', linestyle='--')
        axes[2].text(k + 1, blue_max / 2, f'k = {k}', color='k', fontsize=10, rotation=90, ha='left', va='center')
        axes[2].set_title('Blue Channel Singular Values')

        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

