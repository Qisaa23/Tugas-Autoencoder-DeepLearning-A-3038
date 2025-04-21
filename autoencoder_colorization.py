import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# Path ke folder dataset
gray_folder = "dataset/gray"
color_folder = "dataset/color"

# Fungsi load dan resize gambar
def load_images_from_folder(folder, size=(256, 256), grayscale=False, max_images=100):
    images = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if i >= max_images:
            break
        path = os.path.join(folder, filename)
        with Image.open(path) as img:
            img = img.convert('L') if grayscale else img.convert('RGB')
            img = img.resize(size)
            img_array = np.asarray(img) / 255.0
            images.append(img_array)
    return np.array(images)

# Load data
x_gray = load_images_from_folder(gray_folder, grayscale=True, max_images=200)
x_color = load_images_from_folder(color_folder, grayscale=False, max_images=200)


# Ubah grayscale ke shape: (n, 256, 256, 1)
x_gray = np.expand_dims(x_gray, axis=-1)

print(f"Grayscale shape: {x_gray.shape}, RGB shape: {x_color.shape}")

# === Model Autoencoder ===
input_img = Input(shape=(256, 256, 1))

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')
autoencoder.summary()

# Training
autoencoder.fit(x_gray, x_color,
                epochs=10,
                batch_size=8,
                shuffle=True,
                validation_split=0.1)

# === Visualisasi Hasil ===
decoded_imgs = autoencoder.predict(x_gray[:10])

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # Input grayscale
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_gray[i].reshape(256, 256), cmap='gray')
    plt.title("Input (gray)")
    plt.axis("off")

    # Prediksi autoencoder
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Output (predicted)")
    plt.axis("off")

    # Target asli (RGB)
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(x_color[i])
    plt.title("Target (color)")
    plt.axis("off")

plt.tight_layout()
plt.show()