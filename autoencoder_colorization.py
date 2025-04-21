# === IMPORT LIBRARY ===
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab, lab2rgb

# === KONFIGURASI DATASET ===
color_folder = "dataset/color"
output_folder = "output/predicted"
os.makedirs(output_folder, exist_ok=True)

# Fungsi load dan konversi RGB -> LAB

def load_images_rgb_lab(folder, size=(256, 256), max_images=200):
    images_l = []
    images_ab = []
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if i >= max_images:
            break
        path = os.path.join(folder, filename)
        img = Image.open(path).convert('RGB').resize(size)
        img_np = np.asarray(img) / 255.0
        img_lab = rgb2lab(img_np).astype("float32")

        L = img_lab[:, :, 0] / 100.0               # Normalisasi L
        ab = (img_lab[:, :, 1:] + 128) / 255.0     # Normalisasi a,b

        images_l.append(L.reshape(size[0], size[1], 1))
        images_ab.append(ab)

    return np.array(images_l), np.array(images_ab)

# Load dataset
x_l, y_ab = load_images_rgb_lab(color_folder)

print("L shape:", x_l.shape, "AB shape:", y_ab.shape)

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_l, y_ab, test_size=0.1, random_state=42)

# === DEFINISI MODEL AUTOENCODER ===
input_layer = Input(shape=(256, 256, 1))

# Encoder (deeper)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

output_layer = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)

model = Model(input_layer, output_layer)
model.compile(optimizer=Adam(1e-4), loss='mse')
model.summary()

# === TRAINING ===
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=50,
                    batch_size=8,
                    shuffle=True)

# === VISUALISASI HASIL ===
preds_ab = model.predict(x_val[:10])

plt.figure(figsize=(20, 6))
for i in range(10):
    L = x_val[i].reshape(256, 256, 1) * 100  # Kembali ke skala 0-100
    ab = preds_ab[i] * 255 - 128             # Kembali ke skala -128 hingga 127
    lab_image = np.concatenate((L, ab), axis=-1)
    rgb_image = lab2rgb(lab_image)

    # Simpan gambar hasil prediksi
    save_path = os.path.join(output_folder, f"predicted_{i}.png")
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    Image.fromarray(rgb_uint8).save(save_path)

    # Input
    ax = plt.subplot(3, 10, i + 1)
    plt.imshow(x_val[i].reshape(256, 256), cmap='gray')
    plt.axis("off")
    plt.title("Input")

    # Prediksi
    ax = plt.subplot(3, 10, i + 1 + 10)
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.title("Predicted")

    # Target
    ab_true = y_val[i] * 255 - 128
    lab_true = np.concatenate((L, ab_true), axis=-1)
    rgb_true = lab2rgb(lab_true)
    ax = plt.subplot(3, 10, i + 1 + 20)
    plt.imshow(rgb_true)
    plt.axis("off")
    plt.title("Target")

plt.tight_layout()
plt.show()

# === VISUALISASI LOSS ===
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
