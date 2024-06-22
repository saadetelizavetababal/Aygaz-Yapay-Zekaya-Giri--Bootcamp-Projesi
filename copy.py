import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Veri setini yükleyin
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Veri seti boyutlarını yazdırın
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Görüntü boyutlarını yazdırın
print(f"Image shape: {X_train[0].shape}")

import matplotlib.pyplot as plt

# Verileri normalize edin
X_train = X_train / 255.0
X_test = X_test / 255.0

# Görüntüleri görselleştirin
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Etiketleri one-hot encode edin
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Modeli oluşturun
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Modeli derleyin
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
history = model.fit(X_train[..., np.newaxis], y_train, epochs=10, validation_data=(X_test[..., np.newaxis], y_test))
