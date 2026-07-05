# Warning Prevention:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ----------------------------------------------------------------------------------------------------------------------
# Dataset:
import pandas as pd
df = pd.read_csv('students.csv')
# ----------------------------------------------------------------------------------------------------------------------
# X/y Split:
X = df[['hours', 'attendance', 'assignments']]
y = df['passed']
# ----------------------------------------------------------------------------------------------------------------------
# Train/Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
# ----------------------------------------------------------------------------------------------------------------------
# Data Scaling (Train Samples Standardization):
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ----------------------------------------------------------------------------------------------------------------------
# Models Building:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
model1 = Sequential([Input(shape=(3,)),                       # Input Layer
                     Dense(units= 16, activation= 'relu'),    # 1st Hidden Layer
                     Dense(units= 1, activation= 'sigmoid')]) # Output Layer
model2 = Sequential([Input(shape=(3,)),                       # Input Layer
                     Dense(units= 64, activation= 'relu'),    # 1st Hidden Layer
                     Dropout(0.3),
                     Dense(units= 32, activation= 'relu'),    # 2nd Hidden Layer
                     Dropout(0.2),
                     Dense(units= 1, activation= 'sigmoid')]) # Output Layer
# ----------------------------------------------------------------------------------------------------------------------
# Models Compiling:
from tensorflow.keras.optimizers import Adam
model1.compile(optimizer= Adam(learning_rate=0.0001), loss= "binary_crossentropy", metrics= ["accuracy"])
model2.compile(optimizer= Adam(learning_rate=0.0001), loss= "binary_crossentropy", metrics= ["accuracy"])
# ----------------------------------------------------------------------------------------------------------------------
# Models Training:
model1.summary()
history1= model1.fit(X_train, y_train, validation_split= 0.2, epochs= 100, batch_size= 32, shuffle= True, verbose= 2)
model2.summary()
history2= model2.fit(X_train, y_train, validation_split= 0.2, epochs= 100, batch_size= 32, shuffle= True, verbose= 2)
# ----------------------------------------------------------------------------------------------------------------------
# Drawing Learning Curves:
import matplotlib.pyplot as plt
plt.subplot(2, 2, 1)
plt.grid(True)
plt.plot(history1.history["loss"], label="Train Loss")
plt.plot(history1.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Model1 Loss")
plt.subplot(2, 2, 2)
plt.grid(True)
plt.plot(history1.history["accuracy"], label="Train Accuracy")
plt.plot(history1.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model1 Accuracy")
plt.subplot(2, 2, 3)
plt.grid(True)
plt.plot(history2.history["loss"], label="Train Loss")
plt.plot(history2.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Model2 Loss")
plt.subplot(2, 2, 4)
plt.grid(True)
plt.plot(history2.history["accuracy"], label="Train Accuracy")
plt.plot(history2.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model2 Accuracy")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Making Prediction:
pred1 = (model1.predict(X_test) > 0.5).astype(int)
pred2 = (model2.predict(X_test) > 0.5).astype(int)
# ----------------------------------------------------------------------------------------------------------------------
# Confusion Matrix:
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred1, cmap= "Blues", colorbar= False, ax= ax[0])
ax[0].set_title("Model 1")
ConfusionMatrixDisplay.from_predictions(y_test, pred2, cmap= "Greens", colorbar= False, ax= ax[1])
ax[1].set_title("Model 2")
plt.tight_layout()
plt.show()
