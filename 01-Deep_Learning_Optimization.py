# Warning Prevention:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ----------------------------------------------------------------------------------------------------------------------
# Train Dataset:
import numpy as np
from random import randint
from sklearn.utils import shuffle
train_labels = []
train_samples = []
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)
# ----------------------------------------------------------------------------------------------------------------------
# Data Scaling (Train Samples Normalization):
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
# ----------------------------------------------------------------------------------------------------------------------
# Model Building:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
model = Sequential([Input(shape=(1,)),                       # Input Layer
                    Dense(units= 256, activation= 'relu'),   # 1st Hidden Layer
                    Dropout(0.3),
                    Dense(units= 128, activation= 'relu'),   # 2nd Hidden Layer
                    Dense(units= 2, activation= 'softmax')]) # Output Layer
print(model.summary())
# ----------------------------------------------------------------------------------------------------------------------
# Model Compiling:
from tensorflow.keras.optimizers import Adam
model.compile(optimizer= Adam(learning_rate= .0001), loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
# ----------------------------------------------------------------------------------------------------------------------
# Model Training (+validation & History):
history = model.fit(x= scaled_train_samples, y= train_labels, validation_split= 0.1,
                    batch_size= 10, epochs= 30, shuffle= True, verbose= 2)
# ----------------------------------------------------------------------------------------------------------------------
# Drawing Learning Curves:
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# Test Dataset:
test_labels = []
test_samples = []
for i in range(10):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
for i in range(200):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)
# ----------------------------------------------------------------------------------------------------------------------
# Data Scaling (Test Samples Normalization):
scaled_test_samples = scaler.transform(test_samples.reshape(-1,1))
# ----------------------------------------------------------------------------------------------------------------------
# Making Prediction:
predictions = model.predict(x= scaled_test_samples, batch_size= 10, verbose= 0)
rounded_predictions = np.argmax(predictions, axis=1)
# ----------------------------------------------------------------------------------------------------------------------
# Confusion Matrix:
from sklearn.metrics import confusion_matrix
import itertools
cm = confusion_matrix(y_true= test_labels, y_pred= rounded_predictions)
def plot_confusion_matrix(cm, classes, normalize= False, title= 'Confusion Matrix', cmap= 'Greens'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color= "white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm_plot_labels = ['no_side_effect', 'had_side_effect']
plot_confusion_matrix(cm= cm, classes= cm_plot_labels, title= 'Confusion Matrix')
plt.show()
