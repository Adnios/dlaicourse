from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []


for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# for i in train_samples:
#     print(i)
# for i in train_labels:
#     print(i)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# for i in scaler_train_samples:
#     print(i)


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax'),
])
model.summary()


model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x=scaler_train_samples, y=train_labels, batch_size=10, epochs=30,
#           shuffle=True, verbose=2)

model.fit(x=scaler_train_samples, y=train_labels,
          validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)


test_labels = []
test_samples = []

for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaler_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

predictions = model.predict(x=scaler_test_samples, batch_size=10, verbose=0)
# for i in predictions:
#     # print(i)

rounded_predictions = np.argmax(predictions, axis=-1)

# for i in rounded_predictions:
#     print(i)

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
           padding='same', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax'),
])
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

test_batches.classes

predictions = model.predict(x=test_batches, verbose=0)

np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))

test_batches.class_indices

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='confusion Matrix')

vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()


def count_params(model):
    non_trainable_params = np.sum(
        [np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list())
                               for v in model.trainable_weights])
    return {'non_trainable_params': non_trainable_params, 'trainable_params': trainable_params}


params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()

params = count_params(model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 134260544

for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))
model.summary()

params = count_params(model)
assert params['non_trainable_params'] == 134260544
assert params['trainable_params'] == 8194

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

predictions = model.predict(x=test_batches, verbose=0)
test_batches.classes

cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))

test_batches.class_indices

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='confusion Matrix')
