import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==========================
# Audio Feature Extraction
# ==========================

def extract_features(file_path):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            res_type='kaiser_fast'
        )

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(
            mfccs.T,
            axis=0
        )

        return mfccs_scaled

    except Exception as e:
        print("Error:", e)
        return None

# ==========================
# Dataset Loading
# ==========================

audio_dataset_path = "dataset"

features = []

for root, dirs, files in os.walk(audio_dataset_path):

    for file in files:

        if file.endswith(".wav"):

            class_label = os.path.basename(root)

            file_path = os.path.join(
                root,
                file
            )

            data = extract_features(
                file_path
            )

            if data is not None:

                features.append(
                    [data, class_label]
                )

# ==========================
# DataFrame
# ==========================

features_df = pd.DataFrame(
    features,
    columns=[
        'feature',
        'class'
    ]
)

X = np.array(
    features_df.feature.tolist()
)

y = np.array(
    features_df['class'].tolist()
)

# ==========================
# Label Encoding
# ==========================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = tf.keras.utils.to_categorical(
    y
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.reshape(
    X_train.shape[0],
    40,
    1,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    40,
    1,
    1
)

# ==========================
# CNN Model
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3,1),
        activation='relu',
        input_shape=(40,1,1)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Conv2D(
        64,
        kernel_size=(3,1),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        y.shape[1],
        activation='softmax'
    )
)

# ==========================
# Compile
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# Evaluate
# ==========================

score = model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    score[1] * 100
)

# ==========================
# Prediction Function
# ==========================

def predict_sound(file_path):

    feature = extract_features(
        file_path
    )

    feature = feature.reshape(
        1,
        40,
        1,
        1
    )

    prediction = model.predict(
        feature
    )

    index = np.argmax(
        prediction
    )

    label = encoder.inverse_transform(
        [index]
    )

    return label[0]

# ==========================
# Example Prediction
# ==========================

sample_file = "test.wav"

print(
    "Predicted Class:",
    predict_sound(sample_file)
)import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==========================
# Audio Feature Extraction
# ==========================

def extract_features(file_path):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            res_type='kaiser_fast'
        )

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(
            mfccs.T,
            axis=0
        )

        return mfccs_scaled

    except Exception as e:
        print("Error:", e)
        return None

# ==========================
# Dataset Loading
# ==========================

audio_dataset_path = "dataset"

features = []

for root, dirs, files in os.walk(audio_dataset_path):

    for file in files:

        if file.endswith(".wav"):

            class_label = os.path.basename(root)

            file_path = os.path.join(
                root,
                file
            )

            data = extract_features(
                file_path
            )

            if data is not None:

                features.append(
                    [data, class_label]
                )

# ==========================
# DataFrame
# ==========================

features_df = pd.DataFrame(
    features,
    columns=[
        'feature',
        'class'
    ]
)

X = np.array(
    features_df.feature.tolist()
)

y = np.array(
    features_df['class'].tolist()
)

# ==========================
# Label Encoding
# ==========================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = tf.keras.utils.to_categorical(
    y
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.reshape(
    X_train.shape[0],
    40,
    1,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    40,
    1,
    1
)

# ==========================
# CNN Model
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3,1),
        activation='relu',
        input_shape=(40,1,1)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Conv2D(
        64,
        kernel_size=(3,1),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        y.shape[1],
        activation='softmax'
    )
)

# ==========================
# Compile
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# Evaluate
# ==========================

score = model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    score[1] * 100
)

# ==========================
# Prediction Function
# ==========================

def predict_sound(file_path):

    feature = extract_features(
        file_path
    )

    feature = feature.reshape(
        1,
        40,
        1,
        1
    )

    prediction = model.predict(
        feature
    )

    index = np.argmax(
        prediction
    )

    label = encoder.inverse_transform(
        [index]
    )

    return label[0]

# ==========================
# Example Prediction
# ==========================

sample_file = "test.wav"

print(
    "Predicted Class:",
    predict_sound(sample_file)
)import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==========================
# Audio Feature Extraction
# ==========================

def extract_features(file_path):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            res_type='kaiser_fast'
        )

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(
            mfccs.T,
            axis=0
        )

        return mfccs_scaled

    except Exception as e:
        print("Error:", e)
        return None

# ==========================
# Dataset Loading
# ==========================

audio_dataset_path = "dataset"

features = []

for root, dirs, files in os.walk(audio_dataset_path):

    for file in files:

        if file.endswith(".wav"):

            class_label = os.path.basename(root)

            file_path = os.path.join(
                root,
                file
            )

            data = extract_features(
                file_path
            )

            if data is not None:

                features.append(
                    [data, class_label]
                )

# ==========================
# DataFrame
# ==========================

features_df = pd.DataFrame(
    features,
    columns=[
        'feature',
        'class'
    ]
)

X = np.array(
    features_df.feature.tolist()
)

y = np.array(
    features_df['class'].tolist()
)

# ==========================
# Label Encoding
# ==========================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = tf.keras.utils.to_categorical(
    y
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.reshape(
    X_train.shape[0],
    40,
    1,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    40,
    1,
    1
)

# ==========================
# CNN Model
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3,1),
        activation='relu',
        input_shape=(40,1,1)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Conv2D(
        64,
        kernel_size=(3,1),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        y.shape[1],
        activation='softmax'
    )
)

# ==========================
# Compile
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# Evaluate
# ==========================

score = model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    score[1] * 100
)

# ==========================
# Prediction Function
# ==========================

def predict_sound(file_path):

    feature = extract_features(
        file_path
    )

    feature = feature.reshape(
        1,
        40,
        1,
        1
    )

    prediction = model.predict(
        feature
    )

    index = np.argmax(
        prediction
    )

    label = encoder.inverse_transform(
        [index]
    )

    return label[0]

# ==========================
# Example Prediction
# ==========================

sample_file = "test.wav"

print(
    "Predicted Class:",
    predict_sound(sample_file)
)import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ==========================
# Audio Feature Extraction
# ==========================

def extract_features(file_path):

    try:
        audio, sample_rate = librosa.load(
            file_path,
            res_type='kaiser_fast'
        )

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfccs_scaled = np.mean(
            mfccs.T,
            axis=0
        )

        return mfccs_scaled

    except Exception as e:
        print("Error:", e)
        return None

# ==========================
# Dataset Loading
# ==========================

audio_dataset_path = "dataset"

features = []

for root, dirs, files in os.walk(audio_dataset_path):

    for file in files:

        if file.endswith(".wav"):

            class_label = os.path.basename(root)

            file_path = os.path.join(
                root,
                file
            )

            data = extract_features(
                file_path
            )

            if data is not None:

                features.append(
                    [data, class_label]
                )

# ==========================
# DataFrame
# ==========================

features_df = pd.DataFrame(
    features,
    columns=[
        'feature',
        'class'
    ]
)

X = np.array(
    features_df.feature.tolist()
)

y = np.array(
    features_df['class'].tolist()
)

# ==========================
# Label Encoding
# ==========================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = tf.keras.utils.to_categorical(
    y
)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.reshape(
    X_train.shape[0],
    40,
    1,
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    40,
    1,
    1
)

# ==========================
# CNN Model
# ==========================

model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3,1),
        activation='relu',
        input_shape=(40,1,1)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Conv2D(
        64,
        kernel_size=(3,1),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,1)
    )
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        y.shape[1],
        activation='softmax'
    )
)

# ==========================
# Compile
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ==========================
# Evaluate
# ==========================

score = model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    score[1] * 100
)

# ==========================
# Prediction Function
# ==========================

def predict_sound(file_path):

    feature = extract_features(
        file_path
    )

    feature = feature.reshape(
        1,
        40,
        1,
        1
    )

    prediction = model.predict(
        feature
    )

    index = np.argmax(
        prediction
    )

    label = encoder.inverse_transform(
        [index]
    )

    return label[0]

# ==========================
# Example Prediction
# ==========================

sample_file = "test.wav"

print(
    "Predicted Class:",
    predict_sound(sample_file)
)
