
import keras_metrics as km
from keras import metrics

from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization

from models import train_multi_epoch, train_deepnn

NUM_FEATURES = 41  # 20, 41, 39


def lstm_gender_model(num_labels):
    model = Sequential() 
    model.add(LSTM(256, input_shape=(35, NUM_FEATURES), dropout=0.3, return_sequences=True))
    model.add(LSTM(256, dropout=0.3))
    model.add(Dense(128 * 2, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def main_class_gender_train():
    dataset = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/gender_data_clean"
    model = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/model/lstm_gender_"
    train_multi_epoch(dataset, model + str(NUM_FEATURES),
                      lstm_gender_model, train_deepnn,
                      num_epoch_start=30,
                      num_features=NUM_FEATURES,
                      file_prefix="gender")


if __name__ == '__main__':
    main_class_gender_train()
