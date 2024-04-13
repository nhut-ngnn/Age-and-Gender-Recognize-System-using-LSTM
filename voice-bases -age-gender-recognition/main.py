import os
from pathlib import Path

import numpy as np

from extract_features import get_audio_features
from gender_model import lstm_gender_model
from age_model import lstm_age_model
from utils import norm_multiple
from file_io import get_data_files

gender_labels = {
    0: "female",
    1: "male"
}

age_labels = {
    0: "fifties_sixties",
    1: "fourties",
    2: "teens",
    3: "thirties",
    4: "twenties"
    
}


data_path = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/audio/"
models_path = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/model/"


def get_gender(out_data):
    out_data = out_data[0]
    return gender_labels[int(np.argmax(out_data))]


def get_age(out_data):
    out_data = out_data[0]
    return age_labels[int(np.argmax(out_data))]

def main_program():

    gender_weights, gender_means, gender_stddev = get_data_files(models_path, "gender", 30)
    age_weights, age_means, age_stddev = get_data_files(models_path, "age", 100)
    np.set_printoptions(precision=3)

    num_gender_labels = len(gender_labels)
    num_age_labels = len(age_labels)

    # declare the models
    gender_model = lstm_gender_model(num_gender_labels)
    age_model = lstm_age_model(num_age_labels)

    # load models
    gender_model.load_weights(gender_weights)
    age_model.load_weights(age_weights)

    mean_paths = [gender_means, age_means]
    stddev_paths = [gender_stddev, age_stddev]

    data_files = os.listdir(data_path)

    for data_file in data_files:
        data = get_audio_features(Path(data_path + data_file),
                                  extra_features=["delta", "delta2", "pitch"])
        data = np.array([data.T])

        data = norm_multiple(data, mean_paths, stddev_paths)
    
        gender_predict = gender_model.predict(data[0])
        print(gender_predict)
        age_predict = age_model.predict(data[1])
        print(age_predict)
        
        gender_print = "{} ==> GENDER(lstm): {} gender_prob: {}".format(data_file,
                        get_gender(gender_predict).upper(), gender_predict)
        age_print = "{} ==> AGE(lstm): {} age_prob: {}".format(data_file,
                         get_age(age_predict).upper(), age_predict)
    
        print('=' * max(len(gender_print), len(age_print)))
        print()
        print(gender_print)
        print(age_print)
        print()
        print('=' * max(len(gender_print), len(age_print)))


if __name__ == '__main__':
    main_program()