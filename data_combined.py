import pandas as pd
import numpy as np
import re

import librosa
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

# Constants
SAMPLING_RATE = 22050
LANGUAGES = ["english", "spanish", "french"]


def load_datasets():
    data = []
    for language in LANGUAGES:
        data.append(pd.read_parquet(f"{language}_data.parquet"))

    return pd.concat(data)


def fix_ages(age):
    format_re = "[0-9]{1,2};[0-9]{1,2}"

    if age == None:
        age = "unknown"
    elif re.match(format_re, age): # remove months from the age.
        age = age.split(";")[0]

    return age


def fix_genders(gender):
    if gender in ["female", "f", "mom"]:
        gender = "F"
    elif gender in ["male", "m", "dad"]:
        gender = "M"
    else: 
        gender= None

    return gender


def convert_stereo(wav):
    '''
    Reshape wav form in order to convert it back to stereo.
    '''

    left = wav[::2]
    right = wav[1::2]

    new_wav = np.array([left, right])
    return new_wav.T


def resample(wav, rate, target_rate=SAMPLING_RATE):
    return librosa.resample(wav, orig_sr=rate, target_sr=target_rate)


def clean_transcript(transcript):
    transcript = re.sub('[^a-zA-Z0-9]', ' ', transcript.lower()) # remove everything that is not a letter or number
    transcript = re.sub('[ ]+', ' ', transcript) # remove additional spaces introduced by removing punctuation


def write_data(data, languages):
    for language in languages:
        temp_lan = data[data['language'] == language]
        temp_lan.to_parquet(f"{language}_data_clean.parquet")


def main():
    print("loading datasets...")
    data = load_datasets()

    print("check for duplicate IDs...")
    if True in data.duplicated(subset=['id']).values: 
        print("ID contains duplicates")

    print("apply all functions to get same format...")
    data['age'] = data['age'].apply(lambda x: fix_ages(x))
    data['gender'] = data['gender'].apply(lambda x: fix_genders(x))
    # data['wav'] = data.apply(lambda x: convert_stereo(x['wav']) if x['audio_type'] == "stereo" else x['wav'], axis=1)
    data['wav'] = data.apply(lambda x: resample(x['wav'], x['rate']) if x['rate'] != SAMPLING_RATE else x['wav'], axis=1)
    
    print("writing data...")
    # write one file per language
    languages = np.unique(data['language'].values)
    write_data(data, languages)

    return

if __name__ == "__main__":
  main()

'''
To convert stereo to mono:

import torch
import torchaudio

def stereo_to_mono_convertor(signal):
    # If there is more than 1 channel in your audio
    if signal.shape[0] > 1:
        # Do a mean of all channels and keep it in one channel
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

# Load audio as tensor
waveform, sr = torchaudio.load('audio.wav')
# Convert it to mono channel
waveform = stereo_to_mono_convertor(waveform)
'''