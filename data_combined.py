import pandas as pd
import numpy as np
import re
import argparse

import librosa
import os

# Constants
SAMPLING_RATE = 22050

def load_datasets(languages):
    data = []
    for language in languages:
        data.append(pd.read_parquet(f"{language}.parquet"))

    return pd.concat(data)


def fix_ages(age):
    year_month = "[0-9]{1,2};[0-9]{0,2}.{0,1}"
    year = "[0-9]{1,2}"
    
    if re.match(year_month, age): # remove months from the age.
        age = age.split(";")[0]
    
    if not re.match(year, age):
        return None

    return int(age)


def fix_genders(gender):
    if gender in ["female", "f", "mom"]:
        gender = "F"
    elif gender in ["male", "m", "dad"]:
        gender = "M"
    else: 
        gender= None

    return gender


def stereo_to_mono(wav):
    left = wav[::2]
    right = wav[1::2]

    new_wav = np.array([left, right])
    return new_wav.T


def mono_to_stereo(wav):
    return wav.flatten()


def resample(wav, audio_type, rate, target_rate=SAMPLING_RATE):
    if audio_type == "mono":
        return librosa.resample(wav, orig_sr=rate, target_sr=target_rate)
    else: 
        wav = stereo_to_mono(wav)
        wav = librosa.resample(wav, orig_sr=rate, target_sr=target_rate)
        return mono_to_stereo(wav)


def clean_transcript(transcript):
    # normalise cues
    transcript = re.sub(' \? ', ' XXX ', transcript)
    transcript = re.sub('xxx', 'XXX', transcript)
    transcript = re.sub('unintelligible', 'XXX', transcript)

    # remove punctuation and any characters that are not letters or numbers
    transcript = re.sub('[^a-zA-Z0-9À-ÿ]', ' ', transcript) 
    transcript = re.sub('[ ]+', ' ', transcript) # remove additional spaces introduced by removing special characters
    return transcript


def write_data(data, languages):
    for language in languages:
        temp_lan = data[data['language'] == language]
        temp_lan.to_parquet(f"{language}_clean.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--languages", action="store", nargs='+', type=str, help="provide a comma seperated (no spaces) list of languages", )
    parser.add_argument('--print', default=False, action="store_true", help="if set the dataframe will be printed")
    parser.add_argument('--no_write', action='store_true', help="if set the dataframe is not written to file")
    parser.add_argument("-sr", "--sampling_rate", action='store', default=None, type=int, help="If set to a value, all audio will be resampled to this value")
    args = parser.parse_args()


    languages = "".join(args.languages).split(",")
    data = load_datasets(languages)
    
    # check for duplicate id's
    if True in data.duplicated(subset=['id']).values: 
        print("ID contains duplicates")

    # apply functions for age and gender
    data['age_group'] = np.where(data['age'] == 'adult', "adult", "child")
    data['age'] = data['age'].apply(lambda x: fix_ages(x))
    data['age_group'] = data.apply(lambda x: "adult" if x['age'] >= 18 else x['age_group'], axis=1) # anyone 18 or above is also an adult
    data['gender'] = data['gender'].apply(lambda x: fix_genders(x))

    # apply functions for wavs and transcripts
    data['wav'] = data['wav'].apply(lambda x: np.array(x, dtype=np.float32))
    if args.sampling_rate is not None:
        data['wav'] = data.apply(lambda x: resample(x['wav'], x['rate'], args.sampling_rate, x['audio_type']), axis=1)
    data['transcript'] = data['transcript'].apply(lambda x: clean_transcript(x))

    # filenames can be dropped at this point
    data = data.drop(columns="filename")

    for column in ["gender", "age_group", "id", "language", "transcript"]:
        data[column] = data[column].astype(str)

    # reorder columns into a logical order
    data = data[['id', 'age', 'age_group', 'gender', 'language', 'transcript', 'wav', 'rate', 'audio_type']]

    # perform some additional checks for testing
    # print(data.columns)
    # print(np.unique(data['age'].values))
    # print(data['gender'].values)
    # print(np.unique(data['rate'].values))

    if args.print:
        print(data)

    if not args.no_write:
        write_data(data, languages)

    return

if __name__ == "__main__":
  main()
