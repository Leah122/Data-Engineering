# standard
import pandas as pd
import numpy as np
import re

# audio
from chamd import ChatReader
import librosa

# system
import os

# parquet
import pyarrow as pa
import pyarrow.parquet as pq

import sys
np.set_printoptions(threshold=sys.maxsize)


def decode_transcript_en(path):
    '''
    Read the transcript and id from the cha file in the given path.
    '''

    reader = ChatReader()
    chat = reader.read_file(path)
    lines = []

    id = chat.metadata['pid'].text

    # go over the transcript line by line to read the transcript
    for item in chat.lines:
        text = item.text
        lines.append(text)

    lines = " ".join(lines)
    return lines, id


def decode_transcript_fr_sp(path):
    '''
    Get all metadata and the transcript from cha file in the given path.
    '''

    reader = ChatReader()
    chat = reader.read_file(path)
    lines = []

    id = chat.metadata['pid'].text

    # go over the transcript line by line to read the transcript
    for item in chat.lines:
        text = item.text
        lines.append(text)

        # get age and sex if the role of the speaker is the target child
        if item.metadata['role'].text == "Target_Child":
            sex = item.metadata['sex'].text
            age = item.metadata['age'].text
    
    # set unknown if age and sex are not found in metadata.
    if sex == None:
        sex = "unknown"
    if age == None:
        age = "unknown"

    lines = " ".join(lines)
    return lines, id, sex, age


def read_transcripts(language, folders):
    '''
    Go through all files in the filestructure and read data from the cha files.
    Return metadata and transcripts in the form of a Dataframe.
    '''

    directory = os.fsencode(f"{language}/").decode('utf-8')
    rows = []
    for path in os.listdir(directory): # go though all folders in the directory
        if path in folders:
            path = directory + path
            for file in os.listdir(path): # go though all files in folders (which contain files of the same age group)
                row = []
                filename = os.fsdecode(file)
                if filename.endswith(".cha"): # only get the cha files

                    # some differences in the meta-data structure for enlglish, so it needs it's own code.
                    if language == "english":
                        id = filename.split(".")[0]
                        speaker = "spk"+id+"_"+path+"-utt01"
                        transcript = decode_transcript_en(path+"/"+filename)

                        row.append(path + "/" + id)
                        row.append(speaker)
                        row.append(path)
                        row.append(re.sub("[0-9]", "", id.split("_")[0])) # first part contains either number and F/M or mom/dad
                        row.append(language)
                        row.append(transcript)

                        rows.append(row)
                    
                    else:
                        transcript, id, gender, age = decode_transcript_fr_sp(path+"/"+filename)
                        row.append(path + "/" + filename.split(".")[0])
                        row.append(id)
                        row.append(age)
                        row.append(gender)
                        row.append(language)
                        row.append(transcript)

                        rows.append(row)

    data = pd.DataFrame(rows, columns=["filename", "id", "age", "gender", "language", "transcript"])
    return data


def add_audio(df):
    '''
    Reads the audio from the file name corresponding to the cha file name.
    Adds the wave, sampling rate, and audio type to the dataframe.
    '''

    wavs = []
    rates = []
    audio_types = []

    for index, row in df.iterrows():
        file_name = row['filename']
        wav_file = librosa.load(file_name + ".wav")
        rate = wav_file[1]
        wav = np.array(wav_file[0])
        
        # shape of the wav shows if it is mono or stereo
        if len(wav.shape) == 1:
            audio_types.append("mono")
        elif len(wav.shape) == 2:
            audio_types.append("stereo")
            wav = wav.flatten() # alternating left and right values

        wavs.append(wav)
        rates.append(rate)

    wavs = np.array(wavs, dtype=object)
    rates = np.array(rates)
    
    df["wav"] = wavs
    df["rate"] = rates
    df["audio_type"] = audio_types

    return df


def main(): 
    print("loading tanscripts and metadata")
    data_en = read_transcripts("english", ["5", "6", "7", "8", "9", "10", "11", "adult"])
    data_fr = read_transcripts("french", ["4", "5", "6", "7", "8", "9", "10"])
    data_sp = read_transcripts("spanish", ["2nd grade", "5th grade"])

    print("loading waves")
    data_en = add_audio(data_en)
    data_fr = add_audio(data_fr)
    data_sp = add_audio(data_sp)
    
    # for testing
    # print(data_en)
    # print(data_fr)
    # print(data_sp)

    print("writing to file")
    data_en.to_parquet("english_data.parquet")
    data_fr.to_parquet("french_data.parquet")
    data_sp.to_parquet("spanish_data.parquet")

if __name__ == "__main__":
  main()