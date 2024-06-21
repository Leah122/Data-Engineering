import pandas as pd
import numpy as np
import re
import os
import argparse

from chamd import ChatReader
import librosa

import pyarrow as pa
import pyarrow.parquet as pq


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
        lines.append(item.text)

        # get age and sex if the role of the speaker is the target child
        if item.metadata['role'].text == "Target_Child":
            sex = item.metadata['sex'].text
            age = item.metadata['age'].text

    lines = " ".join(lines)
    return lines, id, sex, age


def read_transcripts(language):
    '''
    Go through all files in the filestructure and read data from the cha files.
    Return metadata and transcripts in the form of a Dataframe.
    '''

    folder = os.fsencode(f"{language}/").decode('utf-8')
    rows = []
    for path in os.listdir(folder): # go though all age folders in the language folder
        if path[0] != ".": # skip hidden folder (.DS_Store)
            age = path
            path = folder + path
            for file in os.listdir(path): # go though all files in folders (which contain files of the same age group)
                row = []
                filename = os.fsdecode(file)
                if filename.endswith(".cha"): # only get the cha files

                    # some differences in the meta-data structure for enlglish, so it needs it's own code.
                    if language == "english":
                        transcript, id = decode_transcript_en(path+"/"+filename)
                        row.append(path + "/" + filename.split(".")[0])
                        row.append(id)
                        row.append(age)
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
        wav, rate = librosa.load(file_name + ".wav", sr=None, mono=False)
        
        # shape of the wav shows if it is mono or stereo
        if len(wav.shape) == 1:
            audio_types.append("mono")
        elif len(wav.shape) == 2:
            audio_types.append("stereo")
            wav = wav.flatten() # alternating left and right values

        wavs.append(wav)
        rates.append(rate)
    
    df["wav"] = wavs
    df["rate"] = rates
    df["audio_type"] = audio_types

    return df


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--languages", action="store", nargs='+', type=str, help="provide a comma seperated (no spaces) list of languages", )
    parser.add_argument('--print', default=False, action="store_true", help="if set the dataframe will be printed")
    parser.add_argument('--no_write', action='store_true', help="if set the dataframe is not written to file")
    args = parser.parse_args()


    languages = "".join(args.languages).split(",")
    for lang in languages:
        print("processing " + lang)
        data = read_transcripts(lang)
        data = add_audio(data)
        if args.print: 
            print(data)
        if not args.no_write:
            data.to_parquet(lang + ".parquet")

if __name__ == "__main__":
  main()