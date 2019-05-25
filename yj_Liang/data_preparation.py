import os
import librosa
import numpy as np
import pretty_midi
from constants import *

def get_non_pedal_midi():
    for set in SETS:
        f = open(ORIGINAL_DATA_PATH + DATA_PATH + set + '.txt', 'r')
        file_names = f.readlines()
        for file_name in file_names:
            file_name = file_name.split('\n')[0]
            file_path = ORIGINAL_DATA_PATH + file_name + '.midi'
            # print(file_path)

            pedal_midi = pretty_midi.PrettyMIDI(file_path)
            non_pedal_midi = pretty_midi.PrettyMIDI()
            non_pedal_inst = pretty_midi.Instrument(program=0)
            for pedal_inst in pedal_midi.instruments:
                for note in pedal_inst.notes:
                    non_pedal_inst.notes.append(note)
                #for control_change in pedal_inst.control_changes:
                    #non_pedal_inst.control_changes.append(control_change)
            non_pedal_midi.instruments.append(non_pedal_inst)

            non_pedal_file_path = ORIGINAL_DATA_PATH + NON_Pedal_DATA_PATH + file_name + '.midi'
            print(non_pedal_file_path)
            non_pedal_midi.write(non_pedal_file_path)


def get_file_list_txt():
    for year in DATA_YEARS:
        # get file list of directory
        f = open(ORIGINAL_DATA_PATH + DATA_PATH + 'data-'+year+'.txt', 'w')
        file_list = os.listdir(ORIGINAL_DATA_PATH + year)
        file_list.sort()

        # get midi file names
        for item in file_list:
            if item.find('.mid') is not -1:
                file_name = item.split('.')[0]
                f.write(file_name + '\n')
        print(year)
        f.close()

def organize_file_names(set_years):
    if set_years == TRAIN_YEARS:
        write_f = open(ORIGINAL_DATA_PATH + DATA_PATH + 'train.txt', 'w')
    else: # set_years == TEST_YEARS
        write_f = open(ORIGINAL_DATA_PATH + DATA_PATH + 'test.txt', 'w')

    for year in set_years:
        print(year)
        read_f = open(ORIGINAL_DATA_PATH + DATA_PATH + 'data-'+year+'.txt', 'r')
        file_names = read_f.readlines()
        for file_name in file_names:
            write_f.write(year + '/' + file_name)
        read_f.close()
    write_f.close()


def train_test_split():
    organize_file_names(TRAIN_YEARS)
    organize_file_names(TEST_YEARS)


def melspectrogram(file_name):
    y, sr = librosa.load(ORIGINAL_DATA_PATH + file_name, SAMPLING_RATE)
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    mel_basis = librosa.filters.mel(SAMPLING_RATE, n_fft=N_FFT, n_mels=NUM_MELS)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1 + 10 * mel_S)

    return mel_S

def extract_features():
    for set in SETS:
        f = open(ORIGINAL_DATA_PATH + DATA_PATH + set + '.txt', 'r')
        file_names = f.readlines()

        for file_name in file_names:
            file_name = file_name.split('\n')[0]
            audio_file_name = file_name + '.wav'
            feature = melspectrogram(audio_file_name)

            save_name = file_name.split('/')[0] + '-' + file_name.split('/')[1] + '.npy'
            np.save(ORIGINAL_DATA_PATH + DATA_PATH + set + '/' + save_name, feature.astype(np.float32))
            print(save_name)


if __name__ == '__main__':
    #get_file_list_txt()
    #train_test_split()
    #extract_features()
    get_non_pedal_midi()
