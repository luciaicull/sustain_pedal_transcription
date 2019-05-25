import os
import librosa
import numpy as np
import pretty_midi
import csv
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


def pedal_onset_preparation():
    print("Start creating pedal-onset excerpts")


    for SET in SETS:
        f = open(ORIGINAL_DATA_PATH + DATA_PATH + SET + '.txt', 'r')
        file_names = f.readlines()
        f_pedal = open(ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + SET + '_onset_p.csv', 'w')
        f_nonpedal = open(ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + SET + '_onset_np.csv', 'w')
        p_writer = csv.writer(f_pedal)
        np_writer = csv.writer(f_nonpedal)
        for file_name in file_names:
            file_name = file_name.split('\n')[0]
            print('      ' + file_name)
            midi_path = ORIGINAL_DATA_PATH + file_name + '.midi'
            pedal_path = ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + 'pedal/' + file_name + '.wav'
            non_pedal_path = ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + 'non-pedal/' + file_name + '.wav'

            pedal_audio, sr = librosa.load(pedal_path, sr=SAMPLING_RATE)
            non_pedal_audio, sr = librosa.load(non_pedal_path, sr=SAMPLING_RATE)

            midi = pretty_midi.PrettyMIDI(midi_path)
            midi_inst = midi.instruments[0]
            pedal_value = []
            pedal_time = []
            # detect sustain pedal(control change number 64)
            for control_change in midi_inst.control_changes:
                if control_change.number == 64:
                    pedal_value.append(control_change.value)
                    pedal_time.append(control_change.time)

            pedal_onset_time = []
            # detect sustain pedal onset(control change value above 64)
            for idx, value in enumerate(pedal_value):
                if idx > 0 and value >= 64 and pedal_value[idx-1] < 64:
                    pedal_onset_time.append(pedal_time[idx])


            pedal_onset_sample = librosa.time_to_samples(pedal_onset_time, sr=SAMPLING_RATE)
            for seg_idx, sample in enumerate(pedal_onset_sample):
                start = int(sample - TRIM_SECOND_BEFORE * SAMPLING_RATE)
                end = int(sample + TRIM_SECOND_AFTER * SAMPLING_RATE)
                #new_file_name = file_name.replace('/', '-')

                if start > 0 and end < len(non_pedal_audio):
                    # cut pedal excerpt and store its information in another file
                    # cannot store all excerpt audios itself because of lack of storage
                    '''
                    pedal_excerpt_name = '{}-p_{}.wav'.format(new_file_name, seg_idx)
                    pedal_excerpt_path = ORIGINAL_DATA_PATH + DATA_PATH + SET + '/pedal_onset/' + pedal_excerpt_name
                    librosa.output.write_wav(pedal_excerpt_path, pedal_audio[start:end], SAMPLING_RATE)
                    '''
                    p_writer.writerow([file_name + '.wav', str(start), str(end)])

                    # cut non_pedal excerpt and store its information in another file
                    # cannot store all excerpt audios itself because of lack of storage
                    '''
                    non_pedal_excerpt_name = '{}-np_{}.wav'.format(new_file_name, seg_idx)
                    non_pedal_excerpt_path = ORIGINAL_DATA_PATH + DATA_PATH + SET + '/nonpedal_onset/' + non_pedal_excerpt_name
                    librosa.output.write_wav(non_pedal_excerpt_path, non_pedal_audio[start:end], SAMPLING_RATE)
                    '''
                    np_writer.writerow([file_name + '.wav', str(start), str(end)])

    print("End creating pedal-onset excerpts")



def pedal_segment_preparation():
    print("Start creating pedal-segment excerpts")

    min_sp = int(MIN_SRC * SAMPLING_RATE)
    max_sp = int(MAX_SRC * SAMPLING_RATE)

    for SET in SETS:
        f = open(ORIGINAL_DATA_PATH + DATA_PATH + SET + '.txt', 'r')
        file_names = f.readlines()

        f_pedal = open(ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + SET + '_segment_p.csv', 'w')
        f_nonpedal = open(ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + SET + '_segment_np.csv', 'w')
        p_writer = csv.writer(f_pedal)
        np_writer = csv.writer(f_nonpedal)

        for file_name in file_names:
            file_name = file_name.split('\n')[0]
            print('      ' + file_name)
            midi_path = ORIGINAL_DATA_PATH + file_name + '.midi'

            midi = pretty_midi.PrettyMIDI(midi_path)
            midi_inst = midi.instruments[0]
            pedal_value = []
            pedal_time = []
            # detect sustain pedal(control change number 64)
            for control_change in midi_inst.control_changes:
                if control_change.number == 64:
                    pedal_value.append(control_change.value)
                    pedal_time.append(control_change.time)

            pedal_onset_time = []
            pedal_offset_time = []
            # detect sustain pedal onset(control change value above 64) and offset(control change value under 64)
            for idx, value in enumerate(pedal_value):
                if idx > 0 and value >= 64 and pedal_value[idx-1] < 64:
                    pedal_onset_time.append(pedal_time[idx])
                elif idx > 0 and value < 64 and pedal_value[idx-1] >= 64:
                    pedal_offset_time.append(pedal_time[idx])

            pedal_offset_time = [t for t in pedal_offset_time if t > pedal_onset_time[0]]
            segment_idxs = np.min([len(pedal_onset_time), len(pedal_offset_time)])
            pedal_onset_time = pedal_onset_time[:segment_idxs]
            pedal_offset_time = pedal_offset_time[:segment_idxs]

            correct_pedal_data = False
            for seg_idx, offset_time in enumerate(pedal_offset_time):
                if offset_time != pedal_offset_time[-1] and offset_time > pedal_onset_time[seg_idx] and offset_time < pedal_onset_time[seg_idx+1]:
                    correct_pedal_data = True
                elif offset_time == pedal_offset_time[-1] and offset_time > pedal_onset_time[seg_idx]:
                    correct_pedal_data = True

            if correct_pedal_data:
                pedal_onset_sample = librosa.time_to_samples(pedal_onset_time, sr=SAMPLING_RATE)
                pedal_offset_sample = librosa.time_to_samples(pedal_offset_time, sr=SAMPLING_RATE)

                pedal_path = ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + 'pedal/' + file_name + '.wav'
                non_pedal_path = ORIGINAL_DATA_PATH + DATA_PATH + CONVERTED_WAVE_PATH + 'non-pedal/' + file_name + '.wav'

                pedal_audio, sr = librosa.load(pedal_path, sr=SAMPLING_RATE)
                non_pedal_audio, sr = librosa.load(non_pedal_path, sr=SAMPLING_RATE)

                for seg_idx, start in enumerate(pedal_onset_sample):
                    end = pedal_offset_sample[seg_idx]
                    length = end - start
                    if length > max_sp:
                        end = start + max_sp

                    if length >= min_sp and end < len(non_pedal_audio):
                        p_writer.writerow([file_name + '.wav', str(start), str(end)])
                        np_writer.writerow([file_name + '.wav', str(start), str(end)])

    print("End creating pedal-segment excerpts")



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
    #get_non_pedal_midi()
    pedal_onset_preparation()
