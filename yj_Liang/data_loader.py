from abc import abstractmethod
from torch.utils.data import Dataset
import torch
import soundfile
import csv

from constants import *


class ExcerptDataset(Dataset):
    def __init__(self, set):
        self.set = set
        self.excerpt_csv = self.get_csv_file(self.set)

        '''
        # unavailable to use because of lack of RAM size
        self.audio_data = self.load_audio_dict(self.set)
        '''

        self.excerpt_list = self.load_excerpt_list(self.excerpt_csv)

        '''
        # temporary code for testing dataset
        keys = self.audio_data['pedal'].keys()
        print(keys)
        self.excerpt_list = [el for el in self.excerpt_list if el[0] in keys]
        '''


    def __getitem__(self, index):
        excerpt_info = self.excerpt_list[index]
        file_name = excerpt_info[0]
        start = int(excerpt_info[1])
        end = int(excerpt_info[2])
        pedal = 1.0 if excerpt_info[3]=='p' else 0.0

        if pedal:
            '''
            # unavailable to be used
            audio = self.audio_data['pedal'][file_name]
            '''
            audio_data_path = DATA_PATH + CONVERTED_WAVE_PATH + 'pedal/'

        else:
            '''
            # unavailable to be used
            audio = self.audio_data['non-pedal'][file_name]
            '''
            audio_data_path = DATA_PATH + CONVERTED_WAVE_PATH + 'non-pedal/'

        audio, sr = soundfile.read(audio_data_path + file_name + '.flac', dtype='int16')

        excerpt = audio[start:end]
        excerpt = torch.ShortTensor(excerpt)

        pedal = torch.FloatTensor([pedal])

        return excerpt, pedal


    def __len__(self):
        return len(self.excerpt_list)


    @classmethod
    @abstractmethod
    def get_csv_file(self, set):
        '''
        test_onset
        test_segment
        train_onset
        train_segment
        '''
        raise NotImplementedError

    '''
    # unavailable to use because of lack of RAM size
    @abstractmethod
    def load_audio_dict(self, set):
        audio_dict = dict()
        audio_dict['pedal'] = dict()
        audio_dict['non-pedal'] = dict()

        p_dict = audio_dict['pedal']
        np_dict = audio_dict['non-pedal']

        pedal_data_path = DATA_PATH + CONVERTED_WAVE_PATH + 'pedal/'
        nonpedal_data_path = DATA_PATH + CONVERTED_WAVE_PATH + 'non-pedal/'

        f = open(DATA_PATH + set + '.txt', 'r')
        file_names = f.readlines()

        i = 0
        for file_name in file_names:
            if i % 10 == 0 and i is not 0:
                print(i)
                break
            i += 1
            file_name = file_name.split('\n')[0]

            pedal_audio, sr = soundfile.read(pedal_data_path + file_name + '.flac', dtype='int16')
            non_pedal_audio, sr = soundfile.read(nonpedal_data_path + file_name + '.flac', dtype='int16')

            pedal_audio = torch.ShortTensor(pedal_audio)
            non_pedal_audio = torch.ShortTensor(non_pedal_audio)

            p_dict[file_name] = pedal_audio
            np_dict[file_name] = non_pedal_audio

        return audio_dict
    '''

    @abstractmethod
    def load_excerpt_list(self, excerpt_csv):
        '''
        excerpt_csv ==  test_onset
                        test_segment
                        train_onset
                        train_segment
        '''
        excerpt_list = []

        csv_path = DATA_PATH + CONVERTED_WAVE_PATH + excerpt_csv + '.csv'


        f = open(csv_path, 'r')
        reader = csv.reader(f)

        for line in reader:
            excerpt_list.append([line[0], line[1], line[2], 'p'])
            excerpt_list.append([line[0], line[1], line[2], 'np'])

        return excerpt_list


class OnsetExcerptDataset(ExcerptDataset):
    def __init__(self, set):
        super(OnsetExcerptDataset, self).__init__(set)

    @classmethod
    def get_csv_file(self, set):
        return set + '_onset'


class SegmentExcerptDataset(ExcerptDataset):
    def __init__(self, set):
        super(SegmentExcerptDataset, self).__init__(set)

    @classmethod
    def get_csv_file(self, set):
        return set + '_segment'

if __name__ == '__main__':
    dataset = OnsetExcerptDataset('train')
    a = iter(dataset)
    print('len = ' + str(len(dataset)))
    b = next(a)
    print(b)
