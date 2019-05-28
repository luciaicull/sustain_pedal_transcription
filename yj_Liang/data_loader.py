from abc import abstractmethod
from torch.utils.data import Dataset
import torch
import soundfile
import csv

from constants import *


class ExcerptDataset(Dataset):
    def __init__(self, set, pre_load=True):
        self.set = set
        self.pre_load = pre_load
        self.excerpt_csv = self.get_csv_file(self.set)
<<<<<<< HEAD

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
=======
        self.excerpt_list = self.load_excerpt_list(self.excerpt_csv)

        if self.pre_load:
            self.audio_data = self.load_audio_dict(self.set)
            keys = self.audio_data['pedal'].keys()
            # Only necessary when loading incomplete audio files?
            self.excerpt_list = [el for el in self.excerpt_list if el[0] in keys]
            print(keys)
>>>>>>> 93648bf9b53d21302328c1dd4095ab3c034a54c1


    def __getitem__(self, index):
        excerpt_info = self.excerpt_list[index]
        file_name = excerpt_info[0]
        start = int(excerpt_info[1])
        end = int(excerpt_info[2])
        pedal = 1.0 if excerpt_info[3]=='p' else 0.0

<<<<<<< HEAD
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
=======
        if self.pre_load:
            if pedal:
                audio = self.audio_data['pedal'][file_name]
            else:
                audio = self.audio_data['non-pedal'][file_name]
        else:
            audios = self.load_audio_file(file_name)
            audio = audios['pedal'] if pedal == 1 else audios['non-pedal']

        excerpt = audio[start:end]
        print(pedal)
        pedal = torch.ByteTensor([pedal])
>>>>>>> 93648bf9b53d21302328c1dd4095ab3c034a54c1

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
<<<<<<< HEAD
        audio_dict = dict()
        audio_dict['pedal'] = dict()
        audio_dict['non-pedal'] = dict()

        p_dict = audio_dict['pedal']
        np_dict = audio_dict['non-pedal']

        pedal_data_path = DATA_PATH + CONVERTED_WAVE_PATH + 'pedal/'
        nonpedal_data_path = DATA_PATH + CONVERTED_WAVE_PATH + 'non-pedal/'
=======
        p_dict = dict()
        np_dict = dict()       
>>>>>>> 93648bf9b53d21302328c1dd4095ab3c034a54c1

        f = open(DATA_PATH + set + '.txt', 'r')
        file_names = f.readlines()

        i = 0
        for file_name in file_names:
            if i % 10 == 0 and i is not 0:
                print(i)
                break
            i += 1
            file_name = file_name.split('\n')[0]
            
            audios = self.load_audio_file(file_name)
            # print(pedal_audio)
            p_dict[file_name] = audios['pedal']
            np_dict[file_name] = audios['non-pedal']

        audio_dict = dict()
        audio_dict['pedal'] = p_dict
        audio_dict['non-pedal'] = np_dict
            
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
    

    @abstractmethod
    def load_audio_file(self, file_name):
        pedal_audio, sr = soundfile.read(PEDAL_DATA_PATH + file_name + '.flac', dtype='int16')
        non_pedal_audio, sr = soundfile.read(NON_PEDAL_DATA_PATH + file_name + '.flac', dtype='int16')

        pedal_audio = torch.ShortTensor(pedal_audio)
        non_pedal_audio = torch.ShortTensor(non_pedal_audio)
        audios = {  'pedal': pedal_audio,
                    'non-pedal': non_pedal_audio}
        return audios


class OnsetExcerptDataset(ExcerptDataset):
    def __init__(self, set, pre_load=True):
        super(OnsetExcerptDataset, self).__init__(set, pre_load=pre_load)

    @classmethod
    def get_csv_file(self, set):
        return set + '_onset'


class SegmentExcerptDataset(ExcerptDataset):
    def __init__(self, set, pre_load=True):
        super(SegmentExcerptDataset, self).__init__(set, pre_load=pre_load)

    @classmethod
    def get_csv_file(self, set):
        return set + '_segment'


if __name__ == '__main__':
    dataset = OnsetExcerptDataset('train', pre_load=False)
    a = iter(dataset)
    print('len = ' + str(len(dataset)))
    b = next(a)
    print(b)
    for audio in dataset:
        print(audio)
