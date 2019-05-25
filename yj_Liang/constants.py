# feature parameters
SAMPLING_RATE = 44100
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 512 # determines number of frames
NUM_MELS = 128


# training parameters


# paths
ORIGINAL_DATA_PATH = '/ssd2/maestro/maestro-v1.0.0/'
DATA_YEARS = ['2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017']

TRAIN_YEARS = ['2004', '2006', '2008', '2009', '2013', '2014', '2015', '2017']
TEST_YEARS = ['2011']

DATA_PATH = 'yj_dataset/'
NON_Pedal_DATA_PATH = DATA_PATH + 'non_pedal/'

# to rebuild Beici's paper
PEDAL_MIDI_PATH = ORIGINAL_DATA_PATH
CONVERTED_WAVE_PATH = ORIGINAL_DATA_PATH + DATA_PATH + 'converted_wavefile/'

SETS = ['train', 'test']