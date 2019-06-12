from __future__ import division

import pretty_midi
from data_loader import FullSongDataset
from constants import *
import models
import librosa.display
import numpy as np
import librosa
from scipy.signal import medfilt
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('agg')  # Tkinter is not available on the server
import matplotlib.pyplot as plt


def intervals1tointervals01(segintervals1, paudio_duration):
    idx2del = []
    for idx in np.arange(1, len(segintervals1)):
        if segintervals1[idx-1][1] >= segintervals1[idx][0]:
            segintervals1[idx] = [
                segintervals1[idx-1][0], segintervals1[idx][1]]
            idx2del.append(idx-1)
    segintervals1 = np.delete(segintervals1, idx2del, axis=0)

    labels = []
    segintervals01 = np.zeros((len(segintervals1)*2+1, 2))

    for idx in range(len(segintervals01)):
        if idx == 0:
            segintervals01[idx] = [0, segintervals1[0][0]]
            labels.append('np')
        elif idx == len(segintervals01)-1:
            segintervals01[idx] = [segintervals1[-1][-1], paudio_duration]
            labels.append('np')
        elif idx % 2:
            segintervals01[idx] = segintervals1[int(np.floor(idx/2))]
            labels.append('p')
        else:
            segintervals01[idx] = [segintervals1[int(
                np.floor(idx/2)-1)][-1], segintervals1[int(np.floor(idx/2))][0]]
            labels.append('np')

    idx2del = []
    for idx, seginterval in enumerate(segintervals01):
        if seginterval[0] == seginterval[1]:
            idx2del.append(idx)
    segintervals01 = np.delete(segintervals01, idx2del, axis=0)
    labels = np.delete(labels, idx2del)

    return segintervals1, segintervals01, labels


def generate_full_song_gt(midi_path):
    # get ground truth pedal onset time from midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    pedal_v = []
    pedal_t = []
    for control_change in pm.instruments[0].control_changes:
        if control_change.number == 64:
            pedal_v.append(control_change.value)
            pedal_t.append(control_change.time)

    pedal_onset = []
    pedal_offset = []
    for i, v in enumerate(pedal_v):
        if i > 0 and v >= 64 and pedal_v[i-1] < 64:
            pedal_onset.append(pedal_t[i])
        elif i > 0 and v < 64 and pedal_v[i-1] >= 64:
            pedal_offset.append(pedal_t[i])

    pedal_offset = [t for t in pedal_offset if t > pedal_onset[0]]
    seg_idxs = np.min([len(pedal_onset), len(pedal_offset)])
    pedal_offset = pedal_offset[:seg_idxs]
    pedal_onset = pedal_onset[:seg_idxs]
    for seg_idx, offset in enumerate(pedal_offset):
        if offset != pedal_offset[-1] and offset > pedal_onset[seg_idx] and offset < pedal_onset[seg_idx+1]:
            correct_pedal_data = True
        elif offset == pedal_offset[-1] and offset > pedal_onset[seg_idx]:
            correct_pedal_data = True
        else:
            correct_pedal_data = False

    # if correct_pedal_data:
    #     filenames.append(filename)
    #     pedal_onsets.append(pedal_onset)
    #     pedal_offsets.append(pedal_offset)
    #     categories.append(category_list[indx])
    #     authors.append(author_list[indx])

    if correct_pedal_data:
        return np.array(pedal_onset), np.array(pedal_offset)
    else:
        return None, None


class ManualFusion():
    def __init__(self, onset_model, segment_model, onset_threshold=0.98, segment_threshold=0.98):
        self.device = "cuda:0"
        self.onset_model = onset_model.to(self.device)
        self.segment_model = segment_model.to(self.device)

        self.onset_model.eval()
        self.segment_model.eval()

        self.onset_threshold = onset_threshold
        self.segment_threshold = segment_threshold

    def predict(self, audio_data, pedal_gt, batch_size=4):
        pedal_onset_gt, pedal_offset_gt = pedal_gt
        # ======= Get prediciton of all onsets =======
        len_onset_shape = int(
            SAMPLING_RATE * (TRIM_SECOND_BEFORE + TRIM_SECOND_AFTER))
        onsethop_length = HOP_LENGTH
        onsethop_duration = onsethop_length / SAMPLING_RATE
        n_onset = int(
            np.ceil((len(audio_data) - len_onset_shape) / onsethop_length))
        print("Audio duration: {}s".format(
            librosa.get_duration(y=audio_data, sr=SAMPLING_RATE)))
        print("n_onset: {}".format(n_onset))

        data_full_song = FullSongDataset(
            audio_data, n_onset, len_onset_shape, "onset", onsethop_length)
        loader = DataLoader(data_full_song, batch_size)
        pred_onset, _ = models.run_on_dataset(
            self.onset_model, loader, device=self.device)
        pred_onset = pred_onset.squeeze()
        # print("Onset Prediction:\n{}".format(pred_onset))

        pred_onset_filter = medfilt(pred_onset, 15)
        frmtime_onset = np.arange(n_onset) * \
            onsethop_duration + TRIM_SECOND_BEFORE
        print("Filtered Onset Prediction:\n{}\nMax: {}, Min: {}\n".format(
            pred_onset_filter, np.max(pred_onset_filter), np.min(pred_onset_filter)))

        # ======= Get prediciton of all segments =======
        len_segment_shape = int(SAMPLING_RATE * MIN_SRC)
        seghop_length = HOP_LENGTH * 10
        seghop_duration = seghop_length / SAMPLING_RATE
        n_segment = int(
            np.ceil((len(audio_data) - len_segment_shape) / seghop_length))
        print("n_segment: {}".format(n_segment))

        data_full_song.type_excerpt, data_full_song.n_elem = "segment", n_segment
        pred_segment, _ = models.run_on_dataset(
            self.segment_model, loader, device=self.device)
        pred_segment = pred_segment.squeeze()
        # print("Segment Prediction:\n{}".format(pred_segment))

        pred_segment_filter = medfilt(pred_segment, 3)

        frmtime_segment = np.arange(n_segment) * seghop_duration + MIN_SRC / 2
        audio_firstonsettime = librosa.frames_to_time(
            librosa.onset.onset_detect(y=audio, sr=SAMPLING_RATE), sr=SAMPLING_RATE)[0]
        n_segment_tozero = 0
        for t in frmtime_segment:
            if t < audio_firstonsettime:
                n_segment_tozero += 1
            else:
                break
        print("length frmtime_segment: {}".format(len(frmtime_segment)))
        print("n_segment_tozero: {}".format(n_segment_tozero))
        pred_segment_filter[:n_segment_tozero] = 0
        print("Filtered Segment Prediction:\n{}".format(pred_segment))

        # ======= Fuse Prediction =======
        self.onset_threshold = np.median(pred_onset_filter)
        self.segment_threshold = np.median(pred_segment_filter)

        pred_onset_todetect = np.copy(pred_onset_filter)
        # print(pred_onset_todetect)
        pred_onset_todetect[pred_onset_todetect < self.onset_threshold] = 0
        pred_onset_todetect[pred_onset_todetect >= self.onset_threshold] = 1

        pred_segment_todetect = np.copy(pred_segment_filter)
        pred_segment_todetect[pred_segment_todetect <
                              self.segment_threshold] = 0
        pred_segment_todetect[pred_segment_todetect >=
                              self.segment_threshold] = 1

        # print(pred_segment_todetect.any())
        # print(pred_onset_todetect.any())

        # decide the initial indexes of pedal segment boundary
        onseg_initidxs = []
        offseg_initidxs = []
        for idx, v in enumerate(pred_segment_todetect):
            if idx > 0 and idx < len(pred_segment_todetect) - 1:
                if pred_segment_todetect[idx - 1] == 0 and v == 1 and pred_segment_todetect[idx + 1] == 1:
                    onseg_initidxs.append(idx - 1)
                elif pred_segment_todetect[idx - 1] == 1 and v == 1 and pred_segment_todetect[idx + 1] == 0:
                    offseg_initidxs.append(idx + 1)

        print("onseg_initidxs: {}\n{}".format(
            len(onseg_initidxs), onseg_initidxs))
        print("offseg_initidxs: {}\n{}".format(
            len(offseg_initidxs), offseg_initidxs))

        if offseg_initidxs[0] <= onseg_initidxs[0]:
            del offseg_initidxs[0]
        if onseg_initidxs[-1] >= offseg_initidxs[-1]:
            del onseg_initidxs[-1]

        if (len(onseg_initidxs) != len(offseg_initidxs)) or not len(pedal_offset_gt) or not len(pedal_onset_gt):
            print(" skip!")
        else:
            onseg_idxs = []
            offseg_idxs = []
            for idx in range(len(onseg_initidxs)):
                if onseg_initidxs[idx] < offseg_initidxs[idx]:
                    onseg_idxs.append(onseg_initidxs[idx])
                    offseg_idxs.append(offseg_initidxs[idx])

            if not len(onseg_idxs) or not len(offseg_idxs):
                print("no detection!")
            else:
                # decide the boundary times in seconds, combining the effect of pedal onset
                onseg_times = []
                offseg_times = []
                for idx, onseg_idx in enumerate(onseg_idxs):
                    onponset_idx = onseg_idx * 10 - 5
                    if any(pred_onset_todetect[onponset_idx - 5: onponset_idx + 5]):
                        offseg_idx = offseg_idxs[idx]
                        offseg_times.append(frmtime_segment[offseg_idx])
                        onseg_times.append(frmtime_segment[onseg_idx])
                segintervals_est = np.stack(
                    (np.asarray(onseg_times), np.asarray(offseg_times)), axis=-1)

                # set the ground truth and estimation results frame by frame
                audio_duration = librosa.get_duration(
                    y=audio_data, sr=SAMPLING_RATE)
                n_frames = int(np.ceil(audio_duration / seghop_duration))
                segframes_gt = np.zeros(n_frames)
                segframes_est = np.zeros(n_frames)

                longpseg_idx = np.where(
                    (pedal_offset_gt-pedal_onset_gt) > seghop_duration)[0]
                longseg_onset_gt = pedal_onset_gt[longpseg_idx]
                longseg_offset_gt = pedal_offset_gt[longpseg_idx]
                segintervals_gt = np.stack(
                    (longseg_onset_gt, longseg_offset_gt), axis=-1)

                for idx, onset_t in enumerate(longseg_onset_gt):
                    offset_t = longseg_offset_gt[idx]
                    onset_frm = int(onset_t // seghop_duration)
                    offset_frm = int(offset_t // seghop_duration)
                    segframes_gt[onset_frm:offset_frm] = 1

                for idx, onset_t in enumerate(onseg_times):
                    offset_t = offseg_times[idx]
                    onset_frm = int(onset_t // seghop_duration)
                    offset_frm = int(offset_t // seghop_duration)
                    segframes_est[onset_frm: offset_frm] = 1

                # set the ground truth and estimation results as interval format
                segintervals1_gt, segintervals01_gt, labels_gt = intervals1tointervals01(
                    segintervals_gt, audio_duration)
                segintervals1_est, segintervals01_est, labels_est = intervals1tointervals01(
                    segintervals_est, audio_duration)

            frmtimes = np.arange(n_frames) * seghop_duration
            # left, right = [150, 170]
            plt.figure(figsize=(15, 5))
            librosa.display.waveplot(audio_data, SAMPLING_RATE, alpha=0.8)
            plt.fill_between(frmtimes, 0, 0.5, where=segframes_gt >
                             0, facecolor='green', alpha=0.7, label='ground truth')
            plt.fill_between(frmtimes, -0.5, 0, where=segframes_est >
                             0, facecolor='orange', alpha=0.7, label='estimation')
            # plt.title("Pedal segment detection of {}".format(filename))
            plt.legend()
            # plt.xlim([left,right])
            # plt.show()
            plt.savefig("test")

            return segframes_est


if __name__ == '__main__':
    # Load some test models
    onset_model = models.ConvModel()
    segment_model = models.ConvModel()

    onset_model.load_state_dict(torch.load(
        "test_models/onset_conv.pt")["model_state_dict"])
    segment_model.load_state_dict(torch.load(
        "test_models/segment_conv.pt")["model_state_dict"])

    file_name = "2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav"

    test_audio_file = ORIGINAL_DATA_PATH + file_name + ".flac"
    test_midi_file = ORIGINAL_DATA_PATH + file_name + ".midi"

    # Load any audio file to test
    audio, sr = librosa.load(test_audio_file, sr=SAMPLING_RATE)
    print(audio.shape[0])
    shift = 0
    scale = 1
    start, end = int(
        shift * audio.shape[0] / scale), int((shift + 1) * audio.shape[0] / scale)
    audio = audio[start: end]
    # Load corresponding midi file
    pedal_gt = generate_full_song_gt(test_midi_file)

    manual_fuser = ManualFusion(
        onset_model, segment_model, onset_threshold=0.98, segment_threshold=0.98)
    frame_wise_prediction = manual_fuser.predict(audio, pedal_gt)
