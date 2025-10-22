import os
import pickle
import numpy as np
import imageio
import scipy.signal as sig
from torch.utils.data import Dataset
import cv2
import random
import torch
import torch.nn.functional as F

import nndl.rf.organizer as org
from nndl.rf.proc import create_fast_slow_matrix, find_range

from scipy.signal import butter, filtfilt


def low_pass_filter(data, cutoff, fs, order=5):
    """
    对信号进行低通滤波以防止混叠。

    参数:
    - data: 要滤波的信号（numpy 数组）。
    - cutoff: 截止频率（Hz）。
    - fs: 采样频率（Hz）。
    - order: 滤波器阶数。

    返回:
    - 滤波后的信号。
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=-1)
    return y


class FullData(Dataset):
    def __init__(self, rgb_datapath, rgb_datapaths, rf_datapath, rf_datapaths, recording_str="rgbd_rgb", ppg_str="rgbd",
                 video_length=900, frame_length=64, sampling_ratio=4,window_size=5, samples=256, samp_f=5e6,
                 freq_slope=60.012e12, static_dataset_samples=30) -> None:
        self.samp_f = samp_f
        self.sampling_ratio = sampling_ratio
        # # 下采样参数
        # self.downsample_factor = 4  # 因为 RF 采样率是 PPG 的 4 倍
        # self.downsampled_samp_f = self.samp_f / self.downsample_factor
        #
        # # 低通滤波的截止频率设置为 PPG 的一半采样率（Nyquist 频率）
        # self.cutoff_freq = self.downsampled_samp_f / 2  # 这里假设 PPG 的最高频率小于 cutoff_freq

        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 25
        # Number of samples to be created by oversampling one trial.
        self.num_samps = 30
        # Name of the files being read. Name depends on how the file was save. We have saved the file as rgbd_rgb
        self.id_str = recording_str
        self.ppg_str = ppg_str
        # Number of frames in the input video. (Requires all data-samples to have the same number of frames).
        self.video_length = video_length
        # Number of frames in the output tensor sample.
        self.frame_length = frame_length

        # Data structure for videos.
        self.rgb_datapath = rgb_datapath
        # Load videos and signals.
        self.rgb_video_list = rgb_datapaths
        # The PPG files for the RGB are stored as rgbd_ppg and not rgbd_rgb_ppg.
        # Data structure for videos.
        self.rf_datapath = rf_datapath
        # Load videos and signals.
        self.rf_file_list = rf_datapaths
        self.signal_list = []
        # Load signals
        remove_folders = []
        for folder in self.rgb_video_list:
            file_path = os.path.join(rgb_datapath, folder)
            # Make a list of the folder that do not have the PPG signal.
            if (os.path.exists(file_path)):
                if (os.path.exists(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))):
                    signal = np.load(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))
                    self.signal_list.append(signal[self.ppg_offset:])
                else:
                    print(folder, "ppg doesn't exist.")
                    remove_folders.append(folder)
            else:
                print(folder, " doesn't exist.")
                remove_folders.append(folder)
        # Remove the PPGs
        for i in remove_folders:
            self.rgb_video_list.remove(i)
            print("Removed", i)

        # Extract the stats for the vital signs.
        self.signal_list = np.array(self.signal_list)
        self.vital_mean = np.mean(self.signal_list)
        self.vital_std = np.std(self.signal_list)
        self.signal_list = (self.signal_list - self.vital_mean) / self.vital_std

        # Create a list of video number and valid frame number to extract the data from.
        self.video_nums = np.arange(0, len(self.rgb_video_list))
        self.frame_nums = np.arange(0, self.video_length - frame_length - self.ppg_offset)
        # Save the RF config parameters.
        self.window_size = window_size
        self.samples = samples

        self.freq_slope = freq_slope

        # Window the PPG and the RF samples.
        self.ppg_signal_length = video_length
        self.frame_length_ppg = frame_length


        # Create all possible sampling combinations.
        self.all_idxs = []
        for num in self.video_nums:
            # Generate the start index.
            cur_frame_nums = np.random.randint(low=0,
                                               high=self.video_length - frame_length - self.ppg_offset,
                                               size=self.num_samps)
            rf_cur_frame_nums = cur_frame_nums * 4
            # Append all the start indices.
            for rf_frame_num, cur_frame_num in zip(rf_cur_frame_nums, cur_frame_nums):
                self.all_idxs.append((num,(rf_frame_num, cur_frame_num)))

            # High-ram, compute FFTs before starting training.
        self.rf_data_list = []
        for rf_file in self.rf_file_list:
            # Read the raw RF data
            rf_fptr = open(os.path.join(self.rf_datapath, rf_file, "rf.pkl"), 'rb')
            s = pickle.load(rf_fptr)

            # Organize the raw data from the RF.
            # Number of samples is set ot 256 for our experiments.
            rf_organizer = org.Organizer(s, 1, 1, 1, 2 * self.samples)
            frames = rf_organizer.organize()
            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:, :, :, 0::2]

            # # 应用低通滤波器
            # frames = low_pass_filter(frames, self.cutoff_freq, self.samp_f, order=5)
            # # 下采样 RF 数据
            # frames = frames[:, :, :, ::self.downsample_factor]  # 每隔4个样本取一个

            # Process the organized RF data
            data_f = create_fast_slow_matrix(frames)
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples)
            # Get the windowed raw data for the network
            raw_data = data_f[:, range_index - self.window_size // 2:range_index + self.window_size // 2 + 1]
            # Note that item is a complex number due to the nature of the algorithm used to extract and process the pickle file.
            # Hence for simplicity we separate the real and imaginary parts into 2 separate channels.
            raw_data = np.array([np.real(raw_data), np.imag(raw_data)])
            raw_data = np.transpose(raw_data, axes=(0, 2, 1))

            self.rf_data_list.append(raw_data)

    def __len__(self):
        return int(len(self.all_idxs))

    def __getitem__(self, idx):
        # Get the video number and the starting frame index.
        video_number, (rf_start,frame_start) = self.all_idxs[idx]
        # Get video frames for the output video tensor.
        # (Expects each sample to be stored in a folder with the sample name. Each frame is stored as a png)
        rgb_item = []
        for img_idx in range(self.frame_length):
            image_path = os.path.join(self.rgb_datapath,
                                      str(self.rgb_video_list[video_number]),
                                      f"{self.id_str}_{frame_start + img_idx}.png")
            rgb_item.append(imageio.imread(image_path))
        rgb_item = np.array(rgb_item)

        # Add channel dim if no channels in image.
        if (len(rgb_item.shape) < 4):
            rgb_item = np.expand_dims(rgb_item, axis=3)
        rgb_item = np.transpose(rgb_item, axes=(3, 0, 1, 2))

        # Get the RF data.
        data_f = self.rf_data_list[video_number]
        data_f = data_f[:, :, rf_start: rf_start + (self.sampling_ratio * self.frame_length_ppg)]
        rf_item = data_f
        # Get signal.
        item_sig = self.signal_list[int(video_number)][int(frame_start):int(frame_start + self.frame_length)]

        # Patch for the torch constructor. uint16 is a not an acceptable data-type.
        if (rgb_item.dtype == np.uint16):
            rgb_item = item.astype(np.int32)
        return np.array(rgb_item),np.array(rf_item), np.array(item_sig)



    def lowPassFilter(self, BVP, butter_order=4):
        [b, a] = sig.butter(butter_order, [self.l_freq_bpm / 60, self.u_freq_bpm / 60], btype='bandpass', fs=self.fs)
        filtered_BVP = sig.filtfilt(b, a, np.double(BVP))
        return filtered_BVP
