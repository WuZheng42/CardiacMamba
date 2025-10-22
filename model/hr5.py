import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from nndl.rf import organizer as org
from nndl.rf.proc import create_fast_slow_matrix, find_range
from nndl.utils.errors import getErrors
from nndl.utils.utils import extract_video, pulse_rate_from_power_spectral_density


def eval_rgb_rf_model(rgb_root_dir, session_names, rf_root_path, test_files, model, sequence_length1=64,
                      sequence_length2=64,
                      sampling_ratio=4, adc_samples=256, rf_window_size=5, freq_slope=60.012e12,
                      samp_f=5e6, device=torch.device('cpu'), file_name="rgbd_rgb", ppg_file_name="rgbd_ppg.npy"):
    model.eval()
    video_samples = []
    cur_est_ppgs = []  # Initialize as None
    video_samples1 = []

    # Prepare RGB and RF data in advance to avoid repeated loading
    for cur_session in session_names:
        video_sample = {"video_path": os.path.join(rgb_root_dir, cur_session)}
        video_samples.append(video_sample)

    # Iterate over each session

    for cur_video_sample, rf_folder in tqdm(zip(video_samples, test_files), total=len(session_names)):
        cur_video_path = cur_video_sample["video_path"]
        cur_est_ppgs = None

        frames = extract_video(path=cur_video_path, file_str=file_name)
        if frames is None or frames.shape[0] == 0:
            print(f"Warning: No frames found for video {cur_video_path}")
            continue  # Skip this video sample
        target = np.load(os.path.join(cur_video_path, ppg_file_name))
        if target is None or len(target) == 0:
            print(f"Warning: Empty target data for video {cur_video_path}")
            continue  # Skip this video sample

        rf_fptr = open(os.path.join(rf_root_path, rf_folder, "rf.pkl"), 'rb')
        s = pickle.load(rf_fptr)

        # Number of samples is set ot 256 for our experiments
        rf_organizer = org.Organizer(s, 1, 1, 1, 2 * adc_samples)
        rf_frames = rf_organizer.organize()
        # The RF read adds zero alternatively to the samples. Remove these zeros.
        rf_frames = rf_frames[:, :, :, 0::2]

        data_f = create_fast_slow_matrix(rf_frames)
        range_index = find_range(data_f, samp_f, freq_slope, 256)
        temp_window = np.blackman(rf_window_size)
        raw_data = data_f[:, range_index - len(temp_window) // 2:range_index + len(temp_window) // 2 + 1]
        # circ_buffer = raw_data[0:800]
        #
        # # Concatenate extra to generate ppgs of size 3600
        # raw_data = np.concatenate((raw_data, circ_buffer))
        raw_data = np.array([np.real(raw_data), np.imag(raw_data)])
        raw_data = np.transpose(raw_data, axes=(0, 2, 1))
        rf_data = raw_data

        rf_data = np.transpose(rf_data, axes=(2, 0, 1))


        # Iterate over each sequence
        # 外层循环：遍历每一帧
        for cur_frame_num in range(0, frames.shape[0], sequence_length1):
            cur_rgb_cat_frames = None
            cur_rf_cat_frames = None
            # Process RGB frames
            for rgb_frame_num in range(cur_frame_num, min(cur_frame_num + sequence_length1, frames.shape[0])):
                # 处理RGB帧
                cur_rgb_frames = frames[rgb_frame_num, :, :, :]
                cur_frame_cropped = torch.from_numpy(cur_rgb_frames.astype(np.uint8)).permute(2, 0, 1).float()
                cur_frame_cropped = cur_frame_cropped / 255.0  # 归一化
                cur_frame_cropped = cur_frame_cropped.unsqueeze(0).to(device)
                # Concatenate RGB frames
                if cur_rgb_cat_frames is None:
                    cur_rgb_cat_frames = cur_frame_cropped
                else:
                    cur_rgb_cat_frames = torch.cat((cur_rgb_cat_frames, cur_frame_cropped), dim=0)
                # print(f"rgb:{cur_rgb_cat_frames.shape[0]}")
            # print(f"rgb:{cur_rgb_cat_frames}")
            # Process RF frames
            for rf_frame_num in range(cur_frame_num * sampling_ratio,
                                      min((cur_frame_num + sequence_length2) * sampling_ratio, rf_data.shape[0])):
                cur_rf_frame = rf_data[rf_frame_num, :, :]
                cur_rf_frame = torch.tensor(cur_rf_frame).type(torch.float32)/1.255e5
                # Add the T dim
                cur_rf_frame = cur_rf_frame.unsqueeze(0).to(device)
                # Concatenate RF frames
                if cur_rf_cat_frames is None:
                    cur_rf_cat_frames = cur_rf_frame
                else:
                    cur_rf_cat_frames = torch.cat((cur_rf_cat_frames, cur_rf_frame), dim=0)
                # print(f"rf:{cur_rf_cat_frames.shape[0]}")
            # print(f"rf:{cur_rf_cat_frames}")
            # print(f"rf{cur_rf_cat_frames.shape[0]}")
            # print(f"rgb:{cur_rgb_cat_frames.shape[0]}")
            # Pass through the model
            if cur_rgb_cat_frames.shape[0] == sequence_length1 and cur_rf_cat_frames.shape[
                0] == sequence_length2 * sampling_ratio:
                with torch.no_grad():
                    # rgb
                    cur_rgb_cat_frames = cur_rgb_cat_frames.unsqueeze(0)
                    cur_rgb_cat_frames = torch.transpose(cur_rgb_cat_frames, 1, 2)
                    # rf
                    cur_rf_cat_frames = cur_rf_cat_frames.unsqueeze(0)
                    cur_rf_cat_frames = torch.transpose(cur_rf_cat_frames, 1, 2)
                    cur_rf_cat_frames = torch.transpose(cur_rf_cat_frames, 2, 3)
                    IQ_frames = torch.reshape(cur_rf_cat_frames,
                                              (cur_rf_cat_frames.shape[0], -1, cur_rf_cat_frames.shape[3]))
                    # print(f"cur:{len(cur_rgb_cat_frames)}")
                    # print(f"IQ:{len(IQ_frames)}")
                    # print(f"cur:{cur_rgb_cat_frames.shape}")
                    # print(f"IQ:{IQ_frames.shape}")
                    target_shape = (1, 3, 64, 128, 128)
                    x_padded = torch.zeros(target_shape)
                    cur_est_ppg = model(cur_rgb_cat_frames, IQ_frames)
                    # print(f"cur_est_ppg:{cur_est_ppg}")
                    cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()


                if cur_est_ppgs is None:
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)


                # print("cur_est_ppg",cur_est_ppg)
        # save
        cur_video_sample['est_ppgs'] = cur_est_ppgs[0:900]
        cur_video_sample['gt_ppgs'] = target[25:]
        video_samples1.append(cur_video_sample)
    if cur_est_ppgs is None or len(cur_est_ppgs) == 0:
        print(f"Warning: Empty estimated PPG for video sample {cur_video_path}")
    if target is None or len(target) == 0:
        print(f"Warning: Empty ground truth PPG for video sample {cur_video_path}")

    print('All finished!')
    # Evaluate performance
    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    # print(f"Number of video samples: {len(video_samples1)}")
    # print(f"Sample video data (first 5 items): {video_samples[:5]}")

    for index, cur_video_sample in enumerate(video_samples1):
        cur_video_path = cur_video_sample['video_path']
        cur_est_ppgs = cur_video_sample['est_ppgs']
        # print(f"len(cur_est_ppgs):{len(cur_est_ppgs)}")
        # Ensure cur_est_ppgs and cur_gt_ppgs are not empty
        if cur_est_ppgs is None or len(cur_est_ppgs) == 0:
            print(f"Warning: Empty estimated PPG for video sample {cur_video_path}")
            continue  # Skip this video sample
        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        # Ensure cur_est_ppgs and cur_gt_ppgs are not empty
        if cur_est_ppgs is None or len(cur_est_ppgs) == 0:
            print(f"Warning: Empty estimated PPG for video sample {cur_video_path}")
            continue  # Skip this video sample
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)

        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)
        all_ppg_est=[]
        all_ppg_gt=[]
        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        # print(len(cur_est_ppgs))
        # print("hh", len(cur_est_ppgs) - hr_window_size)
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]

            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]
            # 打印每个窗口的数据，检查是否为空
            # print(f"ppg_est_window: {ppg_est_window}, ppg_gt_window: {ppg_gt_window}")

            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            all_ppg_est.append(ppg_est_window)
            all_ppg_gt.append(ppg_gt_window)
        # print(f"hr_est: {hr_est_temp}, hr_gt: {hr_gt_temp}")
        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)
        # print(f"hr_est_temp: {hr_est_temp}, hr_gt_temp: {hr_gt_temp}")
        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)
        
       

        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)

        mae_list.append(MAE)
    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt),all_ppg_est,all_ppg_gt


