import sys
import os

# 明确将项目根目录（/root/autodl-tmp/EquiPleth-main）添加到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

print("项目根目录已添加到路径:", project_root)  # 用于调试，确认路径设置正确

import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import yaml

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from nndl.model.rf_diff_miss import *


from nndl.losses.NegPearsonLoss import Neg_Pearson
from nndl.utils.eval import eval_rgb_model
from nndl.model.rit import *
import torch.nn.init as init

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from nndl.rf.model import RF_conv_decoder
from nndl.rf.proc import rotateIQ
from nndl.data.datasets1 import *
from nndl.losses.NegPearsonLoss import Neg_Pearson
from nndl.losses.SNRLoss import SNRLoss_dB_Signals
from nndl.utils.eval import eval_rf_model

CONFIG_PATH = "/root/autodl-tmp/EquiPleth-main/config/config.yaml"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt



def rmse(l1, l2):
    return np.sqrt(np.mean((l1 - l2) ** 2))


def parseArgs():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description='RGB-RF 训练脚本的配置')

    parser.add_argument('-rgbdir', '--rgb-data-dir', type=str, default="/root/autodl-tmp/DATA/rgb_files",
                        help="包含RGB数据的目录。")

    parser.add_argument('-rfdir', '--rf-data-dir', type=str, default="/root/autodl-tmp/DATA/rf_files",
                        help="包含RF数据的目录。")

    parser.add_argument('--folds-path', type=str,
                        default="/root/autodl-tmp/DATA/demo_fold.pkl",
                        help='包含折叠数据的pickle文件。')

    parser.add_argument('--fold', type=int, default=0,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default=None,
                        help="Device on which the model needs to run (input to torch.device). \
                                  Don't specify for automatic selection. Will be modified inplace.")

    parser.add_argument('-ckpt', '--checkpoint-path', type=str,
                        default="/root/autodl-tmp/EquiPleth-main/ckpt/9/best.pth",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    return parser.parse_args(), config


def main(args, config):
    ckpt_path = args.checkpoint_path
    mamba_config = config["mmmamba"]["mamba_config"]
    # 加载折叠数据
    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    test_files1 = files_in_fold[args.fold]["test"]
    test_files2 = files_in_fold[args.fold]["test"]
    test_files2 = [i[2:] for i in test_files2]
    train_files1 = files_in_fold[args.fold]["train"]
    train_files2 = files_in_fold[args.fold]["train"]
    train_files2 = [i[2:] for i in train_files2]
    # dataset_test1 = RGBData(datapath=args.rgb_data_dir, datapaths=test_files1)
    # dataset_test2 = RFDataRAMVersion(datapath=args.rf_data_dir, datapaths=test_files2)
    # dataset_test = FullData(rgb_datapath=args.rgb_data_dir, rgb_datapaths=test_files1, rf_datapath=args.rf_data_dir,
    #                          rf_datapaths=test_files2)
    dataset_test = FullData(rgb_datapath=args.rgb_data_dir, rgb_datapaths=train_files1, rf_datapath=args.rf_data_dir,
                            rf_datapaths=train_files2)

    # 设备选择
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    if args.verbose:
        print('在设备上运行: {}'.format(args.device))

    # datasets1 = {"test": dataset_test1}
    # datasets2 = {"test": dataset_test2}
    datasets = {"test": dataset_test}
    model = FusionNet(mamba_config=mamba_config).to(args.device)
    # model.load_state_dict(torch.load(ckpt_path))
    # 加载完整的检查点
    checkpoint = torch.load("/root/autodl-tmp/EquiPleth-main/ckpt/ppg/best.pth")

    # 从检查点中提取模型的 state_dict
    state_dict = checkpoint['model_state_dict']

    # 加载模型参数
    model.load_state_dict(state_dict)
    mae_loss_list,  (all_hr_est, all_hr_gt), all_ppg_est,all_ppg_gt = eval_rgb_rf_model(rgb_root_dir=args.rgb_data_dir,  # 要修改
                                                               session_names=test_files1,
                                                               rf_root_path=args.rf_data_dir,  # 要修改
                                                               test_files=test_files2, model=model,
                                                               device=args.device)



        # 将当前视频的估计PPG和真值PPG添加到总列表中
    # 设置采样率和持续时间参数
    samples_per_second = 30  # 根据实际采样率调整
    time_duration = 5  # 截取前5秒
    samples_to_show = samples_per_second * time_duration
    # 转换为numpy数组并展平
    ppg_est_array = np.array(all_ppg_est).flatten()
    ppg_gt_array = np.array(all_ppg_gt).flatten()

    # 截取前150个点（前5秒）
    ppg_est_short = ppg_est_array[:samples_to_show]
    ppg_gt_short = ppg_gt_array[:samples_to_show]

    # 创建时间轴（单位：秒），仅生成前150个点对应的时间
    time_axis = np.arange(samples_to_show) / samples_per_second

    # 绘制波形图
    plt.figure(figsize=(16, 10))
    plt.plot(time_axis, ppg_est_short, label="Predicted (PPG)", color='b', linewidth=2)
    plt.plot(time_axis, ppg_gt_short, label="Ground Truth (PPG)", color='r', linewidth=2)

    # 设置坐标轴和标签
    plt.title(f"PPG Waveform ")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(False)

    # 显示并保存
    plt.tight_layout()
    plt.savefig("/root/autodl-tmp/EquiPleth-main/nndl/fig/13.png")
    plt.show()
if __name__ == '__main__':
    args, config = parseArgs()  # 解包元组，获取 `args` 和 `config`
    main(args, config)  # 将 `args` 和 `config` 分别传递给 `main` 函数



