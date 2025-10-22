import sys
import os

# 明确将项目根目录（/root/autodl-tmp/EquiPleth-main）添加到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

print("项目根目录已添加到路径:", project_root)  # 用于调试，确认路径设置正确
from fvcore.nn import FlopCountAnalysis
import numpy as np  # 确保导入 NumPy
import torch
import torch.nn as nn

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
from nndl.model.hr4 import *
import torch.nn.init as init



from nndl.rf.model import RF_conv_decoder
from nndl.rf.proc import rotateIQ
from nndl.data.datasets1 import *
from nndl.losses.NegPearsonLoss import Neg_Pearson
from nndl.losses.SNRLoss import SNRLoss_dB_Signals
from nndl.utils.eval import eval_rf_model
CONFIG_PATH = "/root/autodl-tmp/EquiPleth-main/config/config.yaml"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def parseArgs():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description='RGB-RF 训练脚本的配置')

    parser.add_argument('-rgbdir', '--rgb-data-dir', type=str, default="/root/autodl-tmp/DATA/rgb_files",
                        help="包含RGB数据的目录。")

    parser.add_argument('-rfdir', '--rf-data-dir', type=str, default="/root/autodl-tmp/DATA/rf_files",
                        help="包含RF数据的目录。")

    parser.add_argument('--save-dir', type=str, default="/root/autodl-tmp/ckpt3/dataset1",
                        help="保存输出的目录。")

    parser.add_argument('--folds-path', type=str,
                        default="/root/autodl-tmp/DATA/demo_fold.pkl",
                        help='包含折叠数据的pickle文件。')

    parser.add_argument('--fold', type=int, default=0,
                        help='折叠编号')

    parser.add_argument('--device', type=str, default=None,
                        help="运行模型的设备。如果为None，将自动选择。")

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch Size for the dataloaders.")

    parser.add_argument('--num-workers', type=int, default=2,
                        help="Number of Workers for the dataloaders.")
    parser.add_argument('-ckpts', '--checkpoints-path', type=str,
                        default="/root/autodl-tmp/EquiPleth-main/ckpt/ppg",
                        help='Checkpoint Folder.')

    parser.add_argument('--train-shuffle', action='store_true', help="Shuffle the train loader.")
    parser.add_argument('--val-shuffle', action='store_true', help="Shuffle the val loader.")
    parser.add_argument('--test-shuffle', action='store_true', help="Shuffle the test loader.")

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003,
                        help="Learning Rate for the optimizer.")

    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-2,
                        help="Weight Decay for the optimizer.")

    parser.add_argument('--epochs', type=int, default=30, help="Number of Epochs.")

    parser.add_argument('--checkpoint-period', type=int, default=1,
                        help="Checkpoint save period.")

    parser.add_argument('--epoch-start', type=int, default=1,
                        help="Starting epoch number.")

    parser.add_argument('--multi-gpu', action='store_true', help="Use multiple GPUs if available.")

    return parser.parse_args(), config


def save_checkpoint(model, optimizer, epoch, path):
    state_dict = model.state_dict()
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)




def train_model(args, model, datasets):

    train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size,
                                   shuffle=args.train_shuffle, num_workers=args.num_workers)
    val_dataloader = DataLoader(datasets["val"], batch_size=args.batch_size,
                                 shuffle=args.val_shuffle, num_workers=args.num_workers)
    test_dataloader = DataLoader(datasets["test"], batch_size=args.batch_size,
                                  shuffle=args.test_shuffle, num_workers=args.num_workers)



    # print(f"Train dataset 1 size: {len(datasets1['train'])}")
    # print(f"Train dataset 2 size: {len(datasets2['train'])}")
    # print(f"Train loader 1 batches: {len(train_dataloader1)}")
    # print(f"Train loader 2 batches: {len(train_dataloader2)}")
    # 检查 RGB 数据和 RF 数据的形状
    # for batch_idx, ((imgs, signal1), (rf, signal2)) in enumerate(zip(train_dataloader1, train_dataloader2)):
        #print(f"Batch {batch_idx}:")
        #print(f"RGB Image shape: {imgs}")
        # print(f"Signal shape (RGB): {signal1}")
        #print(f"RF data : {rf}")
        # print(f"Signal shape (RF): {signal2}")
    # 检查模型参数是否参与训练
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")


    if args.verbose:
        print(f"训练迭代次数: {len(train_dataloader)}")
        print(f"验证迭代次数: {len(val_dataloader)}")
        print(f"测试迭代次数: {len(test_dataloader)}")

    if len(train_dataloader) == 0 :
        print("Train dataloaders are empty. Please check your data paths or dataset configuration.")
        return  # 或者 raise 一个异常

    ckpt_path = args.checkpoints_path
    latest_ckpt_path = "/root/autodl-tmp/EquiPleth-main/ckpt/16/5.pth"

    # 损失函数和优化器
    loss_fn1 = Neg_Pearson()
    loss_fn2 = SNRLoss_dB_Signals()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 打印 datasets1["val"].video_list 和 datasets2["val"].rf_file_list 的内容
    # print("Datasets1 - Validation Video List:")
    # print(datasets1["val"].video_list)  # 输出 RGB 数据集的验证集的视频列表
    #
    # print("Datasets2 - Validation RF File List:")
    # print(datasets2["val"].rf_file_list)  # 输出 RF 数据集的验证集的 RF 文件列表

    # 加载检查点（如果存在）
    if os.path.exists(latest_ckpt_path):
        print('存在检查点，正在加载状态字典。')
        checkpoint = torch.load(latest_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        # epoch_start = 6
    else:
        epoch_start = args.epoch_start

    best_loss = np.inf
    epochs = args.epochs
    checkpoint_period = args.checkpoint_period

    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(epoch_start, epochs + 1):
        model.train()
        loss_train = 0

        no_batches = 0



        for batch, (imgs, rf, signal) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # 处理RGB图像
            imgs = imgs.type(torch.float32) / (255)
            # imgs = imgs.type(torch.float32)
            imgs = imgs.to(args.device)


            # 处理RF数据
            rf = rf.type(torch.float32) / 1.255e5
            # rf = rf.type(torch.float32)
            rf = rotateIQ(rf)
            rf = torch.reshape(rf, (rf.shape[0], -1, rf.shape[3])).to(args.device)
            # print(f"img:{imgs.shape}")
            # print(f"rf:{rf.shape}")
            # print(f"signal1.shape:{signal.shape}")
            # print(f"signal2.shape:{signal2.shape}")


            signal = signal.type(torch.float32).to(args.device)

            # 前向传播
            pred_signal = model(imgs, rf)


            loss = loss_fn1(pred_signal, signal)



            #print(f"pred_signal:{pred_signal}")
            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 手动检查梯度
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_mean = param.grad.abs().mean().item()  # 计算梯度的平均值
            #         grad_max = param.grad.abs().max().item()  # 计算梯度的最大值
            #         print(f'Layer {name}: Gradients mean = {grad_mean}, max = {grad_max}')

            optimizer.step()

            # 累积损失
            loss_train += loss.item()

            no_batches += 1





        # 每隔几个epoch保存模型
        if epoch % checkpoint_period == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(os.getcwd(), f"{ckpt_path}/{epoch}.pth"))
            mae_loss_list, (all_hr_est, all_hr_gt) = eval_rgb_rf_model(rgb_root_dir = args.rgb_data_dir,#要修改
                                                                       session_names = datasets["val"].rgb_video_list,
                                                                       rf_root_path = args.rf_data_dir, #要修改
                                                                       test_files = datasets["val"].rf_file_list, model = model, device=args.device)
            valid_indices = ~np.isnan(mae_loss_list)
            mae_loss_list = mae_loss_list[valid_indices]

            if len(mae_loss_list) > 0:
                current_loss = np.mean(mae_loss_list)
            else:
                current_loss = np.inf

            if (current_loss < best_loss):
                best_loss = current_loss
                save_checkpoint(model, optimizer, epoch, os.path.join(os.getcwd(), f"{ckpt_path}/best.pth"))
                print("Best checkpoint saved!")
            print("Saved Checkpoint!")
        print(f"Epoch {epoch}: Training Loss: {loss_train / no_batches:.4f} ")
            # 计算验证损失（val_loss）
            # model.eval()  # 切换为评估模式
            # val_loss = 0
            # val_batches = 0
            # with torch.no_grad():
            #     val_loader = zip(val_dataloader1, val_dataloader2)
            #     for (imgs, signal1), (rf, signal2) in tqdm(val_loader, total=len(val_dataloader1)):
            #         imgs = imgs.type(torch.float32) / (255)
            #         rf = rf.type(torch.float32) / 1.255e5
            #         rf = rotateIQ(rf)
            #         rf = torch.reshape(rf, (rf.shape[0], -1, rf.shape[3])).to(args.device)
            #         signal = signal1.to(args.device)
            #
            #         pred_signal = model(imgs, rf)
            #         loss = loss_fn(pred_signal, signal)
            #
            #         val_loss += loss.item()
            #         val_batches += 1
            #
            # val_loss = val_loss / val_batches
            # print(f"Epoch {epoch}: Training Loss: {loss_train / no_batches:.4f} | Validation Loss: {val_loss:.4f}")
            #

def main(args, config):
    mamba_config = config["mmmamba"]["mamba_config"]
    print(config['mmmamba']['mamba_config'])
    # 加载折叠数据
    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    # RGB数据文件
    train_files1 = files_in_fold[args.fold]["train"]
    val_files1 = files_in_fold[args.fold]["val"]
    test_files1 = files_in_fold[args.fold]["test"]

    # RF数据文件（假设存储方式类似）
    train_files2 = files_in_fold[args.fold]["train"]
    train_files2 = [i[2:] for i in train_files2]
    val_files2 = files_in_fold[args.fold]["val"]
    val_files2 = [i[2:] for i in val_files2]
    test_files2 = files_in_fold[args.fold]["test"]
    test_files2 = [i[2:] for i in test_files2]

    print(f"RF train files: {train_files2}")

    # 如果需要，替换为实际的路径
    # 例如，如果RF文件有不同的前缀或目录结构
    # train_files2 = [modify_path(i) for i in train_files1]

    if args.verbose:
        print(f"训练文件: {train_files1}")
        print(f"验证文件: {val_files1}")
        print(f"测试文件: {test_files1}")

    dataset_train = FullData(rgb_datapath=args.rgb_data_dir, rgb_datapaths=train_files1,rf_datapath=args.rf_data_dir,rf_datapaths=train_files2)
    dataset_val = FullData(rgb_datapath=args.rgb_data_dir, rgb_datapaths=val_files1,rf_datapath=args.rf_data_dir,rf_datapaths=val_files2)
    dataset_test = FullData(rgb_datapath=args.rgb_data_dir, rgb_datapaths=test_files1,rf_datapath=args.rf_data_dir,rf_datapaths=test_files2)

    # 设备选择
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    if args.verbose:
        print('在设备上运行: {}'.format(args.device))

    # 可视化示例（如果需要）
    if args.viz:
        imgs, signal = dataset_train1[0]
        rf, _ = dataset_train2[0]

        plt.figure()
        plt.imshow(np.transpose(imgs[0], (1, 2, 0)))
        plt.title("示例RGB图像")
        plt.figure()
        plt.plot(signal)
        plt.title("示例信号")

        # 如果可能，可视化RF数据
        # plt.figure()
        # plt.plot(rf[0])
        # plt.title("示例RF数据")

        plt.show()

    # 创建检查点目录（如果不存在）
    os.makedirs(args.checkpoints_path, exist_ok=True)

    datasets = {"train": dataset_train, "val": dataset_val, "test": dataset_test}


    # 初始化模型
    model = FusionNet(mamba_config=mamba_config).to(args.device)
    # 调用 Kaiming 初始化函数
    #model.apply(initialize_weights_kaiming)

    train_model(args, model, datasets)

if __name__ == '__main__':

    args, config = parseArgs()  # 解包元组，获取 `args` 和 `config`
    main(args, config)  # 将 `args` 和 `config` 分别传递给 `main` 函数
