import datetime
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU, ConvGru, FFN
from src.metrics import MSE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp  # 1、引入FontProperties
import math

# Training parameters
#DATASET_PATH = 'C:\\Users\\14344\Desktop\\数据集\\UCRArchive_2018\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv' # 数据集路径
train_path = 'F:\\Data\\UCRArchive_2018\\ACSF1\\ACSF1_TRAIN.tsv' # 数据集路径
test_path = 'F:\\Data\\UCRArchive_2018\\ACSF1\\ACSF1_TEST.tsv'  # 数据集路径
BATCH_SIZE = 20
NUM_WORKERS = 0
LR = 2e-5
EPOCHS = 200
test_interval = 10
data_length_p = 100

# Model parameters
d_model = 256  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 4  # Number of encoder and decoder to stack
attention_size = 12  # Attention window size
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_input = 1460  # From dataset
d_output = 10  # From dataset

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load dataset
dataset_train = OzeDataset(train_path)
dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=False)
dataset_test = OzeDataset(test_path)
dataloader_test = Data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# ozeDataset = OzeDataset(DATASET_PATH)
#
# # Split between train, validation and test
# dataset_train, dataset_val, dataset_test = random_split(
#     ozeDataset, (38000, 1000, 1000))
#
# dataloader_train = DataLoader(dataset_train,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True,
#                               num_workers=NUM_WORKERS,
#                               pin_memory=False
#                               )
#
# dataloader_val = DataLoader(dataset_val,
#                             batch_size=BATCH_SIZE,
#                             shuffle=True,
#                             num_workers=NUM_WORKERS
#                             )
#
# dataloader_test = DataLoader(dataset_test,
#                              batch_size=BATCH_SIZE,
#                              shuffle=False,
#                              num_workers=NUM_WORKERS
#                              )

# Load transformer with Adam optimizer and MSE loss function
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = OZELoss()

# metrics = {
#     'training_loss': lambda y_true, y_pred: OZELoss(alpha=0.3, reduction='none')(y_true, y_pred).numpy(),
#     'mse_tint_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none'),
#     'mse_cold_total': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none'),
#     'mse_tint_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[-1], reduction='none', occupation=occupation),
#     'mse_cold_occupation': lambda y_true, y_pred: MSE(y_true, y_pred, idx_label=[0, 1, 2, 3, 4, 5, 6], reduction='none', occupation=occupation),
#     'r2_tint': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, -1], y_pred[:, i, -1]) for i in range(y_true.shape[1])]),
#     'r2_cold': lambda y_true, y_pred: np.array([r2_score(y_true[:, i, 0:-1], y_pred[:, i, 0:-1]) for i in range(y_true.shape[1])])
# }

# logger = Logger(f'logs/training.csv', model_name=net.name,
#                 params=[y for key in metrics.keys() for y in (key, key+'_std')])


# Fit model
# with tqdm(total=EPOCHS) as pbar:
#     loss = fit(net, optimizer, loss_function, dataloader_train,
#                 epochs=EPOCHS, pbar=pbar, device=device)

    # if ((epochs + 1) % 100) == 0:

correct_list = []
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in dataloader_test:
            enc_inputs, dec_inputs = x_test.to(device), y_test.to(device)
            test_outputs = net(enc_inputs)
                # test_out = list(test_outputs)
                # test_output = torch.cat(test_out,dim=0)
                # test_output = torch.as_tensor(outputs)
                # import pdb;
                # pdb.set_trace()
            _, predicted = torch.max(test_outputs.data, dim=1)
            total += dec_inputs.size(0)
            correct += (predicted == dec_inputs).sum().item()
        correct_list.append((100 * correct / total))
        print('Accuracy on test set: %d %%' % (100 * correct / total))

val_loss_best = np.inf
loss_list = []
best_correct = 0
best_dropout = 0
best_model = 0
begin_time = time()

# Prepare loss history
with tqdm(total=EPOCHS) as pbar:
    for d_model in [128, 256, 512, 1024]:
        for dropout in [0.01, 0.1, 0.2, 0.3]:
            d_model = d_model
            dropout = dropout
            net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                        dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
            for idx_epoch in range(EPOCHS):
                for idx_batch, (x, y) in enumerate(dataloader_train):
                    optimizer.zero_grad()

                    # Propagate input
                    netout = net(x.to(device))  # [8,1460]

                    # Comupte loss
                    loss = loss_function(y.to(device), netout)
                    print('Epoch:', '%04d' % (idx_epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    loss_list.append(loss.item())

                    # Backpropage loss
                    loss.backward()

                    # Update weights
                    optimizer.step()

                if ((idx_epoch+1)%test_interval) == 0:
                    test()
                    if max(correct_list) > best_correct:
                        best_correct = max(correct_list)
                        best_model = d_model
                        best_dropout = dropout
                val_loss = compute_loss(net, dataloader_train, loss_function, device).item()

                if val_loss < val_loss_best:
                    val_loss_best = val_loss

                if pbar is not None:
                    pbar.update()



    print("The best d_model is ",best_model)
    print("The best dropout is",best_dropout)


    # print('\r\n', loss_list)
    # print(correct_list)

end_time = time()
time_cost = round((end_time - begin_time) / 60, 2)

# 结果可视化 包括绘图和结果打印
def result_visualization():
    my_font = fp(fname=r"C:\windows\Fonts\msyh.ttc")  # 2、设置字体路径

    # 设置风格
    # plt.style.use('ggplot')
    plt.style.use('seaborn')

    fig = plt.figure()  # 创建基础图
    ax1 = fig.add_subplot(311)  # 创建两个子图
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # 添加折线
    ax2.plot(correct_list)

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')


    # 设置文本
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                             f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}' '    '
                             f'最后一轮loss:{loss_list[-1]}' '\n'
                             f'最大correct：{max(correct_list)}%' '    '
                             f'最大correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}' '    '
                             f'最后一轮correct：{correct_list[-1]}%' '\n'
                             f'd_model={d_model}   q={q}   v={v}   h={h}   N={N} attention_size={attention_size} drop_out={dropout}' '\n'
                             f'共耗时{round(time_cost, 2)}分钟' , FontProperties=my_font)

    # 保存结果图   测试不保存图（epoch少于200）
    if EPOCHS > 200:
        plt.savefig(f'result_figure/{optimizer} epoch={EPOCHS} batch={BATCH_SIZE} lr={LR} [{d_model},{q},{v},{h},{N},{attention_size},{dropout}].png')

    # 展示图
    plt.show()

    print('正确率列表', correct_list)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((data_length_p / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：{max(correct_list)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_list.index(max(correct_list)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_list[-1]}')

    print(f'共耗时{round(time_cost, 2)}分钟')


# 调用结果可视化
result_visualization()



# end_time = time()
# time_cost = (end_time - begin_time) / 60
# Switch to evaluation
_ = net.eval()


# # Select target values in test split
# y_true = ozeDataset._y[dataloader_test.dataset.indices]
#
# # Compute predictions
# predictions = torch.empty(len(dataloader_test.dataset), 168, 8)
# idx_prediction = 0
# with torch.no_grad():
#     for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
#         netout = net(x.to(device)).cpu()
#         predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
#         idx_prediction += x.shape[0]
#
# # Compute occupation times
# occupation = ozeDataset._x[dataloader_test.dataset.indices,
#                            :, ozeDataset.labels['Z'].index('occupancy')]
#
# results_metrics = {
#     key: value for key, func in metrics.items() for key, value in {
#         key: func(y_true, predictions).mean(),
#         key+'_std': func(y_true, predictions).std()
#     }.items()
# }
#
# # Log
# logger.log(**results_metrics)
#
# # Save model
# torch.save(net.state_dict(),
#            f'models/{net.name}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth')
