import os.path

import numpy as np
import torch
from typing import List, Tuple
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import OrderedDict
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
from tensorflow import keras


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=1.0, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class CNN_V1(torch.nn.Module):
    """
    平滑单特征卷积回归网络

    """
    def __init__(self, input_size: int, step: int, output_size: int = 1,
                 smooth_kernel_vector: tuple = (0.4, 0.8, 1, 0.8, 0.4)):
        """
        初始化函数

        :param input_size: int,
            输入数据中特征的数量
        :param step: int,
            输入数据中序列的长度
        :param output_size: int, optional=1
            输出结果的维度
        :param smooth_kernel_vector: tuple, option=(0.4, 0.8, 1, 0.8, 0.4)
            用于平滑权重序列的平滑核，也就说加权移动平均的权值向量。
        """
        super(CNN_V1, self).__init__()
        self.input_size = input_size
        # 权重差分序列计算的参数组
        self.constant_vector = torch.from_numpy(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)).cuda()
        self.up_conv_list = []
        for k in range(input_size):
            item_up_conv = torch.nn.Sequential(
                torch.nn.Linear(in_features=3, out_features=10),
                torch.nn.ReLU(True),
                torch.nn.Linear(in_features=10, out_features=step)
            )
            self.up_conv_list.append((f'UpConv-{k}', item_up_conv))
        self.up_conv_list = torch.nn.Sequential(OrderedDict(self.up_conv_list))
        # 从权重差分序列到权重序列的平滑矩阵
        smooth_kernel_vector = np.array(smooth_kernel_vector, dtype=np.float32)
        zero_matrix = np.zeros((step, step + 4), dtype=np.float32)
        for k in range(step):
            zero_matrix[k, k:k+5] = smooth_kernel_vector
        smooth_kernel_matrix = zero_matrix[:, 2: step+2]
        self.smooth_kernel_matrix = torch.from_numpy(smooth_kernel_matrix).cuda()
        # 平滑卷积处理之后的全连接网络
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 10),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, output_size)
        )
        # 用于记录每一个特征上权重序列的列表
        self.smoothed_conv_kernel_list = [torch.FloatTensor() for _ in range(input_size)]

    def forward(self, x):
        # 交换Seq和Feature的顺序
        x = torch.permute(x, dims=(0, 2, 1))
        # 计算每一个特征上的平滑卷积结果
        outs = []
        for k in range(self.input_size):
            # 截取单一特征列
            item_data = x[:, k:k+1, :]
            # 生成权重差分序列
            item_kernel_1 = self.up_conv_list[k](self.constant_vector)
            # 生成权重序列
            item_kernel_2 = torch.matmul(item_kernel_1, self.smooth_kernel_matrix)
            # 记录本轮产生的权重序列
            self.smoothed_conv_kernel_list[k] = item_kernel_2
            # 将权重序列与数据序列相乘求和，并记录
            outs.append(torch.sum(item_data * item_kernel_2, dim=2, keepdim=False))
        # 拼接单特征输出值，全连接映射到最终输出
        outs = torch.concat(outs, dim=1)
        return self.fc(outs)


class RayleighEstimator(object):
    """
    该类用于估计序列对序列学习问题中特征序列对于目标序列的单位冲击响应曲线参数（纯时滞长度系数，容量时滞释放系数）
    """
    # 开始对权重序列进行形状强迫的回合数
    epochs_to_start_smooth = 150
    # 学习速率的下降比例
    learning_rate_decay_ratio = 1.3
    # 学习速率的下限
    learning_rate_down_bound = 0.0001
    # 过程图形文件的存储文件夹
    png_save_dir = 'pngs'
    if not os.path.exists(png_save_dir):
        os.mkdir(png_save_dir)
    # 平滑矩阵的基
    smooth_kernel_vector = (0.1, 0.4, 1.0, 0.4, 0.1)

    @classmethod
    def weibull(cls, x: np.ndarray, lamda: float, k: float) -> List:
        '''
        weibull分布概率密度函数

        :param x: np.ndarray,
            输入序列
        :param lamda: float,
            weibull分布的尺度参数
        :param k: float,
            weibull分布的形状参数
        :return: List,
            基于尺度参数和形状参数的输入序列x处的概率密度序列
        '''
        return list(k/lamda * (x / lamda)**(k-1) * np.exp(-(x/lamda)**k))

    def __init__(self, features_dataframe: pd.DataFrame, target_dataframe: pd.DataFrame, target_freq_min: str,
                 infected_range: int = 36, criterion: str = 'SmoothL1Loss', optimizer_name: str = 'Adam',
                 learning_rate: float = 0.01, weight_decay: float = 0, epoch: int = 50, batches: int = 100,
                 is_kernel_reshape: bool = True, kernel_smooth_freq: int = 5, is_kernel_smooth: bool = True) -> None:
        """
        初始化函数

        :param features_dataframe: pd.DataFrame,
            特征数据框，形状为SeqLen X FeatureNum，索引为时刻
        :param target_dataframe: pd.DataFrame,
            目标数据框，形状为SeqLen X 1，索引为时刻
        :param target_freq_min: str,
            数据的目标采样频率
        :param infected_range: int, optional=36,
            错误数据的感染长度
        :param criterion: str, optional='SmoothL1Loss',
            损失函数的类型，可选值包括"SmoothL1Loss", "L1Loss", "MSELoss"
        :param optimizer_name: str, optional='Adam',
            优化器的类型
        :param learning_rate: float, optional=0.05,
            神经网络参数学习速率
        :param weight_decay: float, optional=0.0,
            神经网络参数惩罚系数
        :param epoch: int, optional=50,
            神经网络的学习回合数
        :param batches: int, optional=100,
            神经网络每轮学习所使用的样本点个数
        :param is_kernel_reshape: bool, optional=True,
            是否对神经网络卷积核的权重进行塑形
        :param kernel_smooth_freq: int, optional=5
            对一维卷积核序列进行平滑的（逆向）频率，即每进行kernel_smooth_freq次目标拟合之后，进行一次卷积核平滑操作
        :param is_kernel_smooth: bool, optional=True
            是否要求权重序列具有一定的平滑性
        """
        self.features_dataframe = features_dataframe
        self.target_dataframe = target_dataframe
        self.target_freq_min = target_freq_min
        self.infected_range = infected_range
        self.criterion = criterion
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.batches = batches
        self.is_kernel_reshape = is_kernel_reshape
        self.kernel_smooth_freq = 1 / kernel_smooth_freq
        self.is_kernel_smooth = is_kernel_smooth
        # 神经网络模型
        if is_kernel_smooth:
            self.cnn_model = CNN_V1(features_dataframe.shape[1], step=self.infected_range, output_size=1,
                                    smooth_kernel_vector=self.smooth_kernel_vector).cuda()
        else:
            self.cnn_model = CNN_V1(features_dataframe.shape[1], step=self.infected_range, output_size=1,
                                    smooth_kernel_vector=(0., 0., 1., 0., 0.)).cuda()
        # 最优的形状参数，尺度参数，无响应参数
        self.best_kernel_params = [None] * features_dataframe.shape[1]
        # 最优的权重序列
        self.best_weight_seq = [None] * features_dataframe.shape[1]
        # 生成一维卷积核目标形状
        self.kernel_weight_list, self.kernel_weight_param_list, self.weight_penal_coef = self.init_weight_seq_shapes()

    def init_weight_seq_shapes(self) -> Tuple[List, List, List]:
        """
        该方法用于生成一组潜在的权重序列集合

        :return: Tuple[List, List, List],
            返回三个不同的列表，分别是权重序列集合，权重序列构造参数集合，权重序列惩罚系数集合
        """
        # 权重序列集合
        weight_list = []
        # 权重序列构造参数集合
        params_record = []
        # 权重序列惩罚系数集合
        weight_penal_coef = []
        # 首先添加一个全零的权重序列
        weight_list.append([0] * self.infected_range)
        params_record.append((-1, -1))
        weight_penal_coef.append(0.85)
        # 然后根据不同的Weibull分布参数生成一个权重序列集合
        xs = np.linspace(0.01, 5, self.infected_range)
        for k in [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
            for lamda in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
                for no_res_stage in range(0, self.infected_range // 3, 2):
                    weibull_weight = [0] * no_res_stage + self.weibull(xs, lamda, k)
                    weight_list.append(weibull_weight[:self.infected_range][::-1])
                    params_record.append((k, lamda, no_res_stage))
                    weight_penal_coef.append(0.9 if np.mean(weight_list[-1][0:3]) < 0.005 else 1.0)
        return weight_list, params_record, weight_penal_coef

    def main(self):
        # 对数据进行预处理
        print('[Info]: 开始对数据进行预处理')
        united_feature_array, united_target_array, _, _ = \
            self.preprocess(self.features_dataframe, self.target_dataframe)

        print('[Info]: 开始进行模型训练')
        # 准备数据
        cuda_x = torch.from_numpy(united_feature_array).cuda()
        cuda_y = torch.from_numpy(united_target_array).cuda()
        # 定义评价指标
        if self.criterion == 'SmoothL1Loss':
            criterion = torch.nn.SmoothL1Loss()
        elif self.criterion == 'L1Loss':
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.MSELoss()
        # 定义优化函数
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        # 计算批数量
        batch_num = cuda_x.shape[0] // self.batches
        for e_index in tqdm.tqdm(range(self.epoch)):
            loss_list = []
            for i_index in range(batch_num):
                sub_cuda_x = cuda_x[i_index * self.batches: (i_index+1) * self.batches]
                sub_cuda_y = cuda_y[i_index * self.batches: (i_index+1) * self.batches]

                if np.random.rand() < self.kernel_smooth_freq:
                    # 进行卷积核平滑操作
                    if self.is_kernel_reshape and e_index > self.epochs_to_start_smooth:
                        self.smooth_kernel(sub_cuda_x, sub_cuda_y)

                pre_sub_cuda_y = self.cnn_model(sub_cuda_x)
                sub_loss = criterion(pre_sub_cuda_y, sub_cuda_y)
                loss_list.append(sub_loss.item())
                optimizer.zero_grad()
                sub_loss.backward()
                optimizer.step()

            if (e_index+1) % 10 == 0 or e_index == self.epoch-1:
                print(f'Epoch[{e_index+1}]: 拟合误差为：{np.mean(loss_list): .4f}')

                # 降低学习率
                if self.optimizer_name == 'Adam':
                    self.learning_rate = max(self.learning_rate / self.learning_rate_decay_ratio,
                                             self.learning_rate_down_bound)
                    optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate,
                                                 weight_decay=self.weight_decay)

                pre_y = pre_sub_cuda_y.cpu().data.numpy().ravel()
                act_y = sub_cuda_y.cpu().data.numpy().ravel()
                self.plot_evaluate_png(pre_y, act_y, e_index)

    def plot_evaluate_png(self, pre_y: np.ndarray, act_y: np.ndarray, e_index: int):
        figure = plt.figure(figsize=(12, 8))
        ax = figure.add_subplot(111)
        plt.plot(act_y, ls='--', color='k', alpha=0.6, lw=1.7)
        plt.plot(pre_y, ls='-', color='r', alpha=0.8, lw=1.0)
        png_name = os.path.join(self.png_save_dir, 'predict_%d.png' % e_index)
        plt.savefig(png_name, dpi=300)
        plt.close(figure)

        cnn_kernels = []
        for cnn_kernel in self.cnn_model.smoothed_conv_kernel_list:
            ws = cnn_kernel.detach().cpu().numpy().ravel()
            cnn_kernels.append(ws)
        cnn_kernels = np.array(cnn_kernels)
        kernel_max = np.max(cnn_kernels) + 0.2
        kernel_min = np.min(cnn_kernels) - 0.2

        cnn_index = 0
        for ws in cnn_kernels:
            figure = plt.figure(figsize=(12, 8))
            ax = figure.add_subplot(111)
            plt.plot(ws, lw=1.5, color='r', alpha=0.8)
            plt.plot([0, len(ws)], [0, 0], color='k', alpha=0.5, ls='--')
            try:
                ys = self.best_weight_seq[cnn_index]
                plt.plot(ys, color='navy', lw=2, alpha=0.6)
            except Exception as e:
                pass
            plt.ylim(kernel_min, kernel_max)
            png_name = os.path.join(self.png_save_dir, 'weight_%d_%d' % (e_index, cnn_index))
            plt.savefig(png_name, dpi=300)
            cnn_index += 1
            plt.close(figure)

    def smooth_kernel(self, cuda_x: torch.Tensor, cuda_b: torch.Tensor) -> None:
        kernel_weight_list = self.kernel_weight_list
        kernel_weight_penal_list = self.weight_penal_coef

        self.cnn_model(cuda_x)

        def smooth_item_kernel(item_smoothed_kernel: torch.Tensor, k: int) -> Tuple[int, float, np.ndarray]:
            optimizer = torch.optim.SGD(self.cnn_model.up_conv_list[k].parameters(), lr=0.01)
            item_smoothed_kernel_array = item_smoothed_kernel.detach().cpu().numpy().ravel()
            # 目前权重序列的平均值
            weight_mean = np.mean(item_smoothed_kernel_array)
            # 确定（调整后）线性相关性最高的权重序列形状序列
            max_score = 0
            max_k = 0
            max_seq = self.kernel_weight_list[max_k]
            for k, kernel_weight in enumerate(kernel_weight_list):
                if np.random.rand() < 0.5:
                    continue
                if k > 0:
                    score = np.corrcoef(item_smoothed_kernel_array, np.array(kernel_weight))[0, 1]
                else:
                    score = 0.2
                score /= kernel_weight_penal_list[k]
                if abs(score) > abs(max_score):
                    max_score = score
                    max_seq = kernel_weight
                    max_k = k
            # 调整权重序列向目标形状序列进行靠近
            max_seq = np.array(max_seq)
            if np.mean(max_seq) == 0:
                pass
            else:
                max_seq = (max_seq / np.mean(max_seq) * abs(weight_mean)) * (max_score / abs(max_score))
            target_item_kernel_weight = torch.from_numpy(np.array([max_seq], dtype=np.float32)).cuda()
            loss = criterion(item_smoothed_kernel, target_item_kernel_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return max_k, max_score, max_seq

        criterion = torch.nn.L1Loss()
        for k, item_smoothed_kernel in enumerate(self.cnn_model.smoothed_conv_kernel_list):
            best_k, best_score, best_seq = smooth_item_kernel(item_smoothed_kernel, k)
            self.best_kernel_params[k] = (self.kernel_weight_param_list[best_k], best_score)
            self.best_weight_seq[k] = best_seq

    def preprocess(self, features_dataframe: pd.DataFrame,
                   target_dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对数据列进行预处理，清洗数据，提高数据质量。相关的操作包括：
            1. 异常值检测
            2. 空值检测
            3. 异常序列剪除
            4. 格式转换

        :param features_dataframe: pd.DataFrame,
            特征数据框
        :param target_dataframe: pd.DataFrame,
            目标数据框
        :return:
        """
        adj_features_dataframe = features_dataframe.resample('%s' % self.target_freq_min).mean()
        adj_target_dataframe = target_dataframe.resample('%s' % self.target_freq_min).mean()
        abnormal_dataframe = pd.DataFrame(index=adj_features_dataframe.index)

        for column in features_dataframe.columns:
            # 首先剔除序列中的异常值
            feature_series = features_dataframe[column]
            feature_series = self.abnormal_value_detect(feature_series, 0.97)
            # 检测并填充序列中的空值
            feature_series, feature_abn_series = self.null_position_fill(feature_series, self.target_freq_min)
            adj_features_dataframe[column] = feature_series
            abnormal_dataframe[column] = feature_abn_series

        for column in target_dataframe.columns:
            target_series = target_dataframe[column]
            target_series = self.abnormal_value_detect(target_series, 0.95)
            # 检测并填充序列中的空值
            target_series, target_abn_series = self.null_position_fill(target_series, self.target_freq_min)
            adj_target_dataframe[column] = target_series
            abnormal_dataframe[column] = target_abn_series

        features_dataframe, target_dataframe, abnormal_dataframe = \
            adj_features_dataframe, adj_target_dataframe, abnormal_dataframe

        # 对无效索引进行计算
        infected_index = [True] * self.infected_range
        check_values = abnormal_dataframe.values
        for i in range(self.infected_range, len(check_values)):
            # 如果从当前行开始回溯指定长度，如果没有严重缺失值则采用该数据点
            if sum(sum(check_values[i - self.infected_range + 1:i + 1])) > 0:
                infected_index.append(True)
            else:
                infected_index.append(False)
        infected_index = np.array(infected_index)

        # ++++++++++将历史数据转换为独立数据块++++++++++
        features_array = features_dataframe.fillna(method='ffill', limit=5).values
        target_array = target_dataframe.fillna(method='ffill', limit=5).values
        united_feature_array = []
        united_target_array = []
        traj_index = []
        step_index = []
        traj_id = 0
        step_id = 0
        for k in range(self.infected_range, len(features_array)):
            if infected_index[k]:
                if step_id > 0:
                    traj_id += 1
                    step_id = 0
            else:
                united_feature_array.append(features_array[k - self.infected_range:k])
                united_target_array.append(target_array[k])
                traj_index.append([traj_id])
                step_index.append([step_id])
                step_id += 1
        # ++++++++++将历史数据转换为独立数据块++++++++++
        return np.array(united_feature_array, dtype=np.float32), np.array(united_target_array, dtype=np.float32), \
               np.array(traj_index), np.array(step_index)

    def abnormal_value_detect(self, seq: pd.Series, threshold: float = 0.98) -> pd.Series:
        """
        剔除序列数据中的异常值。
        所谓的异常值是在序列中极度上升或者下降的数据点，其上升或者下降的速度显著高于正常范围。

        :param seq: pd.Series,
            待处理的一维序列
        :param threshold: float, optional=0.99,
            判断异常值的分位点
        :return: pd.Series,
            修正了异常值之后的序列
        """
        try:
            assert len(seq.values.shape) == 1
        except AssertionError as e:
            print('[Error]: 异常值处理需要单特征列进行，不可一次输入多个特征列')
            raise(e)
        # 该序列的名称
        seq_name = seq.name
        print('[Info]: 开始对*%s*进行异常值处理' % seq_name)
        seq_values = seq.values
        seq_index = seq.index
        # 提取序列中的变换速度序列
        new_seq_values = [seq_values[0]]
        new_seq_index = [seq_index[0]]
        speed = []
        for k in range(1, len(seq_values)):
            cur_value = seq_values[k]
            cur_index = seq_index[k]
            value_diff = abs(cur_value - new_seq_values[-1])
            index_diff = cur_index - new_seq_index[-1]
            index_diff_seconds = index_diff.days * 24 * 3600 + index_diff.seconds
            if index_diff_seconds > 0 and value_diff > 0:
                speed.append(value_diff / index_diff_seconds)
                new_seq_values.append(cur_value)
                new_seq_index.append(cur_index)
        # 计算出变换速度的合理上限
        if len(speed) < 10:
            print('[Info]: 该序列的长度过短，不需要进行异常值处理')
            return seq
        threshold_speed = np.quantile(speed, threshold) * 5
        print('[Info]: 判定异常值的界限为：%.4f' % threshold_speed)
        # 将变换速度大于上限的样本点丢弃
        new_seq_values = [seq_values[0]]
        new_seq_index = [seq_index[0]]
        for k in range(1, len(seq_values)):
            cur_value = seq_values[k]
            cur_index = seq_index[k]
            value_diff = abs(cur_value - new_seq_values[-1])
            index_diff = cur_index - new_seq_index[-1]
            index_diff_seconds = index_diff.days * 24 * 3600 + index_diff.seconds
            if index_diff_seconds > 0:
                item_speed = value_diff / index_diff_seconds
                if item_speed <= threshold_speed:
                    new_seq_values.append(cur_value)
                    new_seq_index.append(cur_index)
                else:
                    pass
            else:
                print('[Warning]: 时间索引重复，丢弃该值')
        abnormal_value_count = len(seq_values) - len(new_seq_values)
        abnormal_value_ratio = abnormal_value_count / len(seq_values)
        print(f'[Info]: 共识别出{abnormal_value_count}个异常值，占比为：{abnormal_value_ratio*100: .4f}%')
        return pd.Series(new_seq_values, index=new_seq_index, name=seq_name)

    def null_position_fill(self, seq: pd.Series, windows: str, fill_type: str = 'interpolate',
                           max_fill_length: int = 2) -> Tuple[pd.Series, pd.Series]:
        """
        对特定频率下的序列进行空值的检测和填充，连续填充有最大长度限制

        :param seq: pd.Series,
            待处理的序列
        :param windows: str,
            重采样时钟长度
        :param fill_type: str, optional='interpoloate',
            空值填充的类型，可以使用'interpolate'，'ffill'
        :param max_fill_length: int, optional=5
            最大连续填充的次数
        :return:
        """
        if fill_type not in ['interpolate', 'ffill']:
            raise TypeError('当前设置的填充方法参数目前还不支持，可选的参数包括"interpolate,ffill"')
        seq_name = seq.name
        print(f'[Info]: 开始进行空值填充处理：{seq_name}')

        def window_avg(item_series):
            values = item_series.values
            if len(values) > 0:
                return np.mean(values)
            else:
                return np.nan

        # 对序列重采样到windows频率，并进行平均
        windowed_seq = seq.resample('%s' % windows).apply(window_avg)
        # 对序列中的空值量进行计数，并记录传染索引
        null_count = []
        init_count = 0
        infect_index = []
        for i in windowed_seq.values:
            if np.isnan(float(i)):
                init_count += 1
                if init_count > max_fill_length:
                    infect_index.append(True)
                else:
                    infect_index.append(False)
            else:
                infect_index.append(False)
                if init_count != 0:
                    null_count.append(init_count)
                    init_count = 0
        null_count = np.array(null_count)
        null_ratio = sum(null_count) / len(windowed_seq)
        if len(null_count > 0) > 0:
            severe_null_ratio = sum(null_count > max_fill_length) / len(null_count > 0)
        else:
            severe_null_ratio = 0.0
        severe_null_count = sum(null_count > max_fill_length)
        print(f'[Info]: 当前序列共有 {null_ratio*100: .2f}% 的空值，'
              f'其中严重的连续空值序列占 {severe_null_ratio*100: .2f}% ，共有 {severe_null_count} 段')
        if fill_type == 'interpolate':
            new_seq = windowed_seq.resample('%s' % windows).interpolate()
        elif fill_type == 'ffill':
            new_seq = windowed_seq.fillna(method='ffill')
        else:
            new_seq = windowed_seq.resample('%s' % windows).interpolate()
        infect_index = pd.Series(infect_index, index=new_seq.index)
        return new_seq, infect_index

    def transform(self, features_dataframe: pd.DataFrame, target_dataframe: pd.DataFrame) -> np.ndarray:
        united_feature_array, united_target_array, traj_index, step_index = \
            self.preprocess(features_dataframe, target_dataframe)
        init_feature_list = []
        for feature in united_feature_array:
            feature = feature.T
            init_feature = []
            for s_f, kernel in zip(feature, self.best_weight_seq):
                init_feature.append(sum(s_f * kernel))
            init_feature_list.append(init_feature)
        init_feature_array = np.array(init_feature_list)
        concat_array = np.hstack([traj_index, step_index, init_feature_array, united_target_array])
        return concat_array

    def predict(self, features_dataframe: pd.DataFrame, target_dataframe: pd.DataFrame = pd.DataFrame()) -> \
            Tuple[np.ndarray,np.ndarray]:
        # 为了与precess接口适配，所以虚拟出一列目标列
        if len(target_dataframe) == 0:
            target_dataframe = features_dataframe.iloc[:, 0:1]
        united_feature_array, united_target_array, _, _ = self.preprocess(features_dataframe, target_dataframe)
        cuda_x = torch.from_numpy(united_feature_array).cuda()
        pred_cuda_y = self.cnn_model(cuda_x)
        pred_target_array = pred_cuda_y.data.cpu().numpy().ravel()
        actual_target_array = united_target_array.ravel()
        return pred_target_array, actual_target_array

    def evaluate(self, features_dataframe: pd.DataFrame, target_dataframe: pd.DataFrame,
                 figure_name: str) -> None:
        pred_target_array, actual_target_array = self.predict(features_dataframe, target_dataframe)
        mae = np.mean(np.abs(pred_target_array - actual_target_array))
        figure = plt.figure(figsize=(14, 8))
        ax = figure.add_subplot(111)
        plt.plot(actual_target_array, color='k', alpha=0.7, ls='--', lw=1.5, label='Actual Trail')
        plt.plot(pred_target_array, color='r', alpha=0.9, lw=1.2, label='Predict Trail')
        plt.legend(loc='best')
        plt.title(f'MAE:{mae: .4f}', fontsize=16)
        plt.xlabel('Step', fontsize=15)
        plt.ylabel('Value', fontsize=15)
        plt.savefig(figure_name, dpi=300)

    def generate_virtue_data(self, seq_len: int) -> pd.DataFrame:
        start_clock = '2022-04-01 00:00:00'
        clock_index = pd.date_range(start=start_clock, periods=seq_len, freq='%dMin' % self.target_freq_min)

        # 奥恩斯坦-乌伦贝克随机过程
        ou_noise_generator = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=0.3, theta=0.2)
        xs_1 = np.array([ou_noise_generator()[0] for _ in range(seq_len)])

        # 奥恩斯坦-乌伦贝克随机过程
        ou_noise_generator = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=0.7, theta=0.4)
        xs_2 = np.array([ou_noise_generator()[0] for _ in range(seq_len)])

        # 奥恩斯坦-乌伦贝克随机过程
        ou_noise_generator = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=0.5, theta=0.3)
        xs_3 = np.array([ou_noise_generator()[0] for _ in range(seq_len)])

        kernel_1 = self.kernel_weight_list[np.random.randint(0, len(self.kernel_weight_list))]
        kernel_2 = self.kernel_weight_list[np.random.randint(0, len(self.kernel_weight_list))]
        kernel_3 = self.kernel_weight_list[np.random.randint(0, len(self.kernel_weight_list))]

        outcome = []
        for k in range(seq_len):
            if k >= self.infected_range:
                o_1 = np.sum(xs_1[k - self.infected_range: k] * kernel_1)
                o_2 = np.sum(xs_2[k - self.infected_range: k] * kernel_2)
                o_3 = np.sum(xs_3[k - self.infected_range: k] * kernel_3)
                o = o_1 + o_2 + o_3
            else:
                o = np.nan
            outcome.append(o)
        outcome = np.array(outcome)

        ou_noise_generator = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=0.7, theta=0.4)
        xs_1 = xs_1 + np.array([ou_noise_generator()[0] for _ in range(seq_len)]) \
               * np.array([1 if np.random.rand() > 0.97 else 0 for _ in range(seq_len)])
        xs_2 = xs_2 + np.array([ou_noise_generator()[0] for _ in range(seq_len)]) \
               * np.array([1 if np.random.rand() > 0.96 else 0 for _ in range(seq_len)])
        xs_3 = xs_3 + np.array([ou_noise_generator()[0] for _ in range(seq_len)]) \
               * np.array([1 if np.random.rand() > 0.95 else 0 for _ in range(seq_len)])
        outcome = outcome + np.array([ou_noise_generator()[0] for _ in range(seq_len)]) \
               * np.array([1 if np.random.rand() > 0.9 else 0 for _ in range(seq_len)]) * 6

        data = np.vstack([xs_1, xs_2, xs_3, outcome]).T
        dataframe = pd.DataFrame(data, index=clock_index, columns=['F_1', 'F_2', 'F_3', 'Output']).dropna()
        return dataframe


if __name__ == '__main__':
    import copy
    '''
    # 华润电厂数据
    example_data = pd.read_csv('example_data/data.csv', encoding='gbk')
    columns = '时刻，发电机功率，4A循环水泵变频器转速，4A送风机入口空气温度，一级工业供热流量，二级工业供热流量，4A循环水泵电流，4B循环水泵电流，循环水泵出口母管压力，4A凝汽器循环冷却水进口温度1，4A凝汽器循环冷却水出口温度，4B凝汽器冷却水出口温度，冷却塔水位，汽机低压缸排气温度-1，汽机低压缸排气温度-2，凝汽器真空度，热井水位，轴加出口凝水流量，主蒸汽流量，凝水温度，A段母线AB相间电压，B段母线AB相间电压，评价指标，主蒸汽母管温度，主蒸汽压力，高再出口温度，过热总减温水流量，再热总减温水流量，中压缸一号阀开度，中压缸二号阀开度'
    columns = columns.split('，')
    example_data.columns = columns
    feature_columns = copy.copy(columns)
    feature_columns.remove('评价指标')
    feature_columns.remove('时刻')
    feature_columns.remove('凝汽器真空度')
    target_column = ['凝汽器真空度']
    date_index = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in example_data['时刻'].values]
    example_data.index = date_index

    sample_num = 10000
    feature_dataframe = example_data[feature_columns].iloc[:sample_num]
    target_dataframe = example_data[target_column].iloc[:sample_num]
    feature_dataframe_test = example_data[feature_columns].iloc[sample_num:sample_num+3000]
    target_dataframe_test = example_data[target_column].iloc[sample_num:sample_num+3000]
    to_transform_feature_dataframe = example_data[feature_columns]
    to_transform_target_dataframe = example_data[target_column]
    '''

    # 二次网供回水温度数据
    example_data = pd.read_excel('example_data_2/云杉镇供回水温度.xlsx')
    date_index = [datetime.datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S')
                  for i in example_data['create_time']]
    example_data = example_data[['BH1_C2_GS_T', 'BH1_C2_HS_T']]
    example_data.index = date_index

    start_index = 10000
    end_index = 35000
    feature_dataframe = example_data[['BH1_C2_GS_T']].iloc[start_index: end_index]
    target_dataframe = example_data[['BH1_C2_HS_T']].iloc[start_index: end_index]
    feature_dataframe_test = example_data[['BH1_C2_GS_T']].iloc[end_index: end_index + 20000]
    target_dataframe_test = example_data[['BH1_C2_HS_T']].iloc[end_index: end_index + 20000]

    '''
    # 模拟数据
    example_data = pd.read_excel('example_data_2/virtue_dataframe.xlsx', index_col=0)
    start_index = 8000; end_index = 10000
    feature_columns = ['F_1', 'F_2', 'F_3']
    target_columns = ['Output']
    feature_dataframe = example_data[feature_columns].iloc[start_index: end_index]
    target_dataframe = example_data[target_columns].iloc[start_index: end_index]
    feature_dataframe_test = example_data[feature_columns].iloc[end_index: ]
    target_dataframe_test = example_data[target_columns].iloc[end_index: ]
    '''
    '''
    # 一次风控制
    example_data = pd.read_csv('example_data_2/downSample_15s.csv', index_col=0)
    date_index = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in example_data.index]
    example_data.index = date_index
    example_data['t_diff'] = example_data['t'].diff(periods=1)
    example_data = example_data.dropna()
    start_index = 0; end_index = 50000
    feature_columns = ['coal', 'cold_open', 'hot_open']
    target_columns = ['exit_air_amount']
    feature_dataframe = example_data[feature_columns].iloc[start_index: end_index]
    target_dataframe = example_data[target_columns].iloc[start_index: end_index]
    feature_dataframe_test = example_data[feature_columns].iloc[end_index: end_index + 15000]
    target_dataframe_test = example_data[target_columns].iloc[end_index: end_index + 15000]
    '''

    features_scaler = StandardScaler()
    target_scaler = StandardScaler()
    features_scaler.fit(feature_dataframe)
    target_scaler.fit(target_dataframe)
    feature_dataframe_array = features_scaler.transform(feature_dataframe)
    target_dataframe_array = target_scaler.transform(target_dataframe)
    feature_dataframe = pd.DataFrame(feature_dataframe_array, columns=feature_dataframe.columns,
                                     index=feature_dataframe.index)
    target_dataframe = pd.DataFrame(target_dataframe_array, columns=target_dataframe.columns,
                                    index=target_dataframe.index)
    feature_dataframe_array = features_scaler.transform(feature_dataframe_test)
    target_dataframe_array = target_scaler.transform(target_dataframe_test)
    feature_dataframe_test = pd.DataFrame(feature_dataframe_array, columns=feature_dataframe_test.columns,
                                     index=feature_dataframe_test.index)
    target_dataframe_test = pd.DataFrame(target_dataframe_array, columns=target_dataframe_test.columns,
                                    index=target_dataframe_test.index)

    rl_estimator = RayleighEstimator(feature_dataframe, target_dataframe, '2Min', epoch=350, infected_range=120,
                                     weight_decay=0.001, is_kernel_reshape=True, learning_rate=0.02,
                                     kernel_smooth_freq=2, batches=100, criterion='L1Loss', is_kernel_smooth=False)

    rl_estimator.main()
    rl_estimator.evaluate(feature_dataframe_test, target_dataframe_test, '无平滑有形状约束.png')
    print('over')








