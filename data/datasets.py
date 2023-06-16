import os
import sys
sys.path.append("./")
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

class Dataloaders_forecasting_encoded_base():
    def __init__(self, data_folder_path, scale, inverse, device, choosed_variate_names=[], freq="h", timeenc=0, trained_model=None, padding=0, pred_len=24, batch_size=256, dataset_name="ETT_h1"):
        """
        Args:
            data_folder_path (str): 数据文件所在目录
            scale (boolean): 是否缩放
            inverse (boolean): 是否使用原数据
            size: (seq_len: 样本特征timestep长度,
                    start_token: 预测需要提供的上文信息timestep长度,
                    pred_len: 预测timestep长度)
            flag: 数据集类型，"train" | "val" | "test"
            choosed_variate_names (List[string]): 使用的数据列，默认为[]表示使用全部数据列
            freq: 编码datetime的精度，"y" | "m" | "w" | "d" | "b" | "h" | "t"
            timeenc: 0 or 1，0对应数值转换（0~60），1对应百分比转换（-0.5~0.5）
        """
        self.data_folder_path = data_folder_path
        self.data_path = os.path.join(self.data_folder_path, f"{dataset_name}.csv")
        self.scale = scale
        self.scaler = StandardScaler()
        self.inverse = inverse
        self.use_cols = choosed_variate_names
        self.timeenc = timeenc
        self.freq = freq
        self.trained_model = trained_model
        self.device = device
        self.padding = padding
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        
    
    def __read_data__(self):
        raise NotImplementedError()
    
    def __gen_encoded_datasets__(self):
        padding = self.padding  # context信息
        step = 1  # 滑窗步长
        pred_len = self.pred_len
        trained_model = self.trained_model
        date_cols = self.data_stamp.shape[-1]
        n_instances, total_timesteps, n_cols = self.processed_series.shape
        
        '''
        将完整序列编码得到表示：
        每次编码一个timestep，使用其之前padding个timestep的信息作为context，编码后丢弃context信息，由于前padding的数据的context不足，所以直接忽略这部分。
        由于要乱序训练模型，n_variates的预测相互独立，所以n_variates可以和samples合并在一维
        '''
        encode_start_index = padding
        encode_end_index = self.processed_series.shape[1]-pred_len
        reprs = []
        timestep_labels = []
        
        print("generate time series samples...\n")
        # [N, seq_len, cols]
        n_samples = total_timesteps-padding-pred_len
        samples = len([i for i in range(encode_start_index, encode_end_index, step)])
        # print(n_samples, samples)
        dataset = np.empty(shape=(n_samples, n_instances, padding+1, n_cols))
        for index in range(encode_start_index, encode_end_index, step):
            dataset[index-encode_start_index] = self.processed_series[:, index-padding:index+1]
            # print(dataset.shape)
        dataset = dataset.reshape(n_instances*n_samples, padding+1, n_cols)
        dataset = TensorDataset(torch.from_numpy(dataset).to(torch.float))  # [n_month*samples, padding+1, cols]
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, drop_last=False)
        
        '''
        生成表示，ts2vec中为每个timestep构建表示时，使用了前padding个timestep的数据作为context，在逐点生成表示后要将padding去掉，目的是使模型能够基于任意timestep的表示预测pred_len的数据
        '''
        print("time series batch encoding..\n")
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            trained_model.eval()
            for (batch_x,) in loader:  # 2048->3.4GB, 9min
                '''batch_x: [batch, padding+1, cols]'''
                batch_x = batch_x.float().to(self.device)
                encode_repr = trained_model(batch_x)  # [batch, padding+1, out_dim]
                reprs.append(encode_repr[:, padding:].detach().cpu())  # [batch, 1, out_dim]
        reprs = np.concatenate(reprs, axis=0).squeeze(1)  # [n_month*samples, _]
        
        '''
        为每个timestep的表示生成对应的labels，这里要预测的是数据，日期列直接忽略
        '''
        real_cols = self.processed_series.shape[-1]-self.data_stamp.shape[-1]
        timestep_labels = np.empty(shape=(n_samples, n_instances, pred_len, real_cols))
        for index in range(encode_start_index, encode_end_index, step):
            timestep_labels[index-encode_start_index] = self.processed_series[:, index+1:index+1+pred_len, self.data_stamp.shape[-1]:]
        timestep_labels = timestep_labels.reshape(n_instances*n_samples, pred_len*real_cols)
        
        '''按照比例生成样本
        '''
        len_reprs = reprs.shape[0]
        train_slice = slice(padding, int(0.6 * len_reprs)-pred_len)
        valid_slice = slice(int(0.6 * len_reprs), int(0.8 * len_reprs))
        test_slice = slice(int(0.8 * len_reprs), None)
        
        train_features = reprs[train_slice]
        train_labels = timestep_labels[train_slice]

        valid_features = reprs[valid_slice]
        valid_labels = timestep_labels[valid_slice]
        
        test_features = reprs[test_slice]
        test_labels = timestep_labels[test_slice]
        # print("=====")
        # print(reprs.shape, timestep_labels.shape)
        # print(train_features.shape, train_labels.shape)
        # print(valid_features.shape, valid_labels.shape)
        # print(test_features.shape, test_labels.shape)
        
        '''构建dataset'''
        self.train_dataset = TensorDataset(
                        torch.from_numpy(train_features).to(torch.float),
                        torch.from_numpy(train_labels).to(torch.float)
                        )
        self.valid_dataset = TensorDataset(
                        torch.from_numpy(valid_features).to(torch.float),
                        torch.from_numpy(valid_labels).to(torch.float)
                        )
        self.test_dataset = TensorDataset(
                        torch.from_numpy(test_features).to(torch.float),
                        torch.from_numpy(test_labels).to(torch.float)
                        )
        del self.trained_model, batch_x
        torch.cuda.empty_cache()
        return
        
    def gen_encoded_dataloaders(self):
        self.__read_data__()
        self.__gen_encoded_datasets__()
        train_loader = DataLoader(self.train_dataset, batch_size=min(self.batch_size, len(self.train_dataset)), shuffle=True, drop_last=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=min(self.batch_size, len(self.valid_dataset)), shuffle=True, drop_last=False)
        test_loader = DataLoader(self.test_dataset, batch_size=min(self.batch_size, len(self.test_dataset)), shuffle=False, drop_last=False)
        
        return train_loader, valid_loader, test_loader

class Dataset_ETT(Dataset):
    """ETT_xx 数据集
    """
    def __init__(self, data_folder_path, scale, inverse, flag, seq_len, choosed_variate_names=[], freq="h", timeenc=0, dataset_name="ETT_h1"):
        """
        Args:
            data_folder_path (str): 数据文件所在目录
            scale (boolean): 是否缩放
            inverse (boolean): 是否使用原数据
            seq_len: 样本特征timestep长度
            pred_len: 预测timestep长度
            flag: 数据集类型，"train" | "val" | "test"
            choosed_variate_names (List[string]): 使用的数据列，默认为[]表示使用全部数据列
            freq: 编码datetime的精度，"y" | "m" | "w" | "d" | "b" | "h" | "t"
            timeenc: 0 or 1，0对应数值转换（0~60），1对应百分比转换（-0.5~0.5）
        """
        self.data_folder_path = data_folder_path
        self.data_path = os.path.join(self.data_folder_path, f"{dataset_name}.csv")
        self.scale = scale
        self.scaler = StandardScaler()
        self.inverse = inverse
        self.use_cols = choosed_variate_names
        self.timeenc = timeenc
        self.freq = freq
        self.seq_len = seq_len
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path, parse_dates=True)
        
        # 使用选定的数据列 'MT_001'
        if len(self.use_cols) != 0:
            col_names = df_raw.columns[1:]  # 忽略date列
            for use_col_name in self.use_cols:
                assert use_col_name in col_names, f"{use_col_name} not in cols"
            cols_data = self.use_cols
        else:
            cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]  # [n_timesteps, n_months]
        
        # 根据类型确定数据的起止位置
        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        start_idxs = [0,                   # train
                      num_train,           # val
                      num_train+num_vali,  # test
                      ]
        end_idxs = [num_train,             # train
                    num_train+num_vali,    # val
                    len(df_raw)            # test
                    ]
        start_idx = start_idxs[self.set_type]
        end_idx = end_idxs[self.set_type]

        # 缩放
        if self.scale:
            scaler_train_data = df_data[start_idxs[0]:end_idxs[0]]
            self.scaler.fit(scaler_train_data.values)  # 使用训练集数据训练scaler
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data = np.expand_dims(data, 0)[:, start_idx:end_idx]  # [1, timesteps, n_variates]
        
        # 将datetime编码为对应的时间变量，date_cols
        # 比如H，m对应Hour和minute两维
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # [timesteps, 1]
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)  # [timesteps, date_cols]
        self.data_stamp = data_stamp[start_idx:end_idx]
        self.data_stamp = np.expand_dims(self.data_stamp, axis=0)  # [1, timesteps, date_cols]
        self.data_stamp = np.repeat(self.data_stamp, repeats=data.shape[0], axis=0)  # [1, timesteps, date_cols]
        
        # 构建X
        self.data_x = np.concatenate([self.data_stamp, data], axis=-1)  # [1, timesteps, n_variates+date_cols]
        
        '''自监督用数据，只有x'''
        # 将完整的timesteps进一步切分成seq_len的定长样本
        sections = self.data_x.shape[1] // self.seq_len
        self.data_x = self.data_x[:, :sections*self.seq_len]
        if sections > 1:
            data_x_sec = self.split_with_nan(self.data_x, sections, axis=1)
            self.data_x = np.concatenate(data_x_sec, axis=0)  # 将分割出来的样本都堆到dim0
        # print("data_x.shape:", self.data_x.shape)
        self.data_x = self.__clean_data__(self.data_x)
        return self.data_x  # [n_samples, seq_len, n_variates+date_cols]

    def pad_nan_to_target(self, array, target_length, axis=0, both_side=False):
        assert array.dtype in [np.float16, np.float32, np.float64]
        pad_size = target_length - array.shape[axis]
        if pad_size <= 0:
            return array
        npad = [(0, 0)] * array.ndim
        if both_side:
            npad[axis] = (pad_size // 2, pad_size - pad_size//2)
        else:
            npad[axis] = (0, pad_size)
        return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

    def split_with_nan(self, x, sections, axis=0):
        assert x.dtype in [np.float16, np.float32, np.float64]
        arrs = np.array_split(x, sections, axis=axis)
        target_length = arrs[0].shape[axis]
        for i in range(len(arrs)):
            arrs[i] = self.pad_nan_to_target(arrs[i], target_length, axis=axis)
        return arrs

    def _centerize_vary_length_series(self, x):
        '''使各个instance内n_features不全为nan的timestep居中
        [[nan, nan, [...]], ...] 
        -> 
        [[nan, [...], nan], ...]
        '''
        prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
        suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
        offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
        rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]  # 产生(dim1,1)数组，类似arange
        offset[offset < 0] += x.shape[1]
        column_indices = column_indices - offset[:, np.newaxis]
        return x[rows, column_indices]
    
    def __clean_data__(self, dataset):
        '''如果这批数据存在全nan的timestep，使各个instance内n_features不全为nan的timestep居中'''
        temporal_missing = np.isnan(dataset).all(axis=-1).any(axis=0)  # 判断是否存在n_features全为nan的dim
        if temporal_missing[0] or temporal_missing[-1]:
            dataset = self._centerize_vary_length_series(dataset)
        
        '''丢弃timestep全为nan的instance'''
        dataset = dataset[~np.isnan(dataset).all(axis=2).all(axis=1)]
        return dataset
    
    def __getitem__(self, index):
        return self.data_x[index]  # [seq_len, date_cols+1]
        
    def __len__(self):
        return self.data_x.shape[0]

class Dataloaders_EET_encoded(Dataloaders_forecasting_encoded_base):
    """EET_xx 数据集，提供编码后的输入
    """
    def __init__(self, data_folder_path, scale, inverse, device, choosed_variate_names=[], freq="h", timeenc=0, trained_model=None, padding=0, pred_len=24, batch_size=256, dataset_name="ETT_h1"):
        super().__init__(
            data_folder_path=data_folder_path,
            scale=scale,
            inverse=inverse,
            device=device,
            choosed_variate_names=choosed_variate_names,
            freq=freq,
            timeenc=timeenc,
            trained_model=trained_model,
            padding=padding,
            pred_len=pred_len,
            batch_size=batch_size,
            dataset_name=dataset_name
        )
        
    def __read_data__(self):
        print("loading raw time series...\n")
        df_raw = pd.read_csv(self.data_path, parse_dates=True)
        
        # 使用选定的数据列如 'OT'
        if len(self.use_cols) != 0:
            col_names = df_raw.columns[1:]  # 忽略date列
            for use_col_name in self.use_cols:
                assert use_col_name in col_names, f"{use_col_name} not in cols"
            cols_data = self.use_cols
        else:
            cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]  # [n_timesteps, n_variates]
        
        # 缩放
        if self.scale:
            self.scaler.fit(df_data.values)  # 使用所有数据训练scaler
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        data = np.expand_dims(data, 0)  # [1, n_timesteps, n_variates]
        
        # 将datetime编码为对应的时间变量，date_cols
        # 比如H，m对应Hour和minute两维
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # [n_timesteps, 1]
        self.data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)  # [n_timesteps, date_cols]
        self.data_stamp = np.expand_dims(self.data_stamp, axis=0)  # [1, n_timesteps, date_cols]
        self.data_stamp = np.repeat(self.data_stamp, repeats=data.shape[0], axis=0)  # [1, n_timesteps, date_cols]
        
        # 构建待编码的完整序列
        self.processed_series = np.concatenate([self.data_stamp, data], axis=-1)  # [1, n_timesteps, date_cols+n_variates]



class Dataset_Electricity(Dataset):
    """Electricity数据集
    """
    def __init__(self, data_folder_path, scale, inverse, flag, seq_len, choosed_variate_names=[], freq="h", timeenc=0, dataset_name="Electricity"):
        """
        Args:
            data_folder_path (str): 数据文件所在目录
            scale (boolean): 是否缩放
            inverse (boolean): 是否使用原数据
            seq_len: 样本特征timestep长度
            pred_len: 预测timestep长度
            flag: 数据集类型，"train" | "val" | "test"
            choosed_variate_names (List[string]): 使用的数据列，默认为[]表示使用全部数据列
            freq: 编码datetime的精度，"y" | "m" | "w" | "d" | "b" | "h" | "t"
            timeenc: 0 or 1，0对应数值转换（0~60），1对应百分比转换（-0.5~0.5）
        """
        self.data_folder_path = data_folder_path
        self.data_path = os.path.join(self.data_folder_path, f"{dataset_name}.csv")
        self.scale = scale
        self.scaler = StandardScaler()
        self.inverse = inverse
        self.use_cols = choosed_variate_names
        self.timeenc = timeenc
        self.freq = freq
        self.seq_len = seq_len
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

        
    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path, parse_dates=True)
        
        # 使用选定的数据列 'MT_001'
        if len(self.use_cols) != 0:
            col_names = df_raw.columns[1:]  # 忽略date列
            for use_col_name in self.use_cols:
                assert use_col_name in col_names, f"{use_col_name} not in cols"
            cols_data = self.use_cols
        else:
            cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]  # [n_timesteps, n_months]
        
        # 根据类型确定数据的起止位置
        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        start_idxs = [0,                              # train
                      num_train-self.seq_len,        # val
                      len(df_raw)-num_test-self.seq_len  # test
                      ]
        end_idxs = [num_train,            # train
                    num_train+num_vali,    # val
                    len(df_raw)     # test
                    ]
        start_idx = start_idxs[self.set_type]
        end_idx = end_idxs[self.set_type]
        
        # 缩放
        if self.scale:
            scaler_train_data = df_data[start_idxs[0]:end_idxs[0]]
            self.scaler.fit(scaler_train_data.values)  # 使用训练集数据训练scaler
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        data = np.expand_dims(data.T, -1)[:, start_idx:end_idx]  # [n_months, timesteps, 1]
        
        # 将datetime编码为对应的时间变量，date_cols
        # 比如H，m对应Hour和minute两维
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # [timesteps, 1]
        data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)  # [timesteps, date_cols]
        self.data_stamp = data_stamp[start_idx:end_idx]
        self.data_stamp = np.expand_dims(self.data_stamp, axis=0)  # [1, timesteps, date_cols]
        self.data_stamp = np.repeat(self.data_stamp, repeats=data.shape[0], axis=0)  # [n_months, timesteps, date_cols]
        
        # 构建X
        self.data_x = np.concatenate([self.data_stamp, data], axis=-1)  # [n_months, timesteps, date_cols+1]
        
        '''自监督用数据，只有x'''
        # 将完整的timesteps进一步切分成seq_len的定长样本
        sections = self.data_x.shape[1] // self.seq_len
        self.data_x = self.data_x[:, :sections*self.seq_len]
        if sections > 1:
            data_x_sec = self.split_with_nan(self.data_x, sections, axis=1)
            self.data_x = np.concatenate(data_x_sec, axis=0)  # 将分割出来的样本都堆到dim0
        self.data_x = self.__clean_data__(self.data_x)
        return self.data_x  # [n_instance, seq_len, date_col+1]

    def pad_nan_to_target(self, array, target_length, axis=0, both_side=False):
        assert array.dtype in [np.float16, np.float32, np.float64]
        pad_size = target_length - array.shape[axis]
        if pad_size <= 0:
            return array
        npad = [(0, 0)] * array.ndim
        if both_side:
            npad[axis] = (pad_size // 2, pad_size - pad_size//2)
        else:
            npad[axis] = (0, pad_size)
        return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

    def split_with_nan(self, x, sections, axis=0):
        assert x.dtype in [np.float16, np.float32, np.float64]
        arrs = np.array_split(x, sections, axis=axis)
        target_length = arrs[0].shape[axis]
        for i in range(len(arrs)):
            arrs[i] = self.pad_nan_to_target(arrs[i], target_length, axis=axis)
        return arrs

    def _centerize_vary_length_series(self, x):
        '''使各个instance内n_features不全为nan的timestep居中
        [[nan, nan, [...]], ...] 
        -> 
        [[nan, [...], nan], ...]
        '''
        prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
        suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
        offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
        rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]  # 产生(dim1,1)数组，类似arange
        offset[offset < 0] += x.shape[1]
        column_indices = column_indices - offset[:, np.newaxis]
        return x[rows, column_indices]
    
    def __clean_data__(self, dataset):
        '''如果这批数据存在全nan的timestep，使各个instance内n_features不全为nan的timestep居中'''
        temporal_missing = np.isnan(dataset).all(axis=-1).any(axis=0)  # 判断是否存在n_features全为nan的dim
        if temporal_missing[0] or temporal_missing[-1]:
            dataset = self._centerize_vary_length_series(dataset)
        
        '''丢弃timestep全为nan的instance'''
        dataset = dataset[~np.isnan(dataset).all(axis=2).all(axis=1)]
        return dataset
    
    def __getitem__(self, index):
        return self.data_x[index]  # [seq_len, date_cols+1]
        
    def __len__(self):
        return self.data_x.shape[0]

class Dataloaders_Electricity_encoded(Dataloaders_forecasting_encoded_base):
    """Electricity数据集，提供编码后的输入
    """
    def __init__(self, data_folder_path, scale, inverse, device, choosed_variate_names=[], freq="h", timeenc=0, trained_model=None, padding=0, pred_len=24, batch_size=256, dataset_name="ETT_h1"):
        super().__init__(
            data_folder_path=data_folder_path,
            scale=scale,
            inverse=inverse,
            device=device,
            choosed_variate_names=choosed_variate_names,
            freq=freq,
            timeenc=timeenc,
            trained_model=trained_model,
            padding=padding,
            pred_len=pred_len,
            batch_size=batch_size,
            dataset_name=dataset_name
        )

    def __read_data__(self):
        print("loading raw time series...\n")
        df_raw = pd.read_csv(self.data_path, parse_dates=True)
        
        # 使用选定的数据列如 'MT_001'
        if len(self.use_cols) != 0:
            col_names = df_raw.columns[1:]  # 忽略date列
            for use_col_name in self.use_cols:
                assert use_col_name in col_names, f"{use_col_name} not in cols"
            cols_data = self.use_cols
        else:
            cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]  # [n_timesteps, n_months]
        
        # 缩放
        if self.scale:
            self.scaler.fit(df_data.values)  # 使用所有数据训练scaler
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        data = np.expand_dims(data.T, -1)  # [n_months, timesteps, 1]
        
        # 将datetime编码为对应的时间变量，date_cols
        # 比如H，m对应Hour和minute两维
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # [timesteps, 1]
        self.data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)  # [timesteps, date_cols]
        self.data_stamp = np.expand_dims(self.data_stamp, axis=0)  # [1, timesteps, date_cols]
        self.data_stamp = np.repeat(self.data_stamp, repeats=data.shape[0], axis=0)  # [n_months, timesteps, date_cols]
        
        # 构建待编码的完整序列
        self.processed_series = np.concatenate([self.data_stamp, data], axis=-1)  # [n_months, timesteps, date_cols+1]

class old_Dataloaders_Electricity_encoded():
    """Electricity数据集，提供编码后的输入
    """
    def __init__(self, data_folder_path, scale, inverse, device, choosed_variate_names=[], freq="h", timeenc=0, trained_model=None, padding=0, pred_len=24, batch_size=256):
        """
        Args:
            data_folder_path (str): 数据文件所在目录
            scale (boolean): 是否缩放
            inverse (boolean): 是否使用原数据
            size: (seq_len: 样本特征timestep长度,
                    start_token: 预测需要提供的上文信息timestep长度,
                    pred_len: 预测timestep长度)
            flag: 数据集类型，"train" | "val" | "test"
            choosed_variate_names (List[string]): 使用的数据列，默认为[]表示使用全部数据列
            freq: 编码datetime的精度，"y" | "m" | "w" | "d" | "b" | "h" | "t"
            timeenc: 0 or 1，0对应数值转换（0~60），1对应百分比转换（-0.5~0.5）
        """
        self.data_folder_path = data_folder_path
        self.data_path = os.path.join(self.data_folder_path, "Electricity.csv")
        self.scale = scale
        self.scaler = StandardScaler()
        self.inverse = inverse
        self.use_cols = choosed_variate_names
        self.timeenc = timeenc
        self.freq = freq
        self.trained_model = trained_model
        self.device = device
        self.padding = padding
        self.pred_len = pred_len
        self.batch_size = batch_size

        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.__read_data__()
        self.__gen_encoded_datasets__()
        
        
    def __read_data__(self):
        print("loading raw time series...\n")
        df_raw = pd.read_csv(self.data_path, parse_dates=True)
        
        # 使用选定的数据列 'MT_001'
        if len(self.use_cols) != 0:
            col_names = df_raw.columns[1:]  # 忽略date列
            for use_col_name in self.use_cols:
                assert use_col_name in col_names, f"{use_col_name} not in cols"
            cols_data = self.use_cols
        else:
            cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]  # [n_timesteps, n_months]
        
        # 缩放
        if self.scale:
            self.scaler.fit(df_data.values)  # 使用所有数据训练scaler
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        data = np.expand_dims(data.T, -1)  # [n_months, timesteps, 1]
        
        # 将datetime编码为对应的时间变量，date_cols
        # 比如H，m对应Hour和minute两维
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)  # [timesteps, 1]
        self.data_stamp = time_features(
            df_stamp, timeenc=self.timeenc, freq=self.freq)  # [timesteps, date_cols]
        self.data_stamp = np.expand_dims(self.data_stamp, axis=0)  # [1, timesteps, date_cols]
        self.data_stamp = np.repeat(self.data_stamp, repeats=data.shape[0], axis=0)  # [n_months, timesteps, date_cols]
        
        # 构建待编码的完整序列
        self.processed_series = np.concatenate([self.data_stamp, data], axis=-1)  # [n_months, timesteps, date_cols+1]
        
    def __gen_encoded_datasets__(self):
        padding = self.padding  # context信息
        step = 1  # 滑窗步长
        pred_len = self.pred_len
        trained_model = self.trained_model
        date_cols = self.data_stamp.shape[-1]
        n_instances, total_timesteps, n_cols = self.processed_series.shape
        
        '''
        编码得到表示：
        每次编码一个timestep，使用其之前padding个timestep的信息作为context，编码后丢弃context信息
        '''
        encode_start_index = padding
        encode_end_index = self.processed_series.shape[1]-pred_len
        reprs = []
        timestep_labels = []
        
        print("generate time series samples...\n")
        # [N, seq_len, cols]
        n_samples = total_timesteps-padding-pred_len
        samples = len([i for i in range(encode_start_index, encode_end_index, step)])
        # print(n_samples, samples)
        dataset = np.empty(shape=(n_samples, n_instances, padding+1, n_cols))
        for index in range(encode_start_index, encode_end_index, step):
            dataset[index-encode_start_index] = self.processed_series[:, index-padding:index+1]
            # print(dataset.shape)
        dataset = dataset.reshape(n_instances*n_samples, padding+1, n_cols)
        dataset = TensorDataset(torch.from_numpy(dataset).to(torch.float))  # [n_month*samples, padding+1, cols]
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, drop_last=False)
        
        print("time series batch encoding..\n")
        with torch.no_grad():
            # with torch.cuda.amp.autocast():
            trained_model.eval()
            for (batch_x,) in loader:  # 2048->3.4GB, 9min
                '''batch_x: [batch, padding+1, cols]'''
                batch_x = batch_x.float().to(self.device)
                encode_repr = trained_model(batch_x)  # [batch, padding+1, out_dim]
                reprs.append(encode_repr[:, padding:].detach().cpu())  # [batch, 1, out_dim]
        reprs = np.concatenate(reprs, axis=0).squeeze(1)  # [n_month*samples, _]
        
        timestep_labels = np.empty(shape=(n_samples, n_instances, pred_len, 1))
        for index in range(encode_start_index, encode_end_index, step):
            timestep_labels[index-encode_start_index] = self.processed_series[:, index+1:index+1+pred_len, self.data_stamp.shape[-1]:]
        timestep_labels = timestep_labels.reshape(n_instances*n_samples, pred_len)
        
        '''生成样本
        ts2vec是分离出0.6*reprs作为训练集，去除开头padding无效部分，去除尾部pred_len，之后将timestep和n_instances合并 [321*int(26304*0.6-200-24), 320] = [4994118, 320]
        同理labels有[4994118, 24]
        
        因为已经对任意timestep进行了encode，所以可以用任意timestep表示去预测pred_len数据
        '''
        len_reprs = reprs.shape[0]
        train_slice = slice(padding, int(0.6 * len_reprs)-pred_len)
        valid_slice = slice(int(0.6 * len_reprs), int(0.8 * len_reprs))
        test_slice = slice(int(0.8 * len_reprs), None)
        
        train_features = reprs[train_slice]
        train_labels = timestep_labels[train_slice]

        valid_features = reprs[valid_slice]
        valid_labels = timestep_labels[valid_slice]
        
        test_features = reprs[test_slice]
        test_labels = timestep_labels[test_slice]
        # print("=====")
        # print(reprs.shape, timestep_labels.shape)
        # print(train_features.shape, train_labels.shape)
        # print(valid_features.shape, valid_labels.shape)
        # print(test_features.shape, test_labels.shape)
        
        '''构建dataset'''
        self.train_dataset = TensorDataset(
                        torch.from_numpy(train_features).to(torch.float),
                        torch.from_numpy(train_labels).to(torch.float)
                        )
        self.valid_dataset = TensorDataset(
                        torch.from_numpy(valid_features).to(torch.float),
                        torch.from_numpy(valid_labels).to(torch.float)
                        )
        self.test_dataset = TensorDataset(
                        torch.from_numpy(test_features).to(torch.float),
                        torch.from_numpy(test_labels).to(torch.float)
                        )
        del self.trained_model, batch_x
        torch.cuda.empty_cache()
        return
        
    def gen_encoded_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=min(self.batch_size, len(self.train_dataset)), shuffle=True, drop_last=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=min(self.batch_size, len(self.valid_dataset)), shuffle=True, drop_last=False)
        test_loader = DataLoader(self.test_dataset, batch_size=min(self.batch_size, len(self.test_dataset)), shuffle=False, drop_last=False)
        
        return train_loader, valid_loader, test_loader


"""
分类任务所用的数据集形式应为 data_folder/ train|val|test.pt

.pt中用samples和labels存放处理好的样本
"""

class Dataset_classification(Dataset):
    """分类数据集，HAR等
    """
    def __init__(self, data_folder_path, flag, x_only=True):
        """
        Args:
            data_folder_path (str): 数据文件所在目录
            flag: 数据集类型，"train" | "val" | "test"
        """
        super
        self.data_folder_path = data_folder_path
        self.data_path = os.path.join(self.data_folder_path, f"{flag}.pt")
        self.x_only = x_only
        self.__read_data__()
    
    def __read_data__(self):
        dataset = torch.load(self.data_path)
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        '''make sure timestep in second dim'''
        if X_train.size(1) < X_train.size(2):
            X_train = X_train.permute(0,2,1)
            # print("X_train:", X_train.shape)
            
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).to(torch.float)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.to(torch.float)
            self.y_data = y_train

        self.len = X_train.shape[0]
    
    def __getitem__(self, index):
        if self.x_only:
            return self.x_data[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DataLoaders_classification(Dataset):
    def __init__(self, data_folder_path, batch_size):
        self.train_ds = Dataset_classification(data_folder_path=data_folder_path, flag="train", x_only=False)
        self.valid_ds = Dataset_classification(data_folder_path=data_folder_path, flag="val", x_only=False)
        self.test_ds = Dataset_classification(data_folder_path=data_folder_path, flag="test", x_only=False)
        self.batch_size = batch_size
    
    def get_dataloader(self):
        batch_size = self.batch_size
        return DataLoader(
            self.train_ds,
            batch_size=min(batch_size,len(self.train_ds)),
            shuffle=True,
            num_workers=4,
            drop_last=True
        ),DataLoader(
            self.valid_ds,
            batch_size=min(batch_size,len(self.valid_ds)),
            shuffle=True,
            num_workers=4,
            drop_last=False
        ),DataLoader(
            self.test_ds,
            batch_size=min(batch_size,len(self.valid_ds)),
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

if __name__ == "__main__":
    data_path = "/home/HaotianF/Exp/Data/Electricity/Electricity.csv"
    scale = True
    inverse = False
    flag = "train"
    seq_len = 3000
    # ds = Dataset_Electricity(data_path=data_path, scale=scale, inverse=inverse, flag=flag, seq_len=seq_len,)  # choosed_variate_names=["MT_001", "MT_002"]
    # print(ds.__len__())
    padding = 200
    # Dataset_Electricity_encode(data_path=data_path, scale=scale, inverse=inverse, seq_len=300, padding=padding)
    
    data_folder_path = "/home/HaotianF/Exp/Data/ETT_h1"
    dataset_name = "ETT_h1"
    # ds = Dataset_ETT(data_folder_path=data_folder_path, scale=scale, inverse=inverse, seq_len=seq_len, flag="val", dataset_name=dataset_name)
    # print(ds.__len__())
    
    data_folder_path="/home/HaotianF/Exp/Data/sleepEDF"
    Dataset_classification(data_folder_path=data_folder_path, flag="train", x_only=False)
    
    