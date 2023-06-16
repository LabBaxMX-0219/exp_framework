import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import time
import numpy as np

from methods.TS2Vec.encoder import TSEncoder
from methods.TS2Vec.losses import hierarchical_contrastive_loss
from methods.TS2Vec.utils import take_per_row

from utils.early_stopping import EarlyStopping

from data.datasets import Dataset_Electricity, Dataset_classification, Dataset_ETT

class Exp_TS2Vec():
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger
        self.encoder, self.swa_encoder = self._build_model()
        
    def _build_model(self):
        configs = self.configs
        device = configs.device
        base_model = TSEncoder(self.configs).to(device=device)
        swa_model = torch.optim.swa_utils.AveragedModel(base_model)
        swa_model.update_parameters(base_model)
        return base_model, swa_model
        
    def _get_data(self, flag):
        """
        Args:
            flag: in ["train", "val"]
        """
        configs = self.configs
        dataset_name = configs.dataset_name
        task = configs.task
        scale = configs.scale if task == "forecasting" else False
        seq_len = configs.seq_len if task == "forecasting" else 0
        pred_len = configs.pred_len if task == "forecasting" else 0
        inverse = configs.inverse if task == "forecasting" else 0
        timeenc = configs.timeenc if task == "forecasting" else 0
        freq = configs.freq if task == "forecasting" else 0
        cols = configs.cols if task == "forecasting" else 0
        batch_size = configs.self_supervised_batch_size
        data_folder_path = configs.data_folder_path
        
        data_dict = {
            'ETT_h1': Dataset_ETT,
            'ETT_h2': Dataset_ETT,
            'ETT_m1': Dataset_ETT,
            # 'ETT_m2': Dataset_ETT,
            'Electricity': Dataset_Electricity,
            'HAR': Dataset_classification,
            'Epilepsy': Dataset_classification,
            'sleepEDF': Dataset_classification,
            'PAMAP2': None,
        }
        Data_cls = data_dict[dataset_name]
        if task == "forecasting":
            data_set = Data_cls(
                data_folder_path=data_folder_path,
                scale=scale,
                flag=flag,
                seq_len=seq_len,
                inverse=inverse,
                timeenc=timeenc,
                freq=freq,
                choosed_variate_names=cols,
                dataset_name=dataset_name
            )
        if task == "classification":
            data_set = Data_cls(
                data_folder_path=data_folder_path,
                flag=flag,
                x_only=True
            )
        actual_batch_size = min(batch_size, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=actual_batch_size,  # ETT_h1只有4个样本...
            shuffle=True,
            num_workers=min(4, actual_batch_size),
            drop_last=True
        )
        return data_set, data_loader
        
    def _select_optimizer(self):
        lr = self.configs.self_supervised_lr
        model_optim = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
        return model_optim
    
    def _select_criterion(self):
        criterion = hierarchical_contrastive_loss
        return criterion

    def _get_augmentations(self, x):
        configs = self.configs
        # max_train_length = configs.max_train_length
        temporal_unit = configs.temporal_unit
        
        # if max_train_length is not None and x.size(1) > max_train_length:
        #     # print("before batch cut:", x.shape)
        #     window_offset = np.random.randint(x.size(1) - max_train_length + 1)
        #     x = x[:, window_offset : window_offset + max_train_length]
        #     # print("after batch cut:", x.shape)
        
        ts_len = x.size(1)  # 每个样本需要被进一步分割的数量
        crop_len = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_len+1)
        crop_left = np.random.randint(ts_len - crop_len + 1)
        crop_right = crop_left + crop_len
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_len + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_len - crop_eright + 1, size=x.size(0))
        
        aug1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        aug2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        
        return aug1, aug2, crop_len
        
        
    def _process_one_batch(self, batch_x1, batch_x2, crop_len):
        device = self.configs.device
        batch_x1 = batch_x1.float().to(device)
        batch_x2 = batch_x2.float().to(device)
        
        with torch.cuda.amp.autocast():  # 混合精度，降低显存占用，提高推理速度
            outputs1 = self.encoder(batch_x1)
            outputs2 = self.encoder(batch_x2)
        return outputs1[:, -crop_len:], outputs2[:,:crop_len]
    
    def valid(self, valid_loader, criterion):
        configs = self.configs
        temporal_unit = configs.temporal_unit
        self.encoder.eval()
        total_loss = []
        with torch.no_grad():
            for i, batch_x in enumerate(valid_loader):
                batch_x1, batch_x2, crop_len = self._get_augmentations(batch_x)
                
                out1, out2 = self._process_one_batch(batch_x1, batch_x2, crop_len)
                
                loss = criterion(
                    out1,
                    out2,
                    temporal_unit=temporal_unit
                )
                total_loss.append(loss.item())
        valid_avg_loss = np.average(total_loss)
        self.encoder.train()
        return valid_avg_loss
        
    def train(self):
        configs = self.configs
        logger = self.logger
        patience = configs.early_stop_patience
        train_epochs = configs.self_supervised_train_epochs
        valid_epochs = configs.supervised_valid_epochs
        use_amp = configs.use_amp
        temporal_unit = configs.temporal_unit
        model_save_path = configs.model_save_dir
        dataset_name = configs.dataset_name
        
        train_data, train_loader = self._get_data(flag='train')
        if 'ETT_h' in dataset_name:
            '''ETT_hx 数据集数据很少，使用训练集loss进行early_stopping'''
            logger.debug("Using train loss for valid_early_stopping..\n")
            vali_data, vali_loader = self._get_data(flag = 'train')
        else:
            vali_data, vali_loader = self._get_data(flag = 'val')
        time_now = time.time()
        
        early_stopping = EarlyStopping(logger=self.logger, patience=patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        logger.debug("Training started ....\n")
        
        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []
            
            self.encoder.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x1, batch_x2, crop_len = self._get_augmentations(batch_x)
                
                out1, out2 = self._process_one_batch(batch_x1, batch_x2, crop_len)
                print(torch.isnan(out1).int().sum())
                print(torch.isnan(out2).int().sum())
                
                loss = criterion(
                    out1,
                    out2,
                    temporal_unit=temporal_unit
                )
                train_loss.append(loss.item())
                print(f"batch {i} loss:", loss.item())
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                self.swa_encoder.update_parameters(self.encoder)
            logger.debug(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.4f} s")

            train_loss = np.average(train_loss)
            
            if (epoch+1) % valid_epochs == 0:
                valid_loss = self.valid(vali_loader, criterion)

                logger.debug(
                               f'Train Loss     : {train_loss:.4f}\n'
                               f'Valid Loss     : {valid_loss:.4f}')
                
                early_stopping(val_loss=valid_loss, model=self.swa_encoder, path=model_save_path)
                
                if early_stopping.early_stop:
                    logger.debug("Early stopping")
                    break
            else:
                logger.debug(f'Train Loss     : {train_loss:.4f}\n')
            
        logger.debug("\n################## Training is Done! #########################")
        
        # 将效果最好的model返回
        best_model_path = os.path.join(model_save_path, "checkpoint.pt")
        self.swa_encoder.load_state_dict(torch.load(best_model_path))
        return self.swa_encoder
    
