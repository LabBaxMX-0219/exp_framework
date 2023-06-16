import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import numpy as np

from methods.TS2Vec.encoder import TSEncoder
from methods.TS2Vec.utils import take_per_row

from utils.early_stopping import EarlyStopping

from data.datasets import Dataloaders_Electricity_encoded, Dataloaders_EET_encoded

class Exp_TS2Vec():
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger
        self.linear = self._load_model()
        self._get_dataloader()
        
    def _load_model(self):
        configs = self.configs
        device = configs.device
        logger = self.logger
        linear_in = configs.linear_input_dim
        linear_out = configs.linear_output_dim
        
        linear = nn.Linear(
            in_features=linear_in,
            out_features=linear_out,
            device=device
        )
        logger.debug(f"linear model mounted on device: {device}\n")
        
        return linear
        
    def _get_dataloader(self):
        configs = self.configs
        device = configs.device
        dataset_name = configs.dataset_name
        task = configs.task
        scale = configs.scale
        padding = configs.padding
        pred_len = configs.pred_len
        inverse = configs.inverse
        timeenc = configs.timeenc
        freq = configs.freq
        cols = configs.cols
        batch_size = configs.train_linear_batch_size
        data_folder_path = configs.data_folder_path
        
        
        # load pretrained_model
        load_from = configs.model_load_from
        linear_in = configs.linear_input_dim
        linear_out = configs.linear_output_dim
        
        base_model = TSEncoder(self.configs).to(device=device)
        swa_model = torch.optim.swa_utils.AveragedModel(base_model)
        swa_model.update_parameters(base_model)
        
        pretrained_dict = torch.load(os.path.join(load_from, "checkpoint.pt"), map_location=device)
        swa_model.load_state_dict(pretrained_dict)
        trained_model = swa_model
        
        dataloader_dict = {
            'ETT_h1': Dataloaders_EET_encoded,
            'ETT_h2': Dataloaders_EET_encoded,
            'ETT_m1': Dataloaders_EET_encoded,
            'ETT_m2': Dataloaders_EET_encoded,
            'Electricity': Dataloaders_Electricity_encoded,
        }
        DataLoader_cls = dataloader_dict[dataset_name]
        self.train_loader, \
        self.valid_loader, \
        self.test_loader = DataLoader_cls(
            data_folder_path=data_folder_path,
            scale=scale,
            inverse=inverse,
            timeenc=timeenc,
            freq=freq,
            choosed_variate_names=cols,
            trained_model=trained_model,
            padding=padding,
            pred_len=pred_len,
            device=device,
            batch_size=batch_size,
            dataset_name=dataset_name
        ).gen_encoded_dataloaders()
        
    def _select_optimizer(self):
        lr = self.configs.train_linear_lr
        model_optim = torch.optim.AdamW(self.linear.parameters(), lr=lr)
        return model_optim
    
    def cal_MAE_MSE_loss(self, y_pred, y_true):
        norm_MAE_loss = torch.abs(y_pred - y_true).mean()
        norm_MSE_loss = ((y_pred - y_true) ** 2).mean()
        return norm_MSE_loss, norm_MAE_loss
    
    def _select_criterion(self):
        self.criterion = self.cal_MAE_MSE_loss
        return self.criterion

    def _process_one_batch(self, batch_x):
        device = self.configs.device
        batch_x = batch_x.float().to(device)
        
        with torch.cuda.amp.autocast():  # 混合精度，降低显存占用，提高推理速度
            outputs = self.linear(batch_x)
        return outputs
    
    def evaluation(self, flag="valid"):
        logger = self.logger
        if flag == "valid":
            eval_loader = self.valid_loader
        else:
            eval_loader = self.test_loader
        configs = self.configs
        self.linear.eval()
        eval_MSE_loss = []
        eval_MAE_loss = []
        preds = []
        out = []
        with torch.no_grad():
            for i, (batch_x, labels) in enumerate(eval_loader):
                device = self.configs.device
                batch_x = batch_x.float().to(device)
                labels = labels.float().to(device)
                out = self._process_one_batch(batch_x)
                
                
                MSE_loss, MAE_loss = self.criterion(
                        y_pred = out,
                        y_true = labels
                    )
                eval_MSE_loss.append(MSE_loss.item())
                eval_MAE_loss.append(MAE_loss.item())
                if flag == "test":
                    # 保存pred和true，便于后续可视化
                    pass

        eval_avg_MSE_loss = np.average(eval_MSE_loss).round(4)
        eval_avg_MAE_loss = np.average(eval_MAE_loss).round(4)
        self.linear.train()
        if flag == "test":
            logger.debug(f'Test:\nMSE Loss:    {eval_avg_MSE_loss:.4f}\nMAE Loss:    {eval_avg_MAE_loss:.4f}\n')
            return 
        return eval_avg_MSE_loss, eval_avg_MAE_loss
        
    def train(self):
        configs = self.configs
        logger = self.logger
        patience = configs.early_stop_patience
        train_epochs = configs.self_supervised_train_epochs
        valid_epochs = configs.train_linear_valid_epochs
        use_amp = configs.use_amp
        model_save_path = configs.model_save_dir
        device = configs.device
        train_loader = self.train_loader
        
        time_now = time.time()
        
        early_stopping = EarlyStopping(logger=logger, patience=patience, verbose=True, delta=0.0001)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        logger.debug("Training started ....\n")
        
        for epoch in range(train_epochs):
            iter_count = 0
            train_MSE_loss = []
            train_MAE_loss = []
            
            self.linear.train()
            epoch_time = time.time()
            for i, (batch_x, labels) in enumerate(train_loader):
                model_optim.zero_grad()
                labels = labels.float().to(device)
                out = self._process_one_batch(batch_x)
                
                MSE_loss, MAE_loss = criterion(
                    y_pred = out,
                    y_true = labels
                )
                train_MSE_loss.append(MSE_loss.item())
                train_MAE_loss.append(MAE_loss.item())

                if use_amp:
                    scaler.scale(MSE_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    MSE_loss.backward()
                    model_optim.step()
                
            logger.debug(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.4f} s")

            train_MSE_loss = np.average(train_MSE_loss).round(4)
            train_MAE_loss = np.average(train_MAE_loss).round(4)
            
            if (epoch+1) % valid_epochs == 0:
                valid_MSE_loss, valid_MAE_loss = self.evaluation( flag="valid")

                logger.debug(
                               f'Train MSE Loss*    : {train_MSE_loss:.4f} | Train MAE Loss     : {train_MAE_loss:.4f}\n'
                               f'Valid MSE Loss*     : {train_MSE_loss:.4f} | Valid MAE Loss     : {valid_MAE_loss}')
                
                early_stopping(val_loss=train_MSE_loss, model=self.linear, path=model_save_path)
                
                if early_stopping.early_stop:
                    logger.debug("Early stopping")
                    break
            else:
                logger.debug(
                               f'Train MSE Loss*    : {train_MSE_loss:.4f} | Train MAE Loss     : {train_MAE_loss:.4f}\n')
            
        logger.debug("\n################## Training is Done! #########################")
        
        # 将效果最好的model返回
        best_model_path = os.path.join(model_save_path, "checkpoint.pt")
        self.linear.load_state_dict(torch.load(best_model_path))
        return self.linear
    
