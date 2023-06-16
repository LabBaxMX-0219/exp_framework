import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import numpy as np

from methods.TS2Vec.encoder import TSEncoder
from methods.TS2Vec.utils import take_per_row

from utils.early_stopping import EarlyStopping
from utils.cal_classification_metrics import calc_classification_metrics

from data.datasets import DataLoaders_classification

class Exp_TS2Vec():
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger
        self.linear = self._load_model()
        self._get_dataloader()
        
    def _load_model(self):
        configs = self.configs
        device = configs.device
        linear_in = configs.linear_input_dim
        linear_out = configs.linear_output_dim
        
        linear = nn.Linear(
            in_features=linear_in,
            out_features=linear_out,
            device=device
        )
        
        return linear
        
    def _get_dataloader(self):
        configs = self.configs
        device = configs.device
        dataset_name = configs.dataset_name
        task = configs.task
        batch_size = configs.train_linear_batch_size
        data_folder_path = configs.data_folder_path
        
        
        # load pretrained_model
        load_from = configs.model_load_from
        linear_in = configs.linear_input_dim
        linear_out = configs.linear_output_dim
        
        base_model = TSEncoder(configs).to(device=device)
        swa_model = torch.optim.swa_utils.AveragedModel(base_model)
        swa_model.update_parameters(base_model)
        
        pretrained_dict = torch.load(os.path.join(load_from, "checkpoint.pt"), map_location=device)
        swa_model.load_state_dict(pretrained_dict)
        self.trained_model = swa_model
        
        dataset_dict = {
            'HAR': DataLoaders_classification,
            'sleepEDF': DataLoaders_classification,
            'Epilepsy': DataLoaders_classification,
        }
        dataset_cls = dataset_dict[dataset_name]
        self.train_loader, \
        self.valid_loader, \
        self.test_loader = dataset_cls(
            data_folder_path=data_folder_path,
            batch_size=batch_size
        ).get_dataloader()

    def _select_optimizer(self):
        lr = self.configs.train_linear_lr
        model_optim = torch.optim.AdamW(self.linear.parameters(), lr=lr)
        return model_optim
    
    def _select_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion

    def _process_one_batch(self, batch_x):
        device = self.configs.device
        batch_x = batch_x.float().to(device)
        
        with torch.cuda.amp.autocast():  # 混合精度，降低显存占用，提高推理速度
            reprs = self.trained_model(batch_x)  #[BN, timesteps, outdim]
            reprs = F.max_pool1d(
                reprs.transpose(1, 2),
                kernel_size = reprs.size(1),  # timesteps
            ).transpose(1, 2).squeeze(1)  # [BN, outdim]
            outputs = self.linear(reprs)  # [BN, n_classes]
        return outputs
    
    def evaluation(self, flag="valid"):
        logger = self.logger
        if flag == "valid":
            eval_loader = self.valid_loader
        else:
            eval_loader = self.test_loader
        configs = self.configs
        experiment_log_dir = configs.experiment_log_dir
        device = configs.device
        self.linear.eval()
        total_loss = []
        total_acc = []
        pred_labels = np.array([])
        true_labels = np.array([])
        with torch.no_grad():
            for i, (batch_x, labels) in enumerate(eval_loader):
                
                batch_x = batch_x.float().to(device)
                labels = labels.long().to(device)
                outs = self._process_one_batch(batch_x)
                preds = outs.detach().argmax(dim=1)
                loss = self.criterion(
                        outs,
                        labels
                    )
                total_loss.append(loss.item())
                total_acc.append(labels.eq(preds).float().mean().item())
                
                if flag == "test":
                    # 保存pred和true，便于后续可视化
                    pred_labels = np.append(pred_labels, preds.cpu().numpy())  # [N, n_classes]
                    true_labels = np.append(true_labels, labels.data.cpu().numpy())  # [N]

        eval_avg_loss = np.average(total_loss).round(4)
        eval_avg_acc = np.average(total_acc).round(4)
        self.linear.train()
        if flag == "test":
            acc, maf1, mif1, wf1 = calc_classification_metrics(pred_labels, true_labels, experiment_log_dir)
            logger.debug(f'Test:\nLoss:    {eval_avg_loss:.4f}\nAcc:    {acc:.4f} | maf1:    {maf1:.4f} | mif1:    {mif1:.4f} | wf1:    {wf1:.4f}\n')
            # return eval_avg_loss, eval_avg_acc
        return eval_avg_loss, eval_avg_acc
        
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
        
        early_stopping = EarlyStopping(logger=logger, patience=patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        logger.debug("Training started ....\n")
        
        for epoch in range(train_epochs):
            iter_count = 0
            total_loss = []
            total_acc = []
            
            self.linear.train()
            epoch_time = time.time()
            for i, (batch_x, labels) in enumerate(train_loader):
                model_optim.zero_grad()
                labels = labels.long().to(device)
                outs = self._process_one_batch(batch_x)
                preds = outs.detach().argmax(dim=1)
                loss = self.criterion(
                        outs,    # [N, n_classes]
                        labels  # [N]
                    )
                total_loss.append(loss.item())
                total_acc.append(labels.eq(preds).float().mean().item())

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
            logger.debug(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.4f} s")

            train_loss = np.average(total_loss).round(4)
            train_acc = np.average(total_acc).round(4)
            
            if (epoch+1) % valid_epochs == 0:
                valid_loss, valid_acc = self.evaluation(flag="valid")

                logger.debug(f'Train Loss*     : {train_loss:.4f} | Train Acc     : {train_acc:.4f}\n'
                               f'Valid Loss     : {valid_loss:.4f} | Valid Acc     : {valid_acc:.4f}')
                
                early_stopping(val_loss=valid_loss, model=self.linear, path=model_save_path)
                
                if early_stopping.early_stop:
                    logger.debug("Early stopping")
                    break
            else:
                logger.debug(f'Train Loss*     : {train_loss:.4f} | Train Acc     : {train_acc:.4f}\n')
                
        logger.debug("\n################## Training is Done! #########################")
        
        # 将效果最好的model返回
        best_model_path = os.path.join(model_save_path, "checkpoint.pt")
        self.linear.load_state_dict(torch.load(best_model_path))
        return self.linear
    
