class Configs(object):
    def __init__(self):
        """__setattr__() from main.py
                experiment_log_dir
                logs_save_dir
                device
                training_mode
                dataset_name
                data_path
                method_name
                model_save_dir
        """
        # data
        self.inverse = True
        self.pred_len = 24
        self.seq_len = 3000
        self.timeenc = 1
        self.freq = 'h'
        self.cols = []
        self.task = "forecasting"
        self.scale = True
        self.padding = 200
        self.real_cols = 7 if len(self.cols) == 0 else len(self.cols)
        self.date_cols = {
            "h": 4,
        }
        self.date_cols = self.date_cols[self.freq]
        
         # TSEncoder(swa)
        self.input_dims = self.date_cols+self.real_cols  # freq='h', 4+7(ETT中col不相互独立)
        self.output_dims=320
        self.hidden_dims = 64
        self.depth = 10
        self.mask_mode = 'binomial'
        
        # utils
        self.use_amp = True
        self.early_stop_patience = 4
        
        # self_supervised train
        self.temporal_unit = 0
        self.self_supervised_batch_size = 8
        self.self_supervised_lr = 0.001
        self.self_supervised_train_epochs = 999  # early_stopping
        self.supervised_valid_epochs = 1
        
        
        # train linear
        self.linear_input_dim = self.output_dims
        self.linear_output_dim = self.pred_len*self.real_cols
        self.train_linear_lr = 5e-3  # 0.005
        self.train_linear_valid_epochs = 5
        self.train_linear_batch_size = 2048
        
        
        