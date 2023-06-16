class Configs(object):
    def __init__(self):
        # TSEncoder(swa)
        self.input_dims = 9  # [BN, 128, 9]
        self.output_dims=320
        self.hidden_dims = 64
        self.depth = 10
        self.mask_mode = 'binomial'

        """__setattr__()
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
        self.task = "classification"
        self.n_classes = 6
        
        # utils
        self.use_amp = True
        self.early_stop_patience = 4
        
        # self_supervised train
        self.temporal_unit = 0
        self.self_supervised_batch_size = 64
        self.self_supervised_lr = 0.001
        self.self_supervised_train_epochs = 999  # early_stopping
        self.supervised_valid_epochs = 5
        
        
        # train linear
        self.linear_input_dim = self.output_dims
        self.linear_output_dim = self.n_classes
        self.train_linear_lr = 0.01
        self.train_linear_valid_epochs = 10
        self.train_linear_batch_size = 128
        
        
        