import os
import argparse
import torch
from datetime import datetime
import numpy as np

from utils.print_options import print_options
from utils.copy_Files import copy_Files
from utils.logger import _logger

start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: self_supervised')
parser.add_argument('--selected_dataset', default='Electricity', type=str,
                    help='Dataset of choice: Electricity')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda:0', type=str,
                    help='cpu, cuda:0 or cuda:1')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--method_name', default="TS2Vec", type=str,
                    help='TS2Vec or Ours')
args = parser.parse_args()


device = torch.device(args.device)
experiment_description = args.experiment_description
dataset_name = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description
method_name = args.method_name
training_mode = args.training_mode
data_folder_path = f"/home/HaotianF/Exp/Data/{dataset_name}"
SEED = args.seed

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

model_save_dir = os.path.join(experiment_log_dir, "saved_models")
os.makedirs(model_save_dir, exist_ok=True)

log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)

load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))

print_options(logger)
logger.debug("=" * 45)
logger.debug(f'Dataset: {dataset_name}')
logger.debug(f'Method:  {method_name}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

exec(f'from configs.{method_name}.{dataset_name}_Configs import Configs', globals())
configs = Configs()

configs.__setattr__('experiment_log_dir', experiment_log_dir)
configs.__setattr__('model_load_from', load_from)
configs.__setattr__('logs_save_dir', logs_save_dir)
configs.__setattr__('device', device)
configs.__setattr__('training_mode', training_mode)
configs.__setattr__('dataset_name', dataset_name)
configs.__setattr__('data_folder_path', data_folder_path)
configs.__setattr__('method_name', method_name)
configs.__setattr__('model_save_dir', model_save_dir)
# configs.__setattr__(,)


proj_path = os.path.dirname(os.path.realpath(__file__))
copy_Files(proj_path, experiment_log_dir)
    
if training_mode == "self_supervised":
    exec(f'from train.{training_mode}.exp_{method_name} import Exp_{method_name} as Exp', globals())
    exp = Exp(configs, logger)
    best_model = exp.train()
    
elif training_mode == "train_linear":
    exec(f'from train.tasks.{configs.task}.exp_{method_name} import Exp_{method_name} as Exp', globals())
    print("preparing exp...\n")
    exp = Exp(configs, logger)
    best_model = exp.train()
    exp.evaluation(flag="test")

logger.debug(f"Training time is : {datetime.now()-start_time}\n\n\n")





















