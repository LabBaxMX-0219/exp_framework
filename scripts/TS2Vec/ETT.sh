# ETTh1
conda activate DL; cd /home/HaotianF/Exp/Ours/baselines/exp_framwork; nohup python main.py --device cuda:0 --logs_save_dir /home/HaotianF/Exp/Results --experiment_description TS2Vec --method_name TS2Vec --run_description ETT_h1 --training_mode self_supervised --seed 123 --selected_dataset ETT_h1 >> TS2Vec_ETT_h1.log 2>&1 &

conda activate DL; cd /home/HaotianF/Exp/Ours/baselines/exp_framwork; nohup python main.py --device cuda:0 --logs_save_dir /home/HaotianF/Exp/Results --experiment_description TS2Vec --method_name TS2Vec --run_description ETT_h1 --training_mode train_linear --seed 123 --selected_dataset ETT_h1 >> TS2Vec_ETT_h1.log 2>&1 &

# ETTh2
conda activate DL; cd /home/HaotianF/Exp/Ours/baselines/exp_framwork; nohup python main.py --device cuda:0 --logs_save_dir /home/HaotianF/Exp/Results --experiment_description TS2Vec --method_name TS2Vec --run_description ETT_h2 --training_mode self_supervised --seed 123 --selected_dataset ETT_h2 >> TS2Vec_ETT_h2.log 2>&1 &

conda activate DL; cd /home/HaotianF/Exp/Ours/baselines/exp_framwork; nohup python main.py --device cuda:0 --logs_save_dir /home/HaotianF/Exp/Results --experiment_description TS2Vec --method_name TS2Vec --run_description ETT_h2 --training_mode train_linear --seed 123 --selected_dataset ETT_h2 >> TS2Vec_ETT_h2.log 2>&1 &

# ETTm1
conda activate DL; cd /home/HaotianF/Exp/Ours/baselines/exp_framwork; nohup python main.py --device cuda:0 --logs_save_dir /home/HaotianF/Exp/Results --experiment_description TS2Vec --method_name TS2Vec --run_description ETT_m1 --training_mode self_supervised --seed 123 --selected_dataset ETT_m1 >> TS2Vec_ETT_m1.log 2>&1 &

conda activate DL; cd /home/HaotianF/Exp/Ours/baselines/exp_framwork; nohup python main.py --device cuda:0 --logs_save_dir /home/HaotianF/Exp/Results --experiment_description TS2Vec --method_name TS2Vec --run_description ETT_m1 --training_mode train_linear --seed 123 --selected_dataset ETT_m1 >> TS2Vec_ETT_m1.log 2>&1 &