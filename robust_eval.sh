CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --save_path robust_results/EASE --file_package checkpoint_path --config_file configs/eval_mosi.yaml --key_eval Has0_acc_2
CUDA_VISIBLE_DEVICES=1 python robust_evaluation.py --save_path robust_results/EASE --file_package checkpoint_path --config_file configs/eval_mosei.yaml --key_eval Has0_acc_2
CUDA_VISIBLE_DEVICES=2 python robust_evaluation.py --save_path robust_results/EASE --file_package checkpoint_path --config_file configs/eval_sims.yaml --key_eval Mult_acc_2
