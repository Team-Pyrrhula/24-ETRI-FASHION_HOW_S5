# $1: --in_file_trn_dialog
# $2: filename of $1
# $3: --in_file_tst_dialog
# $4: filename of $3
# $5: --model_path
# $6: path for saving trained model
# $7: --model_file
# $8: filename of $7 (loaded file after task#1)

CUDA_VISIBLE_DEVICES="0" python3 ./main.py --exp_name baseline \
                                     --seed 2024 \
                                     --mode train \
                                     --task_ids /1/1 \
                                     --in_file_fashion ../aif/data/mdata.wst.txt.2023.08.23 \
                                     --subWordEmb_path ../aif/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                     --mem_size 16 \
                                     --key_size 300 \
                                     --hops 3 \
                                     --eval_node [6000,6000,200][2000] \
                                     --epochs 80 \
                                     --save_freq 160 \
                                     --batch_size 100 \
                                     --learning_rate 0.005 \
                                     --max_grad_norm 20.0 \
                                     --use_dropout True \
                                     --zero_prob 0.5 \
                                     --permutation_iteration 3 \
                                     --num_augmentation 5 \
                                     --corr_thres 0.7 \
                                     $1 $2 \
                                     $3 $4 \
                                     $5 $6 \
                                     $7 $8
