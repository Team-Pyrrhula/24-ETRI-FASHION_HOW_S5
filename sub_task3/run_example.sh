#!/bin/bash
read -p "실험 결과 기록에 사용할 실험명을 작성해주세요: " exp_name

### train task#1 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task1.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
### test task#1 ###
sh run_test.sh --exp_name $exp_name --task_ids /1/1 --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /1/2 --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /1/3 --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /1/4 --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /1/5 --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /1/6 --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model

### train task#2 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task2.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model --model_file gAIa-final.pt
### test task#2 ###
sh run_test.sh --exp_name $exp_name --task_ids /2/1 --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /2/2 --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /2/3 --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /2/4 --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /2/5 --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /2/6 --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model

### train task#3 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task3.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model --model_file gAIa-final.pt
### test task#3 ###
sh run_test.sh --exp_name $exp_name --task_ids /3/1 --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /3/2 --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /3/3 --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /3/4 --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /3/5 --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /3/6 --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model

### train task#4 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task4.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model --model_file gAIa-final.pt
### test task#4 ###
sh run_test.sh --exp_name $exp_name --task_ids /4/1 --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /4/2 --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /4/3 --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /4/4 --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /4/5 --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /4/6 --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model

### train task#5 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task5.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model --model_file gAIa-final.pt
### test task#5 ###
sh run_test.sh --exp_name $exp_name --task_ids /5/1 --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /5/2 --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /5/3 --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /5/4 --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /5/5 --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /5/6 --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model

### train task#6 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task6.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model --model_file gAIa-final.pt
### test task#6 ###
sh run_test.sh --exp_name $exp_name --task_ids /6/1 --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /6/2 --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /6/3 --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /6/4 --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /6/5 --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --exp_name $exp_name --task_ids /6/6 --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model