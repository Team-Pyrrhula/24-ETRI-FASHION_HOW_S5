### train task#1 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task1.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
### train task#2 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task2.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model --model_file gAIa-final.pt
### train task#3 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task3.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model --model_file gAIa-final.pt
### train task#4 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task4.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model --model_file gAIa-final.pt
### train task#5 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task5.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model --model_file gAIa-final.pt
### train task#6 ###
sh run_train.sh --in_file_trn_dialog ../aif/data/task6.ddata.wst.txt --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model --model_file gAIa-final.pt
### test ###
sh run_test.sh --in_file_tst_dialog ../aif/data/cl_eval_task1.wst.dev --model_path ./model
sh run_test.sh --in_file_tst_dialog ../aif/data/cl_eval_task2.wst.dev --model_path ./model
sh run_test.sh --in_file_tst_dialog ../aif/data/cl_eval_task3.wst.dev --model_path ./model
sh run_test.sh --in_file_tst_dialog ../aif/data/cl_eval_task4.wst.dev --model_path ./model
sh run_test.sh --in_file_tst_dialog ../aif/data/cl_eval_task5.wst.dev --model_path ./model
sh run_test.sh --in_file_tst_dialog ../aif/data/cl_eval_task6.wst.dev --model_path ./model

