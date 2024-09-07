import argparse

def parser_arguments():
    """
    parser 설정
    """
    parser = argparse.ArgumentParser(description="train & val parser")
    parser.add_argument("--model", type=str, default="mobilenetv4_conv_small.e2400_r224_in1k", help="MODEL_NAME !!")
    parser.add_argument("--pretrain", type=int, default=0, help='use pretrained model')
    parser.add_argument("--base_path", type=str, default="./", help="project base path")
    parser.add_argument("--seed", type=int, default=42, help="seed num")
    parser.add_argument("--epochs", type=int, default=50, help="epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="num worker num")
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--val_batch_size", type=int, default=128, help="val batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--resize", type=int, default=224, help="img resize size")
    parser.add_argument("--val_resize", type=int, default=224, help="val img resize size")
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss", help="train criterion")
    parser.add_argument("--optimizer", type=str, default="Adam", help="train optimzier")
    parser.add_argument("--scheduler", type=str, default="StepLR", help="train scheduler")
    parser.add_argument("--per_iter", type=float, default=0.3, help="print pipeline iter nums")
    parser.add_argument("--save_path", type=str, default='save', help="save path")
    parser.add_argument("--model_save_type", type=str, default='origin', help="torch model save method [script, origin]")
    parser.add_argument("--val_metric", type=str, default='f1', help='val metric')
    parser.add_argument("--save_img", action='store_true', help='save_aug_img')
    
    #tain val df path
    parser.add_argument("--train_df", type=str, default="Fashion-How24_sub2_train.csv")
    parser.add_argument("--val_df", type=str, default="Fashion-How24_sub2_val.csv")

    # wandb
    parser.add_argument("--wandb", action='store_true', help='wandb use flag')
    parser.add_argument("--project_name", type=str, default='ETRI_sub-task_2', help='wandb project name')
    
    #remgb
    parser.add_argument("--remgb", type=int, default=0, help="use remgb dataset")

    # crop
    parser.add_argument("--crop", type=int, default=0, help="use crop image")

    # sampling
    parser.add_argument("--sampler", type=int, default=0, help="use smapler")

    # img_type
    parser.add_argument("--img_type", type=str, default='RGB', help='using img open tpye : [RGB, LAB, HSV]')

    # custom aug
    parser.add_argument("--custom_aug", action='store_true', help='custom augmentation use')
    
    # dataset mean std 
    parser.add_argument("--mean_std", action='store_true', help='dataset mean std calcul')

    # save model float16
    parser.add_argument("--model_half", action='store_true', help='save model half')
    
    # class weight
    parser.add_argument("--class_weight", action='store_true', help='class weight')

    # class aug
    parser.add_argument("--class_aug", action='store_true', help='class_aug')

    #mix up
    parser.add_argument("--mixup", action='store_true', help='mixup')
    args = parser.parse_args()
    return args

def val_parser_arguments():
    parser = argparse.ArgumentParser(description="Val parser")
    parser.add_argument("--model", type=str, help='inference model name')
    parser.add_argument("--save_path", type=str, default='./save/', help='trained model save path')
    parser.add_argument("--model_path", type=str, help='saved model path')
    parser.add_argument("--resize", type=int, default=224, help='model input size')
    
    args = parser.parse_args()
    return args