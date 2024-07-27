import argparse

def parser_arguments():
    """
    parser 설정
    """
    parser = argparse.ArgumentParser(description="train & val parser")
    parser.add_argument("--model", type=str, default="mobilenetv4_conv_small.e2400_r224_in1k", help="MODEL_NAME !!")
    parser.add_argument("--base_path", type=str, default="./", help="project base path")
    parser.add_argument("--seed", type=int, default=42, help="seed num")
    parser.add_argument("--epochs", type=int, default=50, help="epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="num worker num")
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--val_batch_size", type=int, default=128, help="val batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--resize", type=int, default=224, help="img resize size")
    parser.add_argument("--per_iter", type=float, default=0.3, help="print pipeline iter nums")
    parser.add_argument("--save_path", type=str, default='save', help="save path")

    args = parser.parse_args()
    return args