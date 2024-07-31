'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2023, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2023.02.15.
'''
# torch
import torch

# built-in library
import argparse
import os

# custom modules
from baseline.gaia import *
from exp_utils import set_seed

# global settings
cores = os.cpu_count()
torch.set_num_threads(cores)


def get_udevice() -> torch.device:
    """
    학습 및 추론 시 사용가능한 장치를 찾아 반환합니다.

    Returns:
        torch.device: 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = torch.cuda.device_count()

    else:    
        device = torch.device('cpu')

    print('Using device: {}'.format(device))

    if torch.cuda.is_available():
        print('# of GPU: {}'.format(num_gpu))

    return device

def str2bool(v: str | bool) -> bool:
    """
    string을 bool로 변환합니다.

    Args:
        v (str | bool): 터미널에서 입력된 실험 인자입니다.

    Raises:
        argparse.ArgumentTypeError: string, bool이 아닌 type이 입력되는 경우 예외 처리합니다.

    Returns:
        bool: True 또는 False를 반환합니다.
    """
    if isinstance(v, bool): 
        return v 
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


### input options ###
parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')

# global options
parser.add_argument('--exp_name', type=str,
                    default='exp_cl',
                    help='기록 시 사용할 실험 이름을 입력합니다.')
parser.add_argument('--seed', type=int,
                    default=2024,
                    help='실험 재현을 위한 시드 값을 입력합니다.')
parser.add_argument('--mode', type=str,
                    default='test',
                    help='실행할 모드를 선택합니다. train/test/pred 세 가지 모드가 존재합니다.')
parser.add_argument('--task_ids', type=str,
                    default='/1/1',
                    help='현재 평가 중인 모델을 학습할 때, 마지막으로 사용한 데이터의 task id를 입력합니다.')
parser.add_argument('--model_file', type=str, # TODO: model_file 옵션이 실제로 사용되는지 체크하기
                    default=None, 
                    help='모델 파일의 이름을 입력합니다.')                    
parser.add_argument('--use_multimodal', type=str2bool, # TODO: use_multimodal option 점검하기
                    default=False, 
                    help='텍스트와 함께 이미지 데이터를 사용할지 선택합니다.')
parser.add_argument('--use_cl', type=str2bool,
                    default=True,
                    help='CL 알고리즘을 적용할지 선택합니다.')

# path options
parser.add_argument('--in_file_trn_dialog', type=str, 
                    default='../data/task1.ddata.wst.txt', 
                    help='학습 대화문 데이터의 경로를 입력합니다.')
parser.add_argument('--in_file_tst_dialog', type=str, 
                    default='../data/cl_eval_task1.wst.dev', 
                    help='테스트 대화문 데이터의 경로를 입력합니다.')
parser.add_argument('--in_file_fashion', type=str, 
                    default='../data/mdata.wst.txt.2023.08.23', 
                    help='패션 아이템 메타데이터의 경로를 입력합니다.')
parser.add_argument('--in_file_img_feats', type=str, # TODO: json 파일 존재 여부를 대회 운영 측에 문의하기
                    default='../data/extracted_feat.json', 
                    help='패션 아이템 이미지의 경로를 입력합니다.')
parser.add_argument('--model_path', type=str, 
                    default='./model', 
                    help='모델을 저장하거나 불러올 경로를 입력합니다.')
parser.add_argument('--subWordEmb_path', type=str, 
                    default='../sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat', 
                    help='subword embedding의 경로를 입력합니다.')

# data options
parser.add_argument('--permutation_iteration', type=int,
                    default=3,   
                    help='생성할 순열의 개수를 설정합니다.')
parser.add_argument('--num_augmentation', type=int,
                    default=3,   
                    help='데이터 증강 횟수를 설정합니다.')                    
parser.add_argument('--corr_thres', type=float,
                    default=0.7, 
                    help='데이터 증강 시 사용할 correlation threshold를 설정합니다.')                    

# MemN2N options
parser.add_argument('--hops', type=int, # TODO: MemN2N 리뷰
                    default=3,   
                    help='MemN2N의 hops을 설정합니다.')
parser.add_argument('--mem_size', type=int,
                    default=16,   
                    help='MemN2N의 memory size를 설정합니다.')
parser.add_argument('--key_size', type=int,
                    default=300,   
                    help='MemN2N의 key size를 설정합니다.')

# PolicyNet options
parser.add_argument('--eval_node', type=str, 
                    default='[6000,6000,200][2000]', 
                    help='PolicyNet의 evaluation network의 mlp 노드 구성을 입력합니다.')
parser.add_argument('--use_batch_norm', type=str2bool, 
                    default=False, 
                    help='PolicyNet에 batch normalization을 적용할지 선택합니다.')
parser.add_argument('--use_dropout', type=str2bool, 
                    default=False, 
                    help='PolicyNet에 dropout을 적용할지 선택합니다.')
parser.add_argument('--zero_prob', type=float,
                    default=0.0, 
                    help='PolicyNet의 dropout 비율을 설정합니다.')

# exp options
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, 
                    help='학습률을 설정합니다.')
parser.add_argument('--max_grad_norm', type=float,
                    default=40.0, 
                    help='gradient normalization의 상한선을 설정합니다.')
parser.add_argument('--batch_size', type=int,
                    default=100,   
                    help='학습에 적용할 batch size를 설정합니다.')
parser.add_argument('--epochs', type=int,
                    default=10,   
                    help='학습에 적용할 총 epochs을 설정합니다.')
parser.add_argument('--save_freq', type=int,
                    default=2,   
                    help='모델 저장 주기를 설정합니다.')
parser.add_argument('--evaluation_iteration', type=int,
                    default=10,   
                    help='평가 횟수를 설정합니다.')

args, _ = parser.parse_known_args()
### input options ###


# TODO: wandb 추가 or 실험 기록을 위한 모듈 작성
if __name__ == '__main__':
    
    print('\n')
    print('-'*60)
    print('\t\tAI Fashion Coordinator')
    print('-'*60)
    print('\n')

    mode = args.mode    
    if mode not in ['train', 'test', 'pred'] :
        raise ValueError('Unknown mode {}'.format(mode))

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')
    
    gaia = gAIa(args, get_udevice())
    if mode == 'train':
        # training
        gaia.train()
    elif mode == 'test':
        # test
        gaia.test()
    elif mode == 'pred':
        # pred
        gaia.pred()

