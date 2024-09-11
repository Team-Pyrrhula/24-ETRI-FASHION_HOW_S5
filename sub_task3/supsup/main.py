# torch
import torch

# built-in library
import argparse
import os

# custom modules
from exp_manager import Manager
from exp_utils import set_seed, get_udevice, load_cfg

# global settings
cores = os.cpu_count()
torch.set_num_threads(cores)


### input options ###
parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')

parser.add_argument('--cfg_path', type=str,
                    default='./cfgs/00_baseline.yaml',
                    help="실험에 필요한 값들을 설정해둔 yaml 파일의 경로를 입력합니다.")

args, _ = parser.parse_known_args()
### input options ###


if __name__ == '__main__':
    
    print('\n')
    print('-'*60)
    print('\t\tAI Fashion Coordinator')
    print('-'*60)
    print('\n')

    # yaml 파일 로드
    cfg = load_cfg(args.cfg_path)

    # 실험 유효성 검증
    mode = cfg['global']['mode']
    if mode not in ['train', 'eval', 'test', 'pred'] :
        raise ValueError('Unknown mode {}'.format(mode))

    # 시드 세팅
    set_seed(cfg['global']['seed'])

    # 실험 인자 확인
    print('<Parsed arguments>')
    for category, value in cfg.items():
        print(f"##### {category} #####")
        for name, value in cfg[category].items():
            print(f"{name}: {value}")

        print('-' * 20)

    # 실험 매니저 객체 선언
    manager = Manager(cfg=cfg, device=get_udevice(),
                      exp_name=args.cfg_path.split('/')[-1].split('.')[0])

    # 학습 및 평가
    manager.continual_learning()
