# SUB_TASK 1

## download_pretrained 
- `timm` 라이브러리의 pretrained 모델 pth를 다운받는 스크립트
- 모델의 용량을 확인하기 위해서 사용

## train.py

 - `기본 학습 run`
```python
train.py <parser.py에 있는 args 설정>
```

- `만약 모델 학습 출력 결과를 저장하고 싶다면`  

`windows`
```bash
New-Item -Path <저장경로 (파일 이름 제외)> -ItemType Directory -Force; python ./train.py <args 설정> | Tee-Object -FilePath <저장할 경로 및 파일이름을 합친 경로>
```
`linux && mac`
```bash
mkdir -p <저장할 경로 (파일 이름 제외)> && python ./train.py <args 설정> | tee <저장할 경로 및 파일이름을 합친 경로>
```

## args
`--model` : 모델 이름 (`timm` 라이브러리 기준)  
`--base_path` : 프로젝트 베이스 경로 (`./`이 기본)  
`--seed` : 사용할 seed 값  
`--epochs` : epochs 수  
`--num_workers` : num_workers 수  
`--train_batch_size` : 학습 배치 사이즈  
`--val_batch_size` : val 배치 사이즈  
`--lr` : 학습률  
`--resize` : 입력 이미지 사이즈  
`--per_iter` : train & val 과정에서 중간 loss 값 출력 주기 (ex 0.2 == 20% 주기로 출력)  
`--save_path` : 저장 경로  
`--wandb` : wandb 사용 여부  
`--project_name` : wandb 프로젝트 이름  
`--model_save_type` : model save type -> script, origin  

## 진행 사항 
 [✔️] `Config`   
 [✔️] `Dataset`  
 [✔️] `Augmentation (Transform)`  
 [✔️] `models`  
 [✔️] `train loops`  
 [✔️] `val loops`  
 [✔️] `wandb` 연동 기능  
 [✔️] `Metric`, `Loose`, `Optimizer`, `Scheduler` 설정 기능  
    - optimizer, loose, scheduler의 변수의 경우 이름을 지정해서 (ex - lr=lr) 넘겨줘야함   
 [❌] `Augmentaion` 시각화 기능