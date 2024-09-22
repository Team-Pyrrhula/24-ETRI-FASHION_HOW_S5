### Sub-Task 3: Continual Learning
- 사용자와 코디 챗봇이 나눈 대화문 및 패션 아이템 메타 데이터를 기반으로, **사용자가 원하는 스타일의 패션 코디 조합을 올바른 순서대로 추천해주는 순위 모델**을 개발합니다.
- 학습에 사용 가능한 데이터셋은 총 6개가 주어지며, **순차적으로** 학습해야 합니다. (**모든 데이터셋을 하나로 합쳐서 사용할 수 없음**)
- 서로 다른 데이터셋을 순차적으로 학습하는 과정에서, 과거에 학습한 정보를 잊어버리는 현상인 **파괴적 망각 현상**이 발생하는데, 이를 해결해야 합니다.
- 정리하자면, **파괴적 망각 현상을 완화화면서 추천 모델을 학습**하는 것이 본 task에서 정의한 문제입니다.
<br><br>

### 디렉토리 구조 설명
- `baseline/`: 대회 주최 측에서 제공한 베이스라인 코드(Synaptic Intelligence)를 기반으로 구현한 디렉토리입니다.
- `reports/`: 데이터, 실험 결과 등을 분석한 노트북 파일들을 모아둔 디렉토리입니다.
- `supsup/`: _Supermasks in Superposition_ 논문을 기반으로 구현한 디렉토리입니다. 리더보드에 최종 제출한 모델을 만들 때 본 디렉토리를 사용했습니다.
<br><br>

### 리더보드 성능 재현을 위한 코드 사용법
- Model Size `345.7167MB`, WKT Score `0.7151515`를 기록한 모델을 재현하는 방법에 대해 설명합니다.
---
**Training**
1. 본 repository를 clone합니다.
1. `packagelist.txt`를 기반으로 실험 환경을 구축합니다.
    - **Note: Kaggle Notebook을 활용해 실험을 진행했던 관계로, 패키지 버전 차이에 따른 성능 차이가 존재할 수 있습니다.**
1. `supsup` 디렉토리로 이동합니다.
1. **`cfgs/09_dec_model_size_with_08_masks.yaml` 파일을 열어 `path` 관련 인자들을 수정합니다.**

    ```yaml
    # global options >> lb wkt 0.7151515, weight size 345.7167
    global:
        seed: 2024
        mode: 'train'
        pred_option: 'masks'
        num_tasks: 6
        use_multimodal: False
 
    path:
        in_file_trn_dialog: '/aif/data/task1.ddata.wst.txt' # 경로 주의
        in_file_tst_dialog: '/aif/data/cl_eval_task1.wst.dev' # 경로 주의
        in_file_fashion: '/aif/data/mdata.wst.txt.2023.08.23' # 경로 주의
        in_file_img_feats: '/aif/data/extracted_feat.json' # 경로 주의
        model_path: './model'
        model_file: '09_dec_model_size_with_08_lightweight.pt'
        subWordEmb_path: '/aif/sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat' # 경로 주의

    data:
        permutation_iteration: 6
        num_augmentation: 5
        corr_thres: 0.9
        mem_size: 16

    model:
        etc:
            use_batch_norm: False
            use_dropout: False
            zero_prob: 0.0

        ReqMLP:
            out_size: 300
            req_node: '[3000,2000,1000,500]'
        
        PolicyNet:
            eval_node: '[3000,3000,1000,500,200][2000]'

    exp:
        learning_rate: 0.0001
        max_grad_norm: 40.0 # deprecated
        batch_size: 100
        epochs: 10
        evaluation_iteration: 10
    ```

1. 터미널에서 `python main.py --cfg_path ./cfgs/09_dec_model_size_with_masks.yaml` 명령어를 입력한 뒤, 정상적으로 실험이 진행되는지 확인합니다.
    - Train, validation으로 구성되어 있는 한 번의 실험이 총 6번(task 개수만큼) 진행됩니다.
1. 실험이 끝나면 `model/` 디렉토리 아래에 yaml 파일 내부에 작성한 `model_file` 이름으로 weight file이 저장됩니다.
---
**Prediction**
- **본 대회의 리더보드 제출 방식 특성 상, 추론 과정은 Python 모듈이 아닌 Jupyter Notebook**을 통해 이루어집니다.
- **가장 중요한 것은 실험 설정이 담겨있는 yaml 파일의 `mode` 값을 `pred`로 변경하는 것**입니다.
    - `mode`에 따라 모델 구조 및 데이터 로드 방식이 달라지기 때문에, prediction을 원하는 경우 **무조건 변경**해야 합니다.
- 자세한 내용은 `task.ipynb`를 참고해주세요.