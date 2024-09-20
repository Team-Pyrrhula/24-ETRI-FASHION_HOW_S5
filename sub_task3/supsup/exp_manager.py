# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# built-in library
import os
import timeit
from tqdm import tqdm

# external library
import numpy as np
from scipy import stats

# custom modules
from file_io import *
from exp_utils import save_results
from supermask.base import SupSupMLP
from supermask.utils import set_model_task, cache_masks, rm_scores


### global params ###
NUM_ITEM_IN_COORDI = 4 # of items in fashion coordination      
NUM_META_FEAT = 4 # of metadata features    
NUM_RANKING = 3 # of fashion coordination candidates        
IMG_FEAT_SIZE = 4096 # image feature size


class Manager(object):
    def __init__(self, cfg: dict, device=torch.device, exp_name: str = 'supsup_baseline'):
        """
        실험에 필요한 변수, 데이터, 모델 등을 생성하고 관리합니다.

        Args:
            cfg (dict): yaml 파일로부터 불러온 실험 설정 값입니다.
            device (torch.device): 학습 및 추론 시 사용할 장치입니다.
            exp_name (str): 실험 이름입니다.
        """
        # 카테고리별로 설정 값 분리
        global_cfg = cfg['global']
        path_cfg = cfg['path']
        data_cfg = cfg['data']
        model_cfg = cfg['model']
        exp_cfg = cfg['exp']

        # global variables
        self._device = device
        self._exp_name = exp_name
        self._mode = global_cfg['mode']
        self._pred_option = global_cfg['pred_option']
        self._num_tasks = global_cfg['num_tasks']

        # path variables
        self._model_path = path_cfg['model_path']
        self._model_file = path_cfg['model_file']
        self._in_file_trn_dialog = path_cfg['in_file_trn_dialog']
        self._in_file_tst_dialog = path_cfg['in_file_tst_dialog']
        self._in_file_fashion = path_cfg['in_file_fashion']

        # data variables
        self._permutation_iteration = data_cfg['permutation_iteration']
        self._num_augmentation = data_cfg['num_augmentation']
        self._corr_thres = data_cfg['corr_thres']
        self._mem_size = data_cfg['mem_size']

        # exp variables
        self._batch_size = exp_cfg['batch_size']
        self._epochs = exp_cfg['epochs']
        self._max_grad_norm = exp_cfg['max_grad_norm']
        self._num_eval = exp_cfg['evaluation_iteration']
        
        # PolicyNet variables
        use_dropout = model_cfg['etc']['use_dropout']
        if global_cfg['mode'] == 'test':
            use_dropout = False

        # class instance for subword embedding
        self._swer = SubWordEmbReaderUtil(path_cfg['subWordEmb_path'])
        self._emb_size = self._swer.get_emb_size() # 128
        self._meta_size = NUM_META_FEAT
        self._coordi_size = NUM_ITEM_IN_COORDI
        self._num_rnk = NUM_RANKING
        feats_size = IMG_FEAT_SIZE

        # read metadata DB
        self._metadata, self._idx2item, self._item2idx, \
            self._item_size, self._meta_similarities, \
            self._feats = make_metadata(self._in_file_fashion, self._swer, 
                                self._coordi_size, self._meta_size,
                                global_cfg['use_multimodal'], path_cfg['in_file_img_feats'],
                                feats_size)
        
        # build model
        self._model = SupSupMLP(data_cfg['mem_size'], self._emb_size, model_cfg['ReqMLP']['out_size'],
                                self._meta_size, self._coordi_size, self._num_rnk,
                                model_cfg['ReqMLP']['req_node'], model_cfg['PolicyNet']['eval_node'],
                                model_cfg['etc']['use_batch_norm'], use_dropout, model_cfg['etc']['zero_prob'],
                                global_cfg['use_multimodal'], feats_size, self._mode, self._pred_option, self._num_tasks)
        
        # check model
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)

        if self._mode == 'train':
            # optimizer
            self._optimizer = optim.RMSprop([p for p in self._model.parameters() if p.requires_grad],
                                            lr=exp_cfg['learning_rate'])
            
            # loss function
            self._criterion = nn.CrossEntropyLoss()


    def _get_loss(self, batch: List[torch.tensor]) -> torch.tensor:
        """loss를 구합니다.

        Args:
            batch (List[torch.tensor]): (대화문, 코디 조합, 코디 조합의 순위)로 구성된 배치입니다.

        Returns:
            torch.tensor: loss 계산 결과입니다.
        """
        dlg, crd, rnk = batch
        logits, _ = self._model(dlg, crd)
        loss = self._criterion(logits, rnk) * self._batch_size

        return loss
    

    def prepare_data(self) -> None:
        """self._mode에 따라 학습/평가 데이터셋을 생성합니다.
        24.09.20 기준 ['train'], ['eval', 'test', 'pred'] 모드가 존재하며,
        ['eval', 'test', 'pred'] 모드는 모두 동일한 데이터셋 생성 로직을 사용합니다.
        """
        # 학습용 데이터 준비
        if self._mode == 'train':
            self._dlg, self._crd, self._rnk = make_io_data('prepare', 
                        self._in_file_trn_dialog, self._swer, self._mem_size,
                        self._coordi_size, self._item2idx, self._idx2item, 
                        self._metadata, self._meta_similarities, self._num_rnk,
                        self._permutation_iteration, self._num_augmentation, 
                        self._corr_thres, self._feats)
            
            self._num_examples = len(self._dlg)

            # dataloader
            dataset = TensorDataset(torch.tensor(self._dlg), 
                                    torch.tensor(self._crd), 
                                    torch.tensor(self._rnk, dtype=torch.long))
            self._dataloader = DataLoader(dataset, 
                                    batch_size=self._batch_size, shuffle=True)

        # 평가용 데이터 준비
        elif self._mode in ['eval', 'test', 'pred']:
            self._tst_dlg, self._tst_crd, _ = make_io_data('eval', 
                self._in_file_tst_dialog, self._swer, self._mem_size,
                self._coordi_size, self._item2idx, self._idx2item, 
                self._metadata, self._meta_similarities, self._num_rnk,
                self._num_augmentation, self._num_augmentation, 
                self._corr_thres, self._feats)
        
            self._num_examples = len(self._tst_dlg)


    def _calculate_weighted_kendall_tau(self, pred: np.array, label: np.array, rnk_lst: np.array) -> float:
        """WKT(Weighted Kendall Tau correlation) score를 계산하여 반환합니다.

        Args:
            pred (np.array): 입력 조합에 대해 모델이 예측한 순위입니다.
            label (np.array): 입력 조합의 실제 순위입니다.
            rnk_lst (np.array): 아이템 배치 순서를 나타내는 순열입니다.
                - ex) num_rank: 3 >> np.array([[0, 1, 2], # rank 0
                                                [0, 2, 1], # rank 1
                                                [1, 0, 2], # rank 2
                                                [1, 2, 0], # rank 3
                                                [2, 0, 1], # rank 4
                                                [2, 1, 0]]) # rank 5

        Returns:
            float: -1 ~ 1 사이의 WKT score입니다.
        """
        total_count = 0
        total_corr = 0
        for p, l in zip(pred, label):
            corr, _ = stats.weightedtau(self._num_rnk-1-rnk_lst[l], #
                                        self._num_rnk-1-rnk_lst[p]) #
            total_corr += corr
            total_count += 1

        return (total_corr / total_count)


    def _evaluate(self, eval_dlg: np.array, eval_crd: np.array) -> tuple:
        """모델 성능을 평가합니다.

        Args:
            eval_dlg (np.array): 평가에 사용할 대화문 데이터입니다.
            eval_crd (np.array): 평가에 사용할 코디 조합 데이터입니다.

        Returns:
            tuple: 
                - repeated_preds: 모델의 예측 결과를 모두 저장한 리스트입니다.
                - np.array(eval_corr): WKT score를 모두 저장한 리스트입니다.
                - eval_num_examples: 평가 데이터의 개수입니다.
        """
        # 데이터 개수 확인
        eval_num_examples = eval_dlg.shape[0]

        # wkt score를 저장할 배열 선언
        eval_corr = []

        # 코디 조합에 대한 순위 순열 생성
        rank_lst = np.array(list(permutations(np.arange(self._num_rnk), 
                                              self._num_rnk)))

        # 대화문 데이터를 device로 옮김                                              
        eval_dlg = torch.tensor(eval_dlg).to(self._device)

        # 모델의 예측 결과를 모두 저장할 배열 선언
        repeated_preds = []

        # self._num_eval만큼 성능 평가 과정을 반복
        for _ in range(self._num_eval):
            preds = []

            # 코디 조합을 무작위로 섞어서 다양한 순위 데이터를 생성하고, device로 옮김
            coordi, rnk = shuffle_coordi_and_ranking(eval_crd, self._num_rnk)
            coordi = torch.tensor(coordi).to(self._device)
            
            # 성능 평가 진행
            # dataloader를 사용하지 않기 때문에, batch_size만큼 잘라서 모델에 입력
            for start in range(0, eval_num_examples, self._batch_size):
                end = start + self._batch_size

                # 에러 방지
                if end > eval_num_examples:
                    end = eval_num_examples

                # 결과 예측
                _, pred = self._model(eval_dlg[start:end], 
                                      coordi[start:end])
                pred = pred.cpu().numpy()

                # 예측 결과를 한 개씩 저장
                for j in range(end-start):
                    preds.append(pred[j])

            # dtype 변환
            preds = np.array(preds)

            # WKT score 계산 및 저장
            corr = self._calculate_weighted_kendall_tau(preds, rnk, rank_lst)
            eval_corr.append(corr)
            
            # 예측 결과 저장
            repeated_preds.append(preds)

        return repeated_preds, np.array(eval_corr), eval_num_examples


    def _predict(self, eval_dlg: np.array, eval_crd: np.array) -> tuple:
        """평가용 데이터셋에 대한 모델의 예측 결과를 구할 때 사용합니다.

        Args:
            eval_dlg (np.array): 평가용 대화문 임베딩 벡터입니다.
            eval_crd (np.array): 평가용 코디 조합 임베딩 벡터입니다.

        Returns:
            tuple: 
                - pred: 모델이 예측한 순위입니다.
                - eval_num_examples: 평가 데이터셋의 개수입니다.
        """
        eval_num_examples = eval_dlg.shape[0]

        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        eval_crd = torch.tensor(eval_crd).to(self._device)

        preds = []
        for start in range(0, eval_num_examples, self._batch_size):
            end = start + self._batch_size
            if end > eval_num_examples:
                end = eval_num_examples

            _, pred = self._model(eval_dlg[start:end],
                                  eval_crd[start:end])
            
            pred = pred.cpu().numpy()
            for j in range(end-start):
                preds.append(pred[j])

        preds = np.array(preds)

        return preds, eval_num_examples    
    

    def train(self):
        """모델을 학습하기 위해 사용합니다.
        """
        print('\n<Train>')
        print('total examples in dataset: {}'.format(self._num_examples))

        self._model.train()

        # 모델 파일을 저장할 디렉토리 생성
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

        # cpu or gpu로 모델을 옮김
        self._model.to(self._device)

        # 학습
        for epoch in range(1, self._epochs + 1):
            time_start = timeit.default_timer()
            losses = []
            iter_bar = tqdm(self._dataloader)

            for batch in iter_bar:
                # optimizer gradient 초기화
                self._optimizer.zero_grad()

                # data를 device로 이동
                batch = [t.to(self._device) for t in batch]

                # loss 계산
                loss = self._get_loss(batch).mean()

                # gradient 계산 및 score update
                loss.backward()
                self._optimizer.step()

                # loss 기록
                losses.append(loss)

            time_end = timeit.default_timer()

            # Epoch별 학습 상태 출력
            print('-'*50)
            print('Epoch: {}/{}'.format(epoch, self._epochs))
            print('Time: {:.2f}sec'.format(time_end - time_start))
            print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('-'*50)


    def test(self, task_id) -> np.array:
        """모델 성능을 평가하고 기록합니다.
        """
        time_start = timeit.default_timer()
        self._model.eval()
        
        # 모델 성능 평가
        _, test_corr, num_examples = self._evaluate(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()

        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('Average WKTC over iterations: {:.4f}'.format(np.mean(test_corr)))
        print('Best WKTC: {:.4f}'.format(np.max(test_corr)))
        print('-'*50)

        # 모델 성능 기록
        save_results(exp_name=self._exp_name, score=np.mean(test_corr),
                     task_id=task_id, mode=self._mode)


    def continual_learning(self):
        """학습하고자 하는 task의 개수만큼 학습 및 평가를 반복하는
        연속학습 함수입니다.
        """
        prev_task = 'task1'
        
        for i in range(self._num_tasks):
            # 현재 학습 중인 task 번호 설정(필수)
            print(f"current task: task{i + 1}")
            set_model_task(self._model, task=i)

            # 학습 데이터 준비
            self._mode = 'train'
            self._in_file_trn_dialog = self._in_file_trn_dialog.replace(prev_task, f'task{i + 1}')
            self.prepare_data()           
            
            self.train() # 학습
            
            # 평가 데이터 준비
            self._mode = 'eval'
            self._in_file_tst_dialog = self._in_file_tst_dialog.replace(prev_task, f'task{i + 1}')
            self.prepare_data()

            self.test(task_id=i + 1) # 평가
                
            prev_task = f'task{i + 1}'

        # pred 과정에서 binary masks만 사용하길 원한다면
        if self._pred_option == 'masks':
            # 1. binary masks를 buffer에 저장
            cache_masks(self._model)
            print()

            # 2. 추론에 불필요한 scores 삭제
            rm_scores(self._model)
            print()

        # 최종 weight 파일 저장(self._pred_option 따라 scores만 저장하거나 masks만 저장하게 됨)
        file_name_final = os.path.join(self._model_path, f'{self._exp_name}.pt')
        torch.save({'model': self._model.state_dict()}, file_name_final)


    def pred(self, task_id: int = None) -> np.array:
        """평가용 데이터셋에 대한 모델의 예측 결과를 구합니다.
        sub_task3 리더보드 제출 시 사용합니다.

        Args:
            task_id (int, optional): 추론 시 사용할 task id입니다. Defaults to None.

        Returns:
            np.array: 모델이 예측한 순위입니다.
        """
        print('\n<Predict>')

        # 학습한 모델 파일 불러오기
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)

                print('[*] load success: {}\n'.format(file_name))

            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        
        time_start = timeit.default_timer()

        # 데이터 준비
        self.prepare_data()

        # 추론하고자 하는 task 번호 등록(필수)
        set_model_task(self._model, task_id)

        # predict
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()

        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('-'*50)

        return preds.astype(int)
            