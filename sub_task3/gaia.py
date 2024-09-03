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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# built-in library
import os
import joblib
import shutil
import time
import timeit
import re
# from tqdm import tqdm
import argparse

# external library
import numpy as np
from scipy import stats

# custom modules
from file_io import *
from requirement import *
from policy import *
from si import *
from exp_utils import save_results


### global settings ###
NUM_ITEM_IN_COORDI = 4 # of items in fashion coordination      
NUM_META_FEAT = 4 # of metadata features    
NUM_RANKING = 3 # of fashion coordination candidates        
IMG_FEAT_SIZE = 4096 # image feature size 

# TODO: parameter 관리 방식 변경
### SI parameter ###
si_c = 0.1
epsilon = 0.001


class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, emb_size, key_size, mem_size, 
                 meta_size, hops, item_size, 
                 coordi_size, eval_node, num_rnk, 
                 use_batch_norm, use_dropout, zero_prob,
                 use_multimodal, img_feat_size):
        """
        initialize and declare variables
        """
        super().__init__()
        # class instance for requirement estimation
        self._requirement = RequirementNet(emb_size, key_size, 
                                    mem_size, meta_size, hops)
        # class instance for ranking
        self._policy = PolicyNet(emb_size, key_size, item_size, 
                                 meta_size, coordi_size, eval_node,
                                 num_rnk, use_batch_norm,
                                 use_dropout, zero_prob,
                                 use_multimodal, img_feat_size)

    def forward(self, dlg, crd):
        """
        build graph
        """
        req = self._requirement(dlg)
        logits = self._policy(req, crd)
        preds = torch.argmax(logits, 1)
        return logits, preds


class gAIa(object):
    """ Class for AI fashion coordinator """
    def __init__(self, args: argparse.Namespace, device: torch.device, name='gAIa'):
        """
        실험에 필요한 변수, 데이터, 모델 등을 생성하고 관리합니다.

        Args:
            args (argparse.Namespace): 터미널에서 입력된 실험 인자입니다.
            device (torch.device): 학습 및 추론 시 사용할 장치입니다.
            name (str, optional): _description_. Defaults to 'gAIa'. # deprecated
        """
        # global variables
        self._device = device
        self._mode = args.mode
        self._exp_name = args.exp_name
        self._task_ids = args.task_ids
        self._use_cl = args.use_cl

        # path variables
        self._model_path = args.model_path
        self._model_file = args.model_file
        self._in_file_trn_dialog = args.in_file_trn_dialog

        # exp variables
        self._batch_size = args.batch_size
        self._epochs = args.epochs
        self._max_grad_norm = args.max_grad_norm
        self._save_freq = args.save_freq
        self._num_eval = args.evaluation_iteration
        
        # PolicyNet variables
        use_dropout = args.use_dropout
        if args.mode == 'test':
            use_dropout = False
        
        # class instance for subword embedding
        self._swer = SubWordEmbReaderUtil(args.subWordEmb_path)
        self._emb_size = self._swer.get_emb_size() # 128
        self._meta_size = NUM_META_FEAT
        self._coordi_size = NUM_ITEM_IN_COORDI
        self._num_rnk = NUM_RANKING
        feats_size = IMG_FEAT_SIZE
        
        # read metadata DB
        self._metadata, self._idx2item, self._item2idx, \
            self._item_size, self._meta_similarities, \
            self._feats = make_metadata(args.in_file_fashion, self._swer, 
                                self._coordi_size, self._meta_size,
                                args.use_multimodal, args.in_file_img_feats,
                                feats_size)
        
        # prepare DB for training
        if args.mode == 'train':
            self._dlg, self._crd, self._rnk = make_io_data('prepare', 
                        args.in_file_trn_dialog, self._swer, args.mem_size,
                        self._coordi_size, self._item2idx, self._idx2item, 
                        self._metadata, self._meta_similarities, self._num_rnk,
                        args.permutation_iteration, args.num_augmentation, 
                        args.corr_thres, self._feats)
            self._num_examples = len(self._dlg)

            # dataloader
            dataset = TensorDataset(torch.tensor(self._dlg), 
                                    torch.tensor(self._crd), 
                                    torch.tensor(self._rnk, dtype=torch.long))
            self._dataloader = DataLoader(dataset, 
                                    batch_size=self._batch_size, shuffle=True)
            
        # prepare DB for evaluation
        elif args.mode in ['eval', 'test', 'pred']:
            self._tst_dlg, self._tst_crd, _ = make_io_data('eval', 
                    args.in_file_tst_dialog, self._swer, args.mem_size,
                    self._coordi_size, self._item2idx, self._idx2item, 
                    self._metadata, self._meta_similarities, self._num_rnk,
                    args.num_augmentation, args.num_augmentation, 
                    args.corr_thres, self._feats)
            self._num_examples = len(self._tst_dlg)
        
        # model
        self._model = Model(self._emb_size, args.key_size, args.mem_size, 
                            self._meta_size, args.hops, self._item_size, 
                            self._coordi_size, args.eval_node, self._num_rnk, 
                            args.use_batch_norm, use_dropout, 
                            args.zero_prob, args.use_multimodal, 
                            feats_size)
        
        print('\n<model parameters>')
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)
                n = name.replace('.', '__')
                self._model.register_buffer('{}_SI_prev_task'.format(n), 
                                            param.detach().clone())
                self._model.register_buffer('{}_SI_omega'.format(n), 
                                            torch.zeros(param.shape))

        if args.mode == 'train':
            # optimizer
            self._optimizer = optim.SGD(self._model.parameters(),
                                        lr=args.learning_rate)
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

    def train(self) -> bool:
        """모델 학습에 사용합니다.

        Returns:
            bool: 학습 결과를 나타냅니다.
        """
        print('\n<Train>')
        print('total examples in dataset: {}'.format(self._num_examples))

        # 모델 파일을 저장할 디렉토리 생성
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)
        
        init_epoch = 1

        # weight 파일이 존재한다면
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            
            # 기존 weight 파일을 불러옴
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, 
                                        map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                print('[*] load success: {}\n'.format(file_name))

                ################## deprecated ##################
                # # 학습이 완전히 종료된 weight 파일을 불러왔다면, backup을 생성
                # if self._model_file == 'gAIa-final.pt':
                #     print(f'time.strftime: {time.strftime("%m%d-%H%M%S")}')
                #     file_name_backup = os.path.join(self._model_path, 
                #         'gAIa-final-{}.pt'.format(time.strftime("%m%d-%H%M%S")))
                #     print(f'file_name_backup: {file_name_backup}')
                #     shutil.copy(file_name, file_name_backup)

                # # 학습 중이었던 weight 파일이라면, 학습이 종료된 에폭부터 재학습
                # else:
                #     init_epoch += int(re.findall('\d+', file_name)[-1])
                ################## deprecated ##################

            # weight file이 존재하지 않는다면, 프로세스를 종료
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        
        # cpu or gpu로 모델을 옮김
        self._model.to(self._device)

        # SI 알고리즘 계산을 위해서, 기존 파라미터 크기의 공간 마련 && 백업 수행
        W, p_old = {}, {}
        for n, p in self._model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        
        # training
        end_epoch = self._epochs + init_epoch
        for curr_epoch in range(init_epoch, end_epoch):
            time_start = timeit.default_timer()
            losses = []
            iter_bar = tqdm(self._dataloader)
            
            for batch in iter_bar:
                # optimizer에 누적된 gradients를 초기화
                self._optimizer.zero_grad()

                # batch를 구성하는 data를 device로 옮김
                batch = [t.to(self._device) for t in batch]
                
                # CL 적용 여부에 따라 loss를 계산
                loss_ce = self._get_loss(batch).mean()
                if self._use_cl == True:
                    loss_si = surrogate_loss(self._model)
                    loss = loss_ce + si_c*loss_si

                else:
                    loss = loss_ce

                # gradient 계산
                loss.backward()
                
                # 최적화 과정에서 발생할 수 있는 문제를 방지하기 위해, gradient clip을 적용
                nn.utils.clip_grad_norm_(self._model.parameters(), 
                                         self._max_grad_norm)

                # 파라미터 업데이트 및 loss 기록
                self._optimizer.step()
                losses.append(loss)

                # SI 알고리즘에 필요한 수치 계산 및 저장
                for n, p in self._model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad*(p.detach()-p_old[n]))

                        p_old[n] = p.detach().clone()

            time_end = timeit.default_timer()
            
            # Epoch별 학습 상태 출력
            print('-'*50)
            print('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            print('Time: {:.2f}sec'.format(time_end - time_start))
            print('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            print('-'*50)

            # 지정한 주기마다 모델 파일 저장
            if curr_epoch % self._save_freq == 0:
                file_name = os.path.join(self._model_path, 
                                         'gAIa-{}.pt'.format(curr_epoch))
                torch.save({'model': self._model.state_dict()}, file_name)
                
        print('Done training; epoch limit {} reached.\n'.format(self._epochs))
        
        # SI 알고리즘에 필요한 omega 계산
        update_omega(self._model, self._device, W, epsilon)
        
        # 최종 weight 파일 저장
        # TODO: exp_name 기반으로 weight 파일명 저장
        file_name_final = os.path.join(self._model_path, 'gAIa-final.pt')
        torch.save({'model': self._model.state_dict()}, file_name_final)

        return True
        
    def _calculate_weighted_kendall_tau(self, pred, label, rnk_lst):
        """
        calcuate Weighted Kendall Tau Correlation
        """
        total_count = 0
        total_corr = 0
        for p, l in zip(pred, label):
            corr, _ = stats.weightedtau(self._num_rnk-1-rnk_lst[l], #
                                        self._num_rnk-1-rnk_lst[p]) #
            total_corr += corr
            total_count += 1
        return (total_corr / total_count)
    
    def _predict(self, eval_dlg, eval_crd):
        """
        predict
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
    
    # TODO: 코드 분석
    def pred(self):
        """
        create prediction.csv
        """
        print('\n<Predict>')

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
        # predict
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('-'*50)
        return preds.astype(int)

    def test(self) -> np.array:
        """모델 성능을 평가하고 기록합니다.
        """
        print('\n<Evaluate>')

        # 테스트에 사용할 weight 파일을 불러옴
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)

            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, 
                                        map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                self._model.to(self._device)
                print('[*] load success: {}\n'.format(file_name))

            else:
                print('[!] checkpoints path does not exist...\n')
                return False
            
        else:
            return False
        
        time_start = timeit.default_timer()
        
        # 모델 성능 평가
        repeated_preds, test_corr, num_examples = self._evaluate(self._tst_dlg, 
                                                                 self._tst_crd)
        time_end = timeit.default_timer()

        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('Average WKTC over iterations: {:.4f}'.format(np.mean(test_corr)))
        print('Best WKTC: {:.4f}'.format(np.max(test_corr)))
        print('-'*50)

        # 모델 성능 기록
        _, tr_task_id, eval_task_id = self._task_ids.split('/')
        save_results(exp_name=self._exp_name, score=np.mean(test_corr),
                     tr_task_id=tr_task_id, tt_task_id=eval_task_id,
                     mode=self._mode)


# Test Code
if __name__ == "__main__":
    model = Model(emb_size=128, key_size=300, mem_size=16,
                  meta_size=4, hops=3, item_size=1,
                  coordi_size=4, eval_node='[6000,6000,200][2000,1000,500]',
                  num_rnk=3, use_batch_norm=False, use_dropout=True,
                  zero_prob=0.5, use_multimodal=False, img_feat_size=4096)
    
    print(f"number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")