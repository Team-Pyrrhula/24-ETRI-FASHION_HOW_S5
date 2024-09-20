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

Update: 2022.06.16.
'''
# built-in library
import re
import os
import json
from itertools import permutations
import codecs
import _pickle as pickle
from typing import List

# external library
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def _load_fashion_item(in_file: str, coordi_size: int, meta_size: int) -> tuple:
    """패션 아이템 메타데이터를 불러와 적절한 형태로 변환합니다.

    <패션 아이템 메타데이터의 구성 방식>
        | 패션 아이템의 이름 | 항목 | 패션 아이템의 종류 | 특징 종류 | 특징 기술 |

    <변환 결과>
        names: ['BL-001', 'BL-002', ...]
        metadata: ['단추 여밈 의 전체 오픈형 스탠드 칼라 와 ...', 
                    '면 100% 구김 이 가 기 쉬운...', ...]

    Args:
        in_file (str): 패션 아이템 메타데이터의 경로입니다.
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        meta_size (int): 메타데이터 특징의 개수입니다. 기본값은 4입니다. (형태/소재/색채/감성)

    Returns:
        tuple: 패션 아이템의 이름을 담은 리스트, 아이템별 특징 기술을 담은 리스트를 반환
    """
    print('loading fashion item metadata')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        names = [] # 패션 아이템의 이름을 저장하는 리스트
        metadata = [] # 패션 아이템의 특징 기술을 저장하는 리스트

        prev_name = ''
        prev_feat = ''

        data = ''

        for l in fin.readlines():
            line = l.strip()
            w = line.split()

            name = w[0].replace('L_', '') # 패션 아이템의 이름 슬라이싱 + aif.submit 오류 방지
            if name != prev_name: # 새로운 패션 아이템이 등장하면 추가
                names.append(name)
                prev_name = name

            feat = w[3] # 형태 특징 정보 슬라이싱 (e.g., F/M/C/E)
            if feat != prev_feat: # 새로운 형태 특징 정보가 등장하면, 이전까지 누적한 특징 기술을 추가
                if prev_feat != '': # 
                    metadata.append(data)

                data = w[4] # 특징 기술
                for d in w[5:]: # 공백을 두고 누적
                    data += ' ' + d

                prev_feat = feat

            else:
                for d in w[4:]: # 이전 형태 특징 정보와 동일하다면, 공백을 두고 누적하는 과정을 반복
                    data += ' ' + d

        metadata.append(data) # 추가되지 않은 마지막 특징 기술을 추가

        # 겉옷/상의/하의/신발에 속하지 않는 경우를 처리
        for _ in range(coordi_size*meta_size):
            metadata.append('')

        names.append('NONE-OUTER')
        names.append('NONE-TOP')
        names.append('NONE-BOTTOM')
        names.append('NONE-SHOES')

    return names, metadata


def _position_of_fashion_item(item: str) -> int:
    """패션 아이템의 이름으로 카테고리를 검색한 뒤, 해당 카테고리를 의미하는 인덱스를 반환합니다.

    Args:
        item (str): 패션 아이템의 이름입니다.

    Raises:
        ValueError: 사전에 정의한 이름 외 값이 입력되는 경우 발생합니다.

    Returns:
        int: 패션 아이템의 카테고리를 나타내는 인덱스입니다. (0: 겉옷, 1: 상의, 2: 하의, 3: 신발)
    """
    prefix = item[0:2]

    if prefix in ['JK', 'JP', 'CT', 'CD', 'VT'] or item == 'NONE-OUTER': # 겉옷
        idx = 0
    elif prefix in ['KN', 'SW', 'SH', 'BL'] or item == 'NONE-TOP': # 상의
        idx = 1
    elif prefix in ['SK', 'PT', 'OP'] or item == 'NONE-BOTTOM': # 하의
        idx = 2
    elif prefix == 'SE' or item == 'NONE-SHOES': # 신발
        idx=3
    else:
        raise ValueError('{} do not exists.'.format(item))
    
    return idx


def _insert_into_fashion_coordi(coordi: List[str], items: List[str]) -> List[str]:
    """coordi에 items을 삽입합니다.

    Args:
        coordi (List[str]):
            - 코디 봇이 추천한 패션 코디 조합입니다.
            - ex) ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']
        items (List[str]): 패션 아이템의 이름을 담은 리스트입니다.

    Returns:
        List[str]: items을 삽입한 coordi를 반환합니다.
    """
    new_coordi = coordi[:]

    for item in items:
        # 패션 아이템 이름 전처리
        item = item.split(';')
        new_item = item[len(item)-1].split('_')
        cl_new_item = new_item[len(new_item)-1]

        # item이 속하는 패션 카테고리의 인덱스를 구함
        pos = _position_of_fashion_item(cl_new_item)

        # 만약 OP(원피스)라면, TOP(상의)에 존재하는 아이템을 삭제
        if cl_new_item[0:2]=='OP':
            new_coordi[1] = 'NONE-TOP'

        # coordi의 해당 카테고리 위치에 item을 삽입
        new_coordi[pos] = cl_new_item

    return new_coordi


def _load_trn_dialog(in_file: str) -> tuple:
    """학습 대화문 데이터를 불러와 전처리한 뒤 반환합니다.

    [과정 요약]
    - 대화문: 대화 주체별로 발화문을 누적하여 저장
    - 코디 조합: 대화 주체가 코디봇에서 다른 주체로 바뀔 때마다 코디 조합을 저장
    - 대화문 태그: 발화문에 달린 태그 중 마지막 태그만을 저장

    Args:
        in_file (str): 학습 대화문 데이터의 경로입니다.

    Returns:
        tuple: 
            - data_utter: 대화문을 저장한 리스트
            - data_coordi: 코디 조합을 저장한 리스트
            - data_reward_last: 대화문의 성격을 알려주는 태그를 저장한 리스트
            - delim_utter, delim_coordi, delim_reward: 에피소드별로 구분하기 위한 인덱스 정보를 저장한 리스트
    """
    print('loading dialog DB')

    with open(in_file, encoding='euc-kr', mode='r') as fin:
        # 전처리가 끝난 대화문/코디 조합/대화문 태그를 저장
        data_utter = []
        data_coordi = []
        data_reward = []
        
        # data_***를 다른 함수에서 에피소드별로 구분하기 위해 인덱스 정보를 저장
        delim_utter = []
        delim_coordi = []
        delim_reward = []

        # 구분자(delimiter)로 사용되는 인덱스 정보
        num_turn = 1 # 발화문 인덱스
        num_coordi = 0 # 코디 인덱스
        num_reward = 0 # 태그 인덱스

        # 에피소드 개수를 기록
        num_dialog = 0

        is_first = True
        for l in fin.readlines(): # 전체 대화문을 읽어서
            line = l.strip() # 전처리 후
            w = line.split() # 공백을 기준으로 나눔

            ID = w[1] # 대화 주체(<AC>: 추천된 패션 코디, <CO>: 코디봇, <US>: 사용자)
            
            # 새로운 에피소드의 첫 번째 발화문일 때 (w[0]: 발화번호)
            if w[0] == '0':
                if is_first:
                    is_first = False
                
                else:
                    # 에피소드가 새로 시작했으므로, tot_utter에 남아있는 발화문을 저장
                    data_utter.append(tot_utter.strip())

                    # 코디봇이었다면, 코디를 추가
                    if prev_ID == '<CO>':
                        data_coordi.append(coordi)
                        num_coordi += 1

                    # 사용자였다면, 태그를 추가
                    if prev_ID == '<US>':
                        data_reward.append(tot_func.strip())
                        num_reward += 1

                    # 에피소드간 구분을 위해, 구분자 역할을 하는 변수들을 저장
                    delim_utter.append(num_turn)
                    delim_coordi.append(num_coordi)
                    delim_reward.append(num_reward)

                    num_turn += 1

                prev_ID = ID # 대화 주체를 기록

                # 대화 주체의 발화문과 태그를 기록할 변수 초기화
                tot_utter = ''
                tot_func = ''

                # 코디 조합을 초기화
                coordi = ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']
                num_dialog += 1 # 대화문 개수 증가
            
            # 대화 주체가 <추천된 패션 코디>라면
            if ID == '<AC>':
                # 기존 코디 조합에 해당 아이템들을 추가
                items = w[2:]
                coordi = _insert_into_fashion_coordi(coordi, items)
                # utter = '' # deprecated
                continue
            
            # 태그 전처리
            func = re.sub(pattern='[^A-Z_;]', repl='', string=w[-1])
            func = re.sub(pattern='[;]', repl=' ', string=func)

            # 전처리한 태그를 대상으로 예외 처리
            if func == '_': func = ''

            # 전처리한 태그가 존재한다면, 태그를 제외한 나머지 정보를 w에 저장
            if func != '': w = w[:-1]

            # 새로운 대화 주체가 등장했다면
            if prev_ID != ID:
                # 이전 대화 주체의 발화문을 저장
                data_utter.append(tot_utter.strip())

                # 코디봇이었다면, 코디를 저장
                if prev_ID == '<CO>':
                    data_coordi.append(coordi)
                    num_coordi += 1

                # 사용자였다면, 태그를 저장
                if prev_ID == '<US>':
                    data_reward.append(tot_func.strip())
                    num_reward += 1

                # 이전 대화 주체의 정보들을 저장했으므로, 변수들을 초기화
                tot_utter = ''
                tot_func = ''
                prev_ID = ID

                num_turn += 1

            # 대화 주체가 똑같다면, 발화문과 태그를 누적
            for u in w[2:]:
                tot_utter += ' ' + u 

            tot_func += ' ' + func

        ### 맨 마지막 에피소드를 저장하는 구간 ### (마지막 에피소드는 반복문으로 처리가 안되서 그럼)
        data_utter.append(tot_utter.strip())                  
        delim_utter.append(num_turn)

        if prev_ID == '<CO>':
            data_coordi.append(coordi)
            num_coordi += 1
            
        if prev_ID == '<US>':
            data_reward.append(tot_func.strip())
            num_reward += 1

        delim_coordi.append(num_coordi)
        delim_reward.append(num_reward)
        ### 맨 마지막 에피소드를 저장하는 구간 ###

        print('# of dialog: {} sets'.format(num_dialog))

        # 하나의 발화문에 여러 개의 태그가 존재할 수 있는데, 그 중 마지막 태그만 사용함
        data_reward_last = []
        for r in data_reward:
            r = r.split()
            if len(r) >= 1:
                data_reward_last.append(r[len(r)-1])    

            else:
                data_reward_last.append('')

        return data_utter, data_coordi, data_reward_last, \
               np.array(delim_utter, dtype='int32'), \
               np.array(delim_coordi, dtype='int32'), \
               np.array(delim_reward, dtype='int32')


def _load_eval_dialog(in_file: str, num_rank: int) -> tuple:
    """평가 대화문 데이터를 불러와 전처리한 뒤 반환합니다.

    Args:
        in_file (str): 평가 대화문 데이터의 경로입니다.
        num_rank (int): num_rank (int): 순위를 평가할 패션 코디 조합의 개수입니다. 기본값은 3입니다.

    Returns:
        tuple: 아래 데이터들은 모두 에피소드 단위로 구분되어 있습니다.
            - data_utter: 대화문을 저장한 리스트
            - data_coordi: 코디 조합을 저장한 리스트
            - data_rank: 코디 조합의 순위를 저장한 리스트로,
                        평가 데이터에서 코디 봇이 추천한 코디 조합들 모두
                        순위대로 잘 추천했다고 가정하고 0으로 저장함.
    """
    print('loading dialog DB')

    with open(in_file, encoding='euc-kr', mode='r') as fin:
        # 전처리가 끝난 대화문/코디 조합을 저장
        data_utter = []
        data_coordi = []

        # 에피소드/발화문 개수 확인용 변수
        num_dialog = 0
        num_utter = 0

        is_first = True

        for line in fin.readlines(): # 한 줄씩 읽어와서
            line = line.strip() # 줄바꿈 문자를 삭제
            
            # 평가 데이터에서 세미 콜론은 새 에피소드의 시작을 의미함
            # 따라서 새 에피소드의 시작이라면, 경우에 따라 전처리 수행
            if line[0] == ';':
                # 평가 데이터 전체의 끝이라면, 종료
                if line[2:5] == 'end': break

                # 평가 데이터 전체의 시작이라면, 변수 초기화
                if is_first: is_first = False

                # 이전 에피소드의 대화문과 코디 조합을 저장
                else:
                    data_utter.append(tot_utter)
                    data_coordi.append(tot_coordi)

                # 대화 주체의 발화문과 코디 조합을 기록할 변수 초기화
                tot_utter = []
                tot_coordi = []

                num_dialog += 1 # 대화문 개수 누적

            # 대화 주체가 사용자 또는 코디 봇이라면
            elif line[0:2] == 'US' or line[0:2] == 'CO':
                # 발화문을 저장하고
                utter = line[2:].strip()
                tot_utter.append(utter)

                num_utter += 1 # 발화문 개수를 증가

            # 추천한 코디 조합이라면
            elif line[0] == 'R':
                # 코디 조합을 저장
                coordi = line[2:].strip()
                new_coordi = ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']

                new_coordi = _insert_into_fashion_coordi(new_coordi, coordi.split())
                tot_coordi.append(new_coordi)
        
        # 맨 마지막 에피소드의 발화문과 코디 조합을 저장
        if not is_first:
            data_utter.append(tot_utter)
            data_coordi.append(tot_coordi)

        # 저장한 모든 코디 조합에 대해, rank 0을 부여함
        data_rank = []
        rank = 0    
        for _ in range(len(data_coordi)):
            data_rank.append(rank)

        print('# of dialog: {} sets'.format(num_dialog))
        
        return data_utter, data_coordi, data_rank
        

class SubWordEmbReaderUtil:
    """
    subword embedding을 위해 사용하는 객체입니다.
    """
    def __init__(self, data_path: str) -> None:
        """data_path에 저장되어있는 subword embedding 데이터를 불러옵니다.

        Args:
            data_path (str): subword embedding 데이터의 경로입니다.
        """
        print('\n<Initialize subword embedding>')
        print('loading=', data_path)
        with open(data_path, 'rb') as fp:
            self._subw_length_min = pickle.load(fp) # 2
            self._subw_length_max = pickle.load(fp) # 4
            self._subw_dic = pickle.load(fp, encoding='euc-kr') # subword dictionary
            self._emb_np = pickle.load(fp, encoding='bytes') # embedding vectors
            self._emb_size = self._emb_np.shape[1] # 128 dim

    def get_emb_size(self) -> None:
        """subword embedding vecotr의 dimension을 반환합니다. (default: 128)
        """
        return self._emb_size        

    def _normalize_func(self, s: str) -> str:
        """정해진 규칙에 따라 입력 단어를 정규화합니다.

        [규칙]
        - 공백은 삭제하고, 줄바꿈 문자는 e로 대체합니다.
        - 음절 단위로 분리한 뒤 euc-kr 코덱으로 인코딩했을 때, 특정 범위 내에 있다면 h로 대체합니다.
            - euc-kr에서 b'\xca\xa1'과 b'\xfd\xfe'는 한자의 시작과 끝을 의미합니다.)
            - Reference: https://i18nl10n.com/korean/euckr.html)
        
        note: e와 h로 대체하는 이유에 대해선 모름

        Args:
            s (str): 특징 기술 문장을 구성하는 단어입니다.

        Returns:
            str: 정규화된 단어입니다.
        """
        # 입력 단어에 대한 전처리 진행
        s1 = re.sub(' ', '', s)
        s1 = re.sub('\n', 'e', s1)
        sl = list(s1)

        # 예외처리 진행
        for a in range(len(sl)):
            if sl[a].encode('euc-kr') >= b'\xca\xa1' and \
               sl[a].encode('euc-kr') <= b'\xfd\xfe': sl[a] = 'h'
        
        s1 = ''.join(sl)

        return s1

    def _word2syllables(self, word: str) -> List[str]:
        """입력 단어를 음절 단위로 분리합니다.
        ex) 안녕하세요 -> 안, 녕, 하, 세, 요

        Args:
            word (str): 특징 기술 문장을 구성하는 단어입니다.

        Returns:
            List[str]: 음절 단위로 분리한 결과입니다.
        """
        syl_list = []

        # 입력 단어에 대한 정규화 진행
        dec = codecs.lookup('cp949').incrementaldecoder()
        w = self._normalize_func(dec.decode(word.encode('euc-kr')))

        # 정규화가 끝난 단어를 음절 단위로 분리
        for a in list(w):
            syl_list.append(a.encode('euc-kr').decode('euc-kr'))

        return syl_list

    def _get_cngram_syllable_wo_dic(self, word: str, min: int, max:int) -> List[str]:
        """주어진 단어를 음절 단위로 분해한 뒤, subword를 생성합니다.

        [과정]
        1. 음절 단위로 분리 후 시작과 끝을 알리는 문자 추가
            - '안녕하세요' -> ['<', '안', '녕', '하', '세', '요', '>']
        2. min, max를 범위로 서브워드 생성
            - ['<_안', '안_녕', '녕_하', ...] (범위 2)
            - ['<_안_녕', '안_녕_하', '녕_하_세', ...] (범위 3)

        Args:
            word (str): 특징 기술 문장을 구성하는 단어입니다.
            min (int): subword의 최소 길이입니다. 
            max (int): subword의 최대 길이입니다.

        Returns:
            List[str]: 서브워드 생성 결과입니다.
        """
        # 입력 단어를 음절 단위로 분리
        word = word.replace('_', '')
        p_syl_list = self._word2syllables(word.upper())

        subword = [] # 서브워드 저장을 위한 변수 선언
        syl_list = p_syl_list[:]

        # 단어의 시작과 끝을 알려주는 문자 추가
        syl_list.insert(0, '<')
        syl_list.append('>')

        # 서브워드 생성
        for a in range(len(syl_list)):
            for b in range(min, max+1):
                # 범위를 벗어나는 경우 종료
                if a+b > len(syl_list): break

                x = syl_list[a:a+b]
                k = '_'.join(x)
                subword.append(k)

        return subword

    def _get_word_emb(self, w: str) -> np.array:
        """특징 기술 문장을 구성하는 단어들에 대한 임베딩 벡터를 계산합니다.

        [과정]
        1. 입력 단어에 대한 서브워드를 생성합니다.
        2. 미리 만든 서브워드 사전에서, 새로 생성한 서브워드를 검색합니다.
            - 검색이 된다면, 서브워드의 고유 번호를 사용합니다.
            - 검색되지 않는다면, unknown subword의 고유 번호를 사용합니다.
        3. 미리 만든 전체 서브워드 임베딩 벡터에서, 고유 번호를 이용하여 특정 임베딩 벡터만 가져옵니다.
            - 가져온 서브워드 임베딩 벡터들을 모두 더합니다. (각각의 정보를 합쳐 단어에 대한 정보를 생성)

        Args:
            w (str): 특징 기술 문장을 구성하는 단어입니다.

        Returns:
            np.array: 단어 임베딩 벡터입니다. shape: (128, )
        """
        # 전처리
        word = w.strip()
        assert len(word) > 0

        # 서브워드 생성
        cng = self._get_cngram_syllable_wo_dic(word, self._subw_length_min, self._subw_length_max)

        # 서브워드 사전에서 새로 생성한 서브워드를 검색
        lswi = [self._subw_dic[subw] for subw in cng if subw in self._subw_dic]
        if lswi == []: lswi = [self._subw_dic['UNK_SUBWORD']] # TODO: unknown 처리가 좀 이상한 것 같은데, 추후에 변경해보기

        # 워드 임베딩 생성
        d = np.sum(np.take(self._emb_np, lswi, axis=0), axis=0)

        return d

    def get_sent_emb(self, s: str) -> np.array:
        """패션 아이템의 특징을 기술한 문장에 대한 임베딩 벡터를 계산합니다.

        [과정]
        1. 문장을 단어 단위로 분리한 뒤, 각 단어에 대한 임베딩 벡터를 구합니다.
        2. 이를 한데 모아 평균내는 방식(Reference?)으로 문장의 임베딩 벡터를 구합니다.

        *평균을 내는게 맞나? scaling처럼 고정된 값으로 나눠주는게 정보 변형을 막을 것 같은데..

        Args:
            s (str): 패션 아이템의 특징을 기술한 문장입니다.

        Returns:
            np.array: 문장에 대한 임베딩 벡터입니다. shape: (128, )
        """
        if s != '':
            s = s.strip().split()
            semb_tmp = []

            # 단어 단위로 임베딩한 뒤
            for a in s:
                semb_tmp.append(self._get_word_emb(a))

            # 이를 평균내어 문장 벡터를 구함
            avg = np.average(semb_tmp, axis=0)

        else:
            avg = np.zeros(self._emb_size)

        return avg


def _vectorize_sent(swer: SubWordEmbReaderUtil, sent: str) -> np.array:
    """문장을 벡터로 변환합니다.

    Args:
        swer (SubWordEmbReaderUtil): subword embedding 객체입니다.
        sent (str): 문장 데이터입니다. 문장의 종류는 아래와 같습니다.
            - 패션 아이템의 특징을 기술한 문장
            - 사용자와 코디 봇이 나눈 대화문

    Returns:
        np.array: 벡터 변환 결과입니다. shape: (128, )
    """
    vec_sent = swer.get_sent_emb(sent)

    return vec_sent 


def _vectorize_dlg(swer: SubWordEmbReaderUtil, dialog: List[str]) -> np.array:
    """dialog를 구성하는 문장들을 모두 벡터로 변환합니다.
    이 때 dialog는 패션 아이템의 특징을 기술한 문장 리스트 또는
    사용자와 코디 봇이 주고 받은 대화문 에피소드 리스트입니다.

    Args:
        swer (SubWordEmbReaderUtil): subword embedding 객체입니다.
        dialog (List[str]): 하나의 에피소드를 구성하는 문장들을 담은 리스트입니다. 문장의 종류는 아래와 같습니다.
            - 패션 아이템의 특징을 기술한 문장
            - 사용자와 코디 봇이 나눈 대화문

    Returns:
        np.array: 벡터 변환 결과입니다. shape: (총 문장 개수, 128)
    """
    vec_dlg = []
    for sent in dialog:
        sent_emb = _vectorize_sent(swer, sent)
        vec_dlg.append(sent_emb)

    vec_dlg = np.array(vec_dlg, dtype='float32')

    return vec_dlg


def _vectorize(swer:SubWordEmbReaderUtil, data: List[List[str]]) -> np.array:
    """에피소드 단위로 구분된 대화문 데이터를 벡터로 임베딩합니다.

    Args:
        swer (SubWordEmbReaderUtil): subword embedding 객체입니다.
        data (List[List[str]]): 에피소드 단위로 구분된 대화문 데이터입니다.

    Returns:
        np.array: (에피소드 단위의) 벡터로 변환된 데이터입니다.
    """
    print('vectorizing data')

    vec = []
    for dlg in data:
        dlg_emb = _vectorize_dlg(swer, dlg)
        vec.append(dlg_emb)

    vec = np.array(vec, dtype=object)

    return vec
    

def _memorize(dialog: np.array, mem_size: int, emb_size: int) -> np.array:
    """대화문 데이터(dialog) 중 기억하고자 하는 부분(mem_size)만큼 남깁니다.

    Args:
        dialog (np.array): 에피소드 단위로 구분된 대화문 벡터입니다.
        mem_size (int): MemN2N의 memory size입니다.
        emb_size (int): 임베딩 벡터의 차원 크기입니다.

    Returns:
        np.array: (에피소드 개수, mem_size, emb_size) 형태의 임베딩 벡터입니다.
    """
    print('memorizing data')
    
    zero_emb = np.zeros((1, emb_size))
    memory = []

    # 전체 에피소드 개수만큼 반복
    for i in range(len(dialog)):
        # 하나의 에피소드를 구성하는 문장 중에서
        # 앞쪽 문장 일부를 제외하여 mem_size 개수만큼 문장을 남김
        idx = max(0, len(dialog[i]) - mem_size)
        ss = dialog[i][idx:]

        # 만약 mem_size 개수만큼 문장을 확보하지 못한다면
        # 남은 개수만큼 padding을 진행
        pad = mem_size - len(ss)
        for i in range(pad):
            ss = np.append(ss, zero_emb, axis=0)

        # 동일한 크기의 임베딩 벡터를 저장
        memory.append(ss)

    return np.array(memory, dtype='float32')
    

def _make_ranking_examples(dialog: List[List[str]], coordi: List[List[List[str]]], reward: List[List[str]],
                           item2idx: List[dict], idx2item: List[dict], similarities: np.array,
                           num_rank: int, num_perm: int, num_aug: int, corr_thres: float) -> tuple:
    """학습에 사용할 순위 데이터를 생성합니다.

    [과정 요약]
    - 대화문: 코디 조합에 대응되는 대화문입니다.
    - 코디 조합: 코사인 유사도 기반의 증강을 적용하여, 학습 데이터보다 많은 코디 조합을 생성합니다.
    - 코디 조합의 순위: 생성한 코디 조합의 순서를 무작위로 섞어, 모델이 다양한 순위 데이터를 학습할 수 있도록 합니다.

    Args:
        dialog (List[List[str]]): (에피소드별로) 대화문을 저장한 리스트
        coordi (List[List[List[str]]]): (에피소드별로) 코디 조합을 저장한 리스트
        reward (List[List[str]]): (에피소드별로) 대화문의 성격을 알려주는 태그를 저장한 리스트
        item2idx (List[dict]): (카테고리별로) 패션 아이템의 이름으로 인덱스를 검색할 수 있는 사전입니다.
        idx2item (List[dict]): (카테고리별로) 인덱스로 패션 아이템의 이름을 검색할 수 있는 사전입니다.
        similarities (np.array): 카테고리별 아이템 간의 코사인 유사도입니다.
        num_rank (int): 순위를 평가할 패션 코디 조합의 개수입니다. 기본값은 3입니다.
        num_perm (int): 생성하고자 하는 학습 데이터의 개수입니다. 다시 말해 num_perm만큼 학습 데이터를 생성합니다.
        num_aug (int): 코디 조합 데이터에 대한 증강 횟수입니다. 다시 말해 num_aug만큼 코디 조합 데이터를 증강합니다.
        corr_thres (float): 코디 조합 데이터에 증강을 적용할 때 사용하는 임계값입니다.

    Returns:
        tuple: 모두 같은 shape을 가지며, 같은 index의 값은 서로 대응되는 관계입니다.
            - data_dialog: 에피소드별로 대화문을 저장한 리스트
            - data_coordi: 에피소드별로 코디 조합을 저장한 리스트 (증강이 적용되어 있음)
            - data_rank: 에피소드별로 코디 조합의 순위를 저장한 리스트 (증강이 적용되어 있음)
    """
    print('making ranking_examples')

    # 대화문/코디 조합/대화문 태그를 저장할 리스트
    data_dialog = []
    data_coordi = []
    data_rank = []

    # num_rank개의 코디 조합을 나열할 수 있는 모든 순열을 생성
    idx = np.arange(num_rank)
    rank_lst = np.array(list(permutations(idx, num_rank)))

    num_item_in_coordi = len(coordi[0][0]) # 4

    # 에피소드별로 데이터를 생성
    for i in range(len(coordi)):
        # 특정 에피소드의 코디 조합을 가져온 뒤, 이를 뒤집어서 저장(유의미한 코디 조합은 에피소드의 뒤에 위치하기 때문)
        crd_lst = coordi[i]
        crd_lst = crd_lst[::-1]
        
        crd = [] # 유의미한 코디 정보를 저장할 리스트

        # 데이터 전처리에 필요한 변수 선언
        prev_crd = ['', '', '', '']
        count = 0

        # 특정 에피소드의 코디 조합을 가져와서 확인
        for j in range(len(crd_lst)):
            # 유의미한 정보가 담겨 있으면서 새로운 코디 조합이라면, 이를 추가
            if crd_lst[j] != prev_crd and crd_lst[j] != \
                ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']:
                crd.append(crd_lst[j]) 
                prev_crd = crd_lst[j]
                count += 1

            # 필요한 개수만큼 데이터를 저장했다면 종료
            if count == num_rank:
                break
        
        # 특정 에피소드의 발화문 태그를 가져온 뒤, 이를 뒤집어서 저장
        rwd_lst = reward[i]
        rwd_lst = rwd_lst[::-1]
        
        rwd = '' # 데이터 전처리에 필요한 변수 선언

        # 유의미한 정보가 담긴 발화문 태그라면, 이를 저장하고 종료
        for j in range(len(rwd_lst)):
            if rwd_lst[j] != '':
                rwd = rwd_lst[j]
                break
        
        # 필요한 개수만큼 코디 조합을 확보했다면
        if count >= num_rank:
            # 이를 섞어 다양한 순위의 순열 데이터를 생성(num_perm 수만큼)
            for j in range(num_perm):
                rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd, num_rank) 

                data_rank.append(rank)
                data_dialog.append(dialog[i])
                data_coordi.append(rand_crd)

        # 필요한 개수만큼 코디 조합을 확보하지 못했다면
        elif count >= (num_rank - 1):
            # 코디 조합을 구성하는 4개의 카테고리(겉옷/상의/하의/신발) 중
            # 무작위로 2개의 카테고리를 뽑은 뒤
            itm_lst = list(permutations(np.arange(num_item_in_coordi), 2)) 
            idx = np.arange(len(itm_lst))
            np.random.shuffle(idx)

            # 해당 카테고리의 아이템을 새로운 아이템으로 바꾸는 데이터 증강을 적용하여
            # 필요한 개수(num_rank)만큼 코디 조합을 확보
            crd_new = _replace_item(crd[1], item2idx, idx2item, 
                            similarities, itm_lst[idx[0]], corr_thres)
            crd.append(crd_new)

            # 그리고 이를 섞어 다양한 순위의 순열 데이터를 생성
            for j in range(num_perm):
                rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd, num_rank)
                
                data_rank.append(rank)
                data_dialog.append(dialog[i])
                data_coordi.append(rand_crd)

        # 만약 코디봇이 코디 조합 추천에 성공했다면
        if 'USER_SUCCESS' in rwd:
            # 이를 활용하여 num_aug만큼 코디 조합 증강을 진행
            for m in range(num_aug):
                crd_aug = []
                crd_aug.append(crd[0]) # crd[0] == USER_SUCCESS에 대응되는 코디 조합

                for j in range(1, num_rank):
                    # 코디 조합을 구성하는 4개의 카테고리(겉옷/상의/하의/신발) 중
                    # 무작위로 num_rank개의 카테고리를 뽑은 뒤
                    itm_lst = list(permutations(np.arange(num_item_in_coordi), j)) 
                    idx = np.arange(len(itm_lst))
                    np.random.shuffle(idx)

                    # 해당 카테고리의 아이템을 새로운 아이템으로 바꾸는 데이터 증강을 적용하여
                    # 필요한 개수(num_rank)만큼 코디 조합을 확보
                    crd_new = _replace_item(crd[0], item2idx, idx2item, 
                                    similarities, itm_lst[idx[0]], corr_thres)
                    crd_aug.append(crd_new)

                # 그리고 이를 섞어 다양한 순위의 순열 데이터를 생성
                for j in range(num_perm):
                    rank, rand_crd = _shuffle_one_coordi_and_ranking(rank_lst, crd_aug, num_rank)

                    data_rank.append(rank)
                    data_dialog.append(dialog[i])
                    data_coordi.append(rand_crd)

    return data_dialog, data_coordi, data_rank


def _replace_item(crd: List[str], item2idx: List[dict], idx2item: List[dict],
                  similarities: np.array, pos: List[int], thres: float) -> List[str]:
    """기존 코디 조합(crd) 중 대상 카테고리(pos)의 아이템을 교체합니다.

    [과정 요약]
    1. 대상 카테고리 내의 아이템 인덱스들을 무작위로 섞음
    2. 기존 아이템의 인덱스로 코사인 유사도 값을 검색
    3. 그 중 임계값(thres) 미만의 유사도를 가지는 아이템이 등장한다면, 기존 아이템을 교체

    Args:
        crd (List[str]): 하나의 코디 조합입니다.
        item2idx (List[dict]): (카테고리별로) 패션 아이템의 이름으로 인덱스를 검색할 수 있는 사전입니다.
        idx2item (List[dict]): (카테고리별로) 인덱스로 패션 아이템의 이름을 검색할 수 있는 사전입니다.
        similarities (np.array): 카테고리별 아이템 간의 코사인 유사도입니다.
        pos (List[int]):
            - 교체하고자 하는 카테고리의 인덱스입니다.
            - 겉옷: 0, 상의: 1, 하의: 2, 신발: 3
        thres (float): _description_

    Returns:
        List[str]: 새로 만든 코디 조합입니다.
    """
    new_crd = crd[:]

    # 교체하고자 하는 대상 카테고리의 인덱스를 가져와서
    for p in pos:
        # 기존 조합의 대상 카테고리 아이템으로 인덱스를 검색
        itm = crd[p]
        itm_idx = item2idx[p][itm]    

        # 대상 카테고리 내에 존재하는 전체 인덱스를 무작위로 섞음
        idx = np.arange(len(item2idx[p]))
        np.random.shuffle(idx)

        # 대상 카테고리 내 아이템들 간에 계산한 코사인 유사도를 기반으로
        for k in range(len(item2idx[p])):
            # threshold 미만인 경우 교체를 진행
            if similarities[p][itm_idx][idx[k]] < thres:
                rep_idx = idx[k]
                rep_itm = idx2item[p][rep_idx]
                break

        new_crd[p] = rep_itm

    return new_crd


def _indexing_coordi(data: List[List[List[str]]], coordi_size: int, itm2idx: List[dict]) -> np.array:
    """패션 아이템의 이름을 인덱스로 변환합니다.

    [예시]
    - 변환 전: ['CD-220', 'BL-216', 'SK-287', 'SE-175']
    - 변환 후: [0, 23, 115, 7]

    Args:
        data (List[List[List[str]]]): 에피소드 단위로 구분된 코디 조합 리스트
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        itm2idx (List[dict]): 패션 아이템의 이름으로 인덱스를 검색할 수 있는 사전입니다.

    Returns:
        np.array: 인덱스로 변환한 코디 조합들이 저장되어있는 배열입니다.
    """
    print('indexing fashion coordi')
    vec = []

    # 전체 에피소드에 대해 반복
    for d in range(len(data)):
        vec_crd = []

        # d번째 에피소드를 구성하는 코디 조합들을 한 개씩 가져와서
        for itm in data[d]:
            # 패션 아이템에 미리 할당해 둔 인덱스를 저장.
            # ex) vec_crd = [[0, 255, 33, 6], [...], [...]]
            ss = np.array([itm2idx[j][itm[j]] for j in range(coordi_size)])
            vec_crd.append(ss)

        vec_crd = np.array(vec_crd, dtype='int32')
        vec.append(vec_crd)

    return np.array(vec, dtype='int32')


def _convert_one_coordi_to_metadata(one_coordi: np.array, coordi_size: int,
                                    metadata: np.array, img_feats = None) -> np.array:
    """하나의 코디 조합을 임베딩 벡터로 변환합니다.
    변환 과정에는 패션 아이템의 메타데이터(텍스트) 벡터를 사용하며,
    원하는 경우 패션 아이템의 사진(이미지) 피처 벡터를 추가로 사용할 수 있습니다.

    Args:
        one_coordi (np.array): 인덱스로 변환한 코디 조합 한 개입니다.
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        metadata (np.array): 카테고리별로 구분된 패션 아이템의 임베딩 벡터입니다.
        img_feats (_type_, optional): 패션 아이템의 이미지 feature입니다. Defaults to None.

    Returns:
        np.array: 임베딩 벡터로 변환한 코디 조합입니다.
    """
    # 메타데이터만 사용하는 경우
    if img_feats is None:
        items = None

        # 겉옷/상의/하의/신발 순으로 패션 아이템의 임베딩 벡터를 concat
        for j in range(coordi_size):
            buf = metadata[j][one_coordi[j]]
            
            if j == 0:
                items = buf[:]
            else:
                items = np.concatenate([items[:], buf[:]], axis=0)

    # 메타데이터와 이미지 피처를 함께 사용하는 경우
    else:
        items_meta = None
        items_feat = None

        # 겉옷/상의/하의/신발 순으로 진행
        for j in range(coordi_size):
            buf_meta = metadata[j][one_coordi[j]]
            buf_feat = img_feats[j][one_coordi[j]]

            if j == 0:
                items_meta = buf_meta[:]
                items_feat = buf_feat[:]

            # 메타데이터는 concat, 이미지 피처는 누적합
            else:
                items_meta = np.concatenate(
                                [items_meta[:], buf_meta[:]], axis=0)
                items_feat += buf_feat[:]

        # 이미지 피처에 대한 평균을 구한 뒤, 메타데이터와 concat
        items_feat /= (float)(coordi_size)
        items = np.concatenate([items_meta, items_feat], axis=0)

    return items 


def _convert_dlg_coordi_to_metadata(dlg_coordi: np.array, coordi_size: int,
                                    metadata: np.array, img_feats = None) -> np.array:
    """num_rank개의 코디 조합을 임베딩 벡터로 변환합니다.

    Args:
        dlg_coordi (np.array): (하나의 에피소드에 대해) 인덱스로 변환한 코디 조합들이 저장되어있는 배열입니다.
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        metadata (np.array): 카테고리별로 구분된 패션 아이템의 임베딩 벡터입니다.
        img_feats (_type_, optional): 패션 아이템의 이미지 feature입니다. Defaults to None.

    Returns:
        np.array: 임베딩 벡터로 변환한 num_rank개의 코디 조합입니다.
    """
    # num_rank개의 코디 조합 중 첫 번째 코디 조합을 임베딩 벡터로 변환
    items = _convert_one_coordi_to_metadata(dlg_coordi[0], 
                                coordi_size, metadata, img_feats)

    # 조건식 비교를 위해, 이전 값을 변수에 저장
    prev_coordi = dlg_coordi[0][:]
    prev_items = items[:]

    # concat을 위해 차원 하나를 추가
    scripts = np.expand_dims(items, axis=0)[:]

    # 남은 코디 조합에 대한 변환 수행
    for i in range(1, dlg_coordi.shape[0]):
        # 현재 코디 조합과 이전 코디 조합이 같은 경우, 이전 코디 조합의 임베딩 벡터를 복사
        if np.array_equal(prev_coordi, dlg_coordi[i]):
            items = prev_items[:]

        # 다르면 임베딩 벡터로 변환
        else:
            items = _convert_one_coordi_to_metadata(dlg_coordi[i], 
                                coordi_size, metadata, img_feats)
        
        # 변수 갱신
        prev_coordi = dlg_coordi[i][:]
        prev_items = items[:]

        # 이전 코디 조합의 임베딩 벡터와 concat
        items = np.expand_dims(items, axis=0)
        scripts = np.concatenate([scripts[:], items[:]], axis=0)
        
    return scripts


def _convert_coordi_to_metadata(coordi: np.array, coordi_size: int,
                                metadata: np.array, img_feats = None) -> np.array:
    """전체 에피소드의 코디 조합들을 모두 임베딩 벡터로 변환합니다.

    Args:
        coordi (np.array): 인덱스로 변환한 코디 조합들이 저장되어있는 배열입니다.
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        metadata (np.array): 카테고리별로 구분된 패션 아이템의 임베딩 벡터입니다.
        img_feats (_type_, optional): 패션 아이템의 이미지 feature입니다. Defaults to None.

    Returns:
        np.array: 임베딩 벡터로 변환한 코디 조합입니다.
    """
    print('converting fashion coordi to metadata')
    vec = []

    # 전체 에피소드에 대해
    for d in range(len(coordi)):
        # 특정 에피소드의 코디 조합들을 임베딩 벡터로 변환하여 저장
        vec_meta = _convert_dlg_coordi_to_metadata(coordi[d], 
                                coordi_size, metadata, img_feats)
        vec.append(vec_meta)

    return np.array(vec, dtype='float32')


def _episode_slice(data: List, delim: List) -> List:
    """data를 delim을 기준으로 에피소드 단위로 구분합니다.

    Args:
        data (List): 발화문/코디 조합/발화문 태그가 저장된 리스트
        delim (List): 에피소드별로 구분하기 위한 인덱스 정보를 저장한 리스트

    Returns:
        List: 에피소드 단위로 구분된 data
    """
    episodes = []
    start = 0

    for end in delim:
        epi = data[start:end]
        episodes.append(epi)
        start = end

    return episodes


def _categorize(name: List[str], vec_item: np.array, coordi_size: int) -> tuple:
    """패션 아이템의 이름과 임베딩 벡터를 미리 정의해 둔 카테고리에 맞게 분류합니다.

    Args:
        name (List[str]): 패션 아이템의 이름을 담은 리스트입니다.
        vec_item (np.array): 패션 아이템의 특징을 기술한 문장을 벡터로 변환한 결과입니다. shape: (패션 아이템 개수, 4*128)
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)

    Returns:
        tuple: 카테고리별로 분류된 패션 아이템 및 임베딩 벡터입니다.
    """
    slot_item = [] 
    slot_name = []

    for i in range(coordi_size):
        slot_item.append([])
        slot_name.append([])

    # 패션 아이템의 이름으로 카테고리를 확인한 뒤, 정해진 위치에 이름과 임베딩 벡터를 추가
    for i in range(len(name)):
        pos = _position_of_fashion_item(name[i])

        slot_item[pos].append(vec_item[i])
        slot_name[pos].append(name[i])

    slot_item = np.array([np.array(s) for s in slot_item], dtype=object)

    return slot_name, slot_item


def _shuffle_one_coordi_and_ranking(rank_lst:np. array, coordi: List[List[str]],
                                    num_rank: int) -> tuple:
    """코디 조합을 무작위로 섞어 rank를 부여합니다.

    Args:
        rank_lst (np.array):
            - num_rank로 만들 수 있는 순열입니다.
            - np.array([[0, 1, 2], -> rank 0
                        [0, 2, 1], -> rank 1
                        [1, 0, 2], -> rank 2
                        [1, 2, 0], -> rank 3
                        [2, 0, 1], -> rank 4
                        [2, 1, 0]])-> rank 5
            - rank가 낮을수록 우선순위(선호도)가 높은 순열을 의미합니다.
        coordi (List[List[str]]): 특정 에피소드의 코디 조합입니다.
        num_rank (int): 순위를 평가할 패션 코디 조합의 개수입니다.

    Returns:
        tuple:
            - rank: 순열의 순위입니다.
            - rand_crd: 순위에 맞게 재배치한 코디 조합입니다.
    """
    # [0, 1, 2]를 무작위로 섞음
    idx = np.arange(num_rank)
    np.random.shuffle(idx)

    # 미리 만들어둔 순열과 무작위로 섞은 순열을 비교하여 rank를 부여함
    for k in range(len(rank_lst)):
        if np.array_equal(idx, rank_lst[k]):
            rank = k
            break

    # 부여한 rank에 맞게 코디 조합 순서를 재배치
    rand_crd = []
    for k in range(num_rank):
        rand_crd.append(coordi[idx[k]])

    return rank, rand_crd


def shuffle_coordi_and_ranking(coordi: np.array, num_rank: int) -> tuple:
    """코디 조합을 무작위로 섞어 rank를 부여합니다.
    평가 데이터의 코디 조합에 사용합니다.

    Args:
        coordi (np.array): (평가 데이터의) 코디 조합입니다.
        num_rank (int): 순위를 평가할 패션 코디 조합의 개수입니다.

    Returns:
        tuple:
            - data_coordi_rand: 무작위로 섞은 코디 조합입니다.
            - data_rank: 코디 조합에 대응되는 순위 정보입니다.
    """
    # 순위 정보와 코디 조합을 저장할 배열 선언
    data_rank = []
    data_coordi_rand = []

    # 순위 순열 생성
    idx = np.arange(num_rank)
    rank_lst = np.array(list(permutations(idx, num_rank)))

    # 전체 코디 조합에 대해 반복
    for i in range(len(coordi)):
        # 순서를 shuffle
        idx = np.arange(num_rank)
        np.random.shuffle(idx)

        # shuffle한 순서의 순위 정보를 검색
        for k in range(len(rank_lst)):
            if np.array_equal(idx, rank_lst[k]):
                rank = k
                break

        # 순위 정보를 저장
        data_rank.append(rank)

        # 순위에 맞는 순서대로 기존 코디 조합을 재배치
        coordi_rand = []
        crd = coordi[i]
        for k in range(num_rank):
            coordi_rand.append(crd[idx[k]])

        # 재배치한 코디 조합을 저장
        data_coordi_rand.append(coordi_rand)

    # dtype 변환
    data_coordi_rand = np.array(data_coordi_rand, dtype='float32')    
    data_rank = np.array(data_rank, dtype='int32')

    return data_coordi_rand, data_rank

# deprecated
def _load_fashion_feature(file_name, slot_name, coordi_size, feat_size):
    """
    function: load image features
    """
    with open(file_name, 'r') as fin:
        data = json.load(fin)          
        suffix = '.jpg'
        feats = []
        for i in range(coordi_size):
            feat = []    
            for n in slot_name[i]:
                if n[0:4] == 'NONE':
                    feat.append(np.zeros((feat_size)))
                else:
                    img_name = n + suffix
                    feat.append(np.mean(np.array(data[img_name]), 
                                        axis=0))
            feats.append(np.array(feat))
        feats = np.array(feats, dtype=object)            
        return feats


def make_metadata(in_file_fashion: str, swer: SubWordEmbReaderUtil, coordi_size: int,
                  meta_size: int, use_multimodal: bool, in_file_img_feats: str, img_feat_size: int) -> tuple:
    """학습 및 평가 DB 생성 시 필요한 패션 아이템의 메타데이터를 적절한 형태로 전처리하는 함수입니다.

    Args:
        in_file_fashion (str): 패션 아이템 메타데이터의 경로입니다.
        swer (SubWordEmbReaderUtil): subword embedding 객체입니다.
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        meta_size (int): 메타데이터 특징의 개수입니다. 기본값은 4입니다. (형태/소재/색채/감성)
        use_multimodal (bool): 텍스트와 함께 이미지 데이터를 사용할지 그 여부를 나타냅니다.
        in_file_img_feats (str): 이미지 데이터의 경로입니다.
        img_feat_size (int): 이미지 데이터를 나타내는 피처의 크기입니다.

    Raises:
        ValueError: in_file_fashion 경로가 유효하지 않을 때 에러를 발생시킵니다.

    Returns:
        tuple: 아래 변수들은 모두 카테고리별로 분류되어 있습니다.
            - slot_item: 패션 아이템의 임베딩 벡터입니다.
            - idx2item: 인덱스로 패션 아이템의 이름을 검색할 수 있는 사전입니다.
            - item2idx: 패션 아이템의 이름으로 인덱스를 검색할 수 있는 사전입니다.
            - item_size: 카테고리별 아이템 개수입니다.
            - vec_similarities: 카테고리별 아이템 간의 코사인 유사도입니다.
            - slot_feat: 이미지 피처입니다. 기본값은 None입니다.
    """
    print('\n<Make metadata>')
    if not os.path.exists(in_file_fashion):
        raise ValueError('{} do not exists.'.format(in_file_fashion))

    ### 패션 아이템의 이름 및 특징 기술 문장을 불러옴 ###
    name, data_item = _load_fashion_item(in_file_fashion, 
                                         coordi_size, meta_size)
    
    ### 특징 기술 문장을 기반으로 패션 아이템의 임베딩 벡터를 생성 ###
    print('vectorizing data')
    emb_size = swer.get_emb_size()

    vec_item = _vectorize_dlg(swer, data_item) # embedding
    vec_item = vec_item.reshape((-1, meta_size*emb_size)) # 한 행이 하나의 아이템을 나타내도록 reshape
    
    # 패션 아이템들을 겉옷/상의/하의/신발 카테고리로 분류
    slot_name, slot_item = _categorize(name, vec_item, coordi_size)
    slot_feat = None
    if use_multimodal: # TODO: img feat 사용할 때 분석
        slot_feat = _load_fashion_feature(in_file_img_feats, 
                                    slot_name, coordi_size, img_feat_size)

    # 카테고리별 아이템 간에 코사인 유사도를 계산(추후 데이터 증강에 사용함)
    vec_similarities = []
    for i in range(coordi_size):
        item_sparse = sparse.csr_matrix(slot_item[i])
        similarities = cosine_similarity(item_sparse)
        vec_similarities.append(similarities)

    # [[겉옷 개수, 겉옷 개수], [상의 개수, 상의 개수], ... ] 형태를 가짐
    vec_similarities = np.array(vec_similarities, dtype=object)

    # 학습 및 평가 DB 구성에 활용할 변수를 생성
    idx2item = []
    item2idx = []
    item_size = []
    
    for i in range(coordi_size):
        # 카테고리별로 idx -> item, item -> idx 사전을 생성
        idx2item.append(dict((j, m) for j, m in enumerate(slot_name[i])))
        item2idx.append(dict((m, j) for j, m in enumerate(slot_name[i])))

        # 카테고리 내 아이템 개수를 기록
        item_size.append(len(slot_name[i]))
        
    return slot_item, idx2item, item2idx, item_size, \
           vec_similarities, slot_feat


def make_io_data(mode: str, in_file_dialog: str, swer: SubWordEmbReaderUtil,
                 mem_size: int, coordi_size: int, item2idx: List[dict], idx2item: List[dict], 
                 metadata: np.array, similarities: np.array, num_rank: int, 
                 num_perm: int = 1, num_aug: int = 1, corr_thres: float = 1.0, img_feats = None) -> tuple:
    """학습 및 평가 DB를 생성합니다.

    Args:
        mode (str): 생성할 데이터의 종류를 나타냅니다. prepare(학습), eval(평가) 모드가 존재합니다.
        in_file_dialog (str): 학습 대화문 데이터의 경로입니다.
        swer (SubWordEmbReaderUtil): subword embedding 객체입니다.
        mem_size (int): MemN2N의 memory size입니다.
        coordi_size (int): 하나의 패션 조합을 구성하는 아이템의 개수입니다. 기본값은 4입니다. (겉옷/상의/하의/신발)
        item2idx (List[dict]): (카테고리별로) 패션 아이템의 이름으로 인덱스를 검색할 수 있는 사전입니다.
        idx2item (List[dict]): (카테고리별로) 인덱스로 패션 아이템의 이름을 검색할 수 있는 사전입니다.
        metadata (np.array): (카테고리별로) 패션 아이템의 임베딩 벡터입니다.
        similarities (np.array): 카테고리별 아이템 간의 코사인 유사도입니다.
        num_rank (int): 순위를 평가할 패션 코디 조합의 개수입니다. 기본값은 3입니다.
        num_perm (int, optional): _description_. Defaults to 1.
        num_aug (int, optional): _description_. Defaults to 1.
        corr_thres (float, optional): _description_. Defaults to 1.0.
        img_feats (_type_, optional): 패션 아이템의 이미지 feature입니다. Defaults to None.

    Raises:
        ValueError: 학습 대화문 데이터의 경로가 잘못되었을 때 발생합니다.

    Returns:
        tuple: 아래 값들은 모두 에피소드 단위로 구분되어 있습니다.
            - mem_dialog: 임베딩 벡터로 변환한 대화문 데이터입니다.
            - vec_coordi: 임베딩 벡터로 변환한 코디 조합입니다.
            - data_rank: 코디 조합의 순위 정보입니다.
    """
    print('\n<Make input & output data>')
    if not os.path.exists(in_file_dialog):
        raise ValueError('{} do not exists.'.format(in_file_dialog))
    
    ### 학습용 데이터 생성 ###
    if mode == 'prepare':
        # 학습용 데이터(발화문, 코디 조합, 발화문 태그)를 불러옴
        dialog, coordi, reward, delim_dlg, delim_crd, delim_rwd = \
                                             _load_trn_dialog(in_file_dialog)
        
        # 에피소드 단위로 분리
        dialog = _episode_slice(dialog, delim_dlg)
        coordi = _episode_slice(coordi, delim_crd)
        reward = _episode_slice(reward, delim_rwd)

        # 에피소드 단위로 분리한 학습용 데이터를 기반으로
        # (대화문, 코디 조합, 코디 조합의 순위) 형식의 학습용 데이터 생성
        data_dialog, data_coordi, data_rank = \
                    _make_ranking_examples(dialog, coordi, reward, item2idx, 
                                           idx2item, similarities, num_rank, 
                                           num_perm, num_aug, corr_thres)

    ### 평가용 데이터 생성 ###                                           
    elif mode == 'eval':
        # (대화문, 코디 조합, 코디 조합의 순위) 형식의 평가용 데이터 생성
        data_dialog, data_coordi, data_rank = \
                                    _load_eval_dialog(in_file_dialog, num_rank)

    # 배열 형태로 변환(tensor 변환을 위해서)
    data_rank = np.array(data_rank, dtype='int32')

    # 대화문 데이터를 벡터로 임베딩
    vec_dialog = _vectorize(swer, data_dialog)
    emb_size = swer.get_emb_size()

    # MemN2N이 요구하는 벡터 크기로 변환
    mem_dialog = _memorize(vec_dialog, mem_size, emb_size)

    # 패션 아이템의 이름을 인덱스로 변환
    idx_coordi = _indexing_coordi(data_coordi, coordi_size, item2idx)

    # 패션 아이템을 임베딩 벡터로 변환
    vec_coordi = _convert_coordi_to_metadata(idx_coordi, coordi_size, metadata, img_feats)
    
    return mem_dialog, vec_coordi, data_rank
    
