# custom modules
from supermask.linear import MultitaskMaskLinear


def set_model_task(model, task, verbose=True):
    """task 값을 model 객체의 변수로 등록
    """
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            if verbose:
                print(f"=> Set task of {n} to {task}")
                
            m.task = task
            
def cache_masks(model):
    """supermask를 등록
    """
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            print(f"=> Caching mask state for {n}")
            m.cache_masks()

def rm_scores(model):
    """binary masks만 이용하여 예측하고자 할 때
    불필요한 scores parameters를 삭제
    """
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            print(f"=> Delete scores for {n}")            
            m.rm_scores()
            
def set_num_tasks_learned(model, num_tasks_learned):
    """현재까지 학습한 task의 개수를 model 객체의 변수로 등록
    """
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            print(f"=> Setting learned tasks of {n} to {num_tasks_learned}")
            m.num_tasks_learned = num_tasks_learned
            
def set_alphas(model, alphas, verbose=True):
    """inference 시 사용할 supermask의 alpha 값을 등록
    """
    for n, m in model.named_modules():
        if isinstance(m, MultitaskMaskLinear):
            if verbose:
                print(f"=> Setting alphas for {n}")
                
            m.alphas = alphas