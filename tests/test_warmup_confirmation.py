import copy
import pytest


def test_matched_recipes_change_only_warmup():
    from scripts.warmup_confirmation import RECIPES
    a,b=RECIPES['short'],RECIPES['long']
    assert {k:v for k,v in a.items() if k!='warmup_steps'}=={k:v for k,v in b.items() if k!='warmup_steps'}
    assert a['steps']*a['batch_size']*a['accumulation']*512==1000013824
    assert a['seed']==20261001 and a['warmup_steps']==200 and b['warmup_steps']==1000


def report(correct=6,loss=3.):
    return {'partition':'confirmation','protocol_sha256':'same','trained_tokens':1000013824,
            'parameter_count':49295872,'checkpoint_sha256':'a'*64,
            'questions':[{'id':str(i),'correct':i<correct} for i in range(10)],
            'domains':{d:{'nll':loss} for d in ('fineweb','wiki','stories')}}


def test_confirmation_rejects_protocol_partition_budget_and_domain_mismatch():
    from scripts.warmup_confirmation import summarize_pair
    for key,value in [('partition','screen'),('protocol_sha256','other'),('trained_tokens',128008192),('parameter_count',10)]:
        bad={**report(),key:value}
        with pytest.raises(ValueError): summarize_pair(report(),bad,expected_questions=10)
    bad=copy.deepcopy(report());bad['domains']['wiki']['nll']=float('nan')
    with pytest.raises(ValueError): summarize_pair(report(),bad,expected_questions=10)
    with pytest.raises(ValueError): summarize_pair(report(),report())


def test_pair_requires_interval_gain_and_domain_preservation():
    from scripts.warmup_confirmation import summarize_pair
    assert not summarize_pair(report(),report(),expected_questions=10)['primary_gain_demonstrated']
    assert summarize_pair(report(0),report(10),expected_questions=10)['primary_gain_demonstrated']
    assert not summarize_pair(report(0),report(10,3.06),expected_questions=10)['primary_gain_demonstrated']


def test_both_models_train_before_confirmation_is_exposed():
    from scripts.warmup_confirmation import execute_pair
    events=[]
    def train(label,recipe):
        events.append(('train',label));return label+'-checkpoint'
    def evaluate(path,label):
        assert events[:2]==[('train','short'),('train','long')]
        events.append(('evaluate',label));return path
    result=execute_pair(train,evaluate)
    assert result=={'short':'short-checkpoint','long':'long-checkpoint'}
    assert events==[('train','short'),('train','long'),('evaluate','short'),('evaluate','long')]


def test_training_failure_never_exposes_confirmation():
    from scripts.warmup_confirmation import execute_pair
    evaluated=[]
    def train(label,recipe):
        if label=='long': raise RuntimeError('interrupted')
        return 'short-checkpoint'
    with pytest.raises(RuntimeError): execute_pair(train,lambda *args:evaluated.append(args))
    assert not evaluated


def test_pair_budget_rejects_replay_before_spending_on_first_run():
    from scripts.warmup_confirmation import require_pair_budget
    require_pair_budget({'short':0,'long':0},{'short':0,'long':0})
    require_pair_budget({'short':16384000,'long':0},{'short':16384000,'long':0})
    # The current short run alone could finish, but the matched pair cannot.
    with pytest.raises(ValueError,match='pair'):
        require_pair_budget({'short':16384000+16384,'long':0},{'short':16384000,'long':0})
    with pytest.raises(ValueError,match='accounting'):
        require_pair_budget({'short':0,'long':0},{'short':16384000,'long':0})
