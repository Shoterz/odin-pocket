import copy
import json
import numpy as np
import pytest
import torch


def test_source_split_retains_every_token_and_rejects_wrong_document_count(tmp_path):
    from odin.controlled_data import split_tokens
    np.array([1,2,0,3,0,4,5,0],dtype=np.uint16).tofile(tmp_path/'packed.bin')
    split_tokens(tmp_path/'packed.bin',tmp_path,{'fineweb':2,'wiki':1},0)
    assert np.fromfile(tmp_path/'fineweb.bin',dtype=np.uint16).tolist()==[1,2,0,3,0]
    assert np.fromfile(tmp_path/'wiki.bin',dtype=np.uint16).tolist()==[4,5,0]
    with pytest.raises(ValueError,match='document count'):
        split_tokens(tmp_path/'packed.bin',tmp_path,{'fineweb':1,'wiki':1},0)


def test_source_sampler_counts_tokens_without_changing_random_draws():
    from odin.controlled_train import CountedSampler
    from odin.sampling import TokenSampler
    sources=[(np.arange(100,dtype=np.uint16),.6),(np.arange(100,200,dtype=np.uint16),.4)]
    sampler=CountedSampler(sources,8,256)
    reference=TokenSampler(sources,8,256)
    g=torch.Generator().manual_seed(71)
    h=torch.Generator().manual_seed(71)
    x,y=sampler.batch(100,g)
    rx,ry=reference.batch(100,h)
    assert torch.equal(x,rx) and torch.equal(y,ry)
    assert sampler.source_tokens==[int((x[:,0]<100).sum())*8,int((x[:,0]>=100).sum())*8]


def test_explicit_warmup_and_decay_boundaries():
    from odin.controlled_train import learning_rate_factor
    assert learning_rate_factor(0,100,10)==.1
    assert learning_rate_factor(9,100,10)==1
    assert learning_rate_factor(10,100,10)==1
    assert learning_rate_factor(100,100,10)==pytest.approx(.1)
    with pytest.raises(ValueError):
        learning_rate_factor(0,100,100)


def report(web=3.,wiki=3.,story=2.,correct=6):
    return {'protocol_sha256':'same','partition':'screen','trained_tokens':128,
            'domains':{k:{'nll':v} for k,v in [('fineweb',web),('wiki',wiki),('stories',story)]},
            'questions':[{'id':str(i),'correct':i<correct} for i in range(10)]}


def test_selection_cannot_trade_general_regressions_for_story_loss():
    from odin.controlled_eval import select
    control=report()
    rows={'control':control,'stories':report(web=3.06,story=.5,correct=8),'better':report(web=2.99,correct=7)}
    result=select(rows,'control')
    assert result['selected']=='better'
    assert 'stories' in result['rejected']
    rows['better']=report(correct=5)
    assert select(rows,'control')['selected']=='control'


def test_selection_rejects_missing_questions_protocol_and_nonfinite_loss():
    from odin.controlled_eval import select
    for change in ('questions','protocol_sha256','domains','partition','trained_tokens'):
        bad=copy.deepcopy(report())
        if change=='questions': bad[change]=bad[change][:-1]
        elif change=='domains': bad[change]['wiki']['nll']=float('nan')
        elif change=='trained_tokens': bad[change]=256
        else: bad[change]='other'
        with pytest.raises(ValueError):
            select({'a':report(),'b':bad},'a')


def test_paired_interval_preserves_pairing():
    from odin.controlled_eval import paired_interval
    a=report()['questions']
    assert paired_interval(a,a)=={'difference':0.,'lower':0.,'upper':0.,'n':10}
    b=copy.deepcopy(a)
    b[-1]['correct']=True
    r=paired_interval(a,b)
    assert r['difference']==pytest.approx(.1) and r['lower']>=0
    b[-1]['id']='wrong'
    with pytest.raises(ValueError): paired_interval(a,b)


def test_controlled_training_exact_resume_counts_and_changed_warmup(tmp_path):
    from odin.controlled_train import train
    from odin.train import load_checkpoint
    from test_training import fixture_data
    data,cfg=fixture_data(tmp_path)
    args=dict(data=data,config=cfg,steps=8,batch_size=2,lr=.003,device='cpu',seed=7,eval_every=4,warmup_steps=2)
    full=train(output=tmp_path/'full',**args)
    train(output=tmp_path/'part',stop_after=4,**args)
    resumed=train(output=tmp_path/'part',resume=tmp_path/'part/latest.pt',**args)
    a,_,sa=load_checkpoint(full)
    b,_,sb=load_checkpoint(resumed)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert sa['source_tokens']==sb['source_tokens']=={'train.bin':256}
    with pytest.raises(ValueError,match='identical'):
        train(output=tmp_path/'bad',resume=resumed,**{**args,'warmup_steps':3})


def test_mixture_retains_explicit_weights_and_rejects_mutation(tmp_path):
    from odin.controlled_data import make_mixture
    from odin.sampling import training_sources
    common=tmp_path/'common'; common.mkdir()
    (common/'manifest.json').write_text('{}')
    for name in ('fineweb.bin','wiki.bin','stories.bin','dev.bin','tokenizer.json'):
        (common/name).write_text('fixture')
    weights={'fineweb':.5,'wiki':.4,'stories':.1}
    make_mixture(common,tmp_path/'view',weights)
    assert training_sources(tmp_path/'view')==[{'file':'fineweb.bin','weight':.5},{'file':'wiki.bin','weight':.4},{'file':'stories.bin','weight':.1}]
    with pytest.raises(ValueError): make_mixture(common,tmp_path/'bad',{'fineweb':.9,'wiki':.4,'stories':.1})
    with pytest.raises(ValueError): make_mixture(common,tmp_path/'view',{'fineweb':.6,'wiki':.4,'stories':0})


def test_mixture_recovers_partial_preparation_but_rejects_wrong_link(tmp_path):
    from odin.controlled_data import make_mixture
    common=tmp_path/'common';common.mkdir()
    (common/'manifest.json').write_text('{}')
    for name in ('fineweb.bin','wiki.bin','stories.bin','dev.bin','tokenizer.json'):
        (common/name).write_text('fixture')
    out=tmp_path/'view';out.mkdir()
    (out/'fineweb.bin').symlink_to(common/'fineweb.bin')
    weights={'fineweb':.6,'wiki':.4,'stories':0}
    make_mixture(common,out,weights)
    assert (out/'manifest.json').exists()
    wrong=tmp_path/'wrong';wrong.mkdir()
    (wrong/'fineweb.bin').symlink_to(common/'wiki.bin')
    with pytest.raises(ValueError,match='link'):
        make_mixture(common,wrong,weights)
    (out/'wiki.bin').unlink()
    (out/'wiki.bin').symlink_to(common/'stories.bin')
    with pytest.raises(ValueError,match='link'):
        make_mixture(common,out,weights)


def test_work_ledger_counts_replayed_steps_and_never_erases_attempts(tmp_path):
    from odin.controlled_train import reserve_step,charged_tokens
    path=tmp_path/'work.jsonl'
    with path.open('a',buffering=1) as f:
        reserve_step(f,'attempt-one',1,32)
        reserve_step(f,'attempt-one',2,32)
        reserve_step(f,'attempt-two',2,32)
    assert charged_tokens(path)==96
    with path.open('a') as f: f.write('{broken\n')
    with pytest.raises(ValueError,match='ledger'):
        charged_tokens(path)


def test_checkpoint_run_identity_rejects_bad_source_counts(tmp_path):
    from scripts.controlled_refinement import validate_state
    state={'recipe':{'steps':8,'batch_size':2,'accumulation':1},'config':{'context':16},
           'fingerprint':{'data':'same'},'provenance':{'source_sha256':{'code':'same'}},
           'step':4,'tokens':128,'source_tokens':{'train.bin':128}}
    validate_state(state,state['recipe'],state['config'],state['fingerprint'],{'code':'same'},['train.bin'])
    state['source_tokens']['train.bin']=127
    with pytest.raises(ValueError,match='source'):
        validate_state(state,state['recipe'],state['config'],state['fingerprint'],{'code':'same'},['train.bin'])


def test_evaluation_is_repeatable_and_rejects_incomplete_scores(tmp_path):
    from odin.controlled_eval import evaluate,validate_report,blind_review
    from odin.tokenizer import Tokenizer
    from odin.model import LanguageModel,ModelConfig
    from odin.data import encode_documents,sha256
    tok=Tokenizer.train(['The cup is on the table. '*100],260)
    tok.save(tmp_path/'tokenizer.json')
    for name in ('fineweb','wiki','stories'):
        encode_documents(['The cup is on the table. '*300],tok,tmp_path/f'{name}-dev.bin')
    (tmp_path/'manifest.json').write_text('{}')
    (tmp_path/'questions.json').write_text(json.dumps([{'id':'q1','partition':'screen','prompt':'The cup is','choices':[' on the table.',' blue.'],'expected':0}]))
    prompts=tmp_path/'prompts.json'
    prompts.write_text(json.dumps({'generation':['The cup'],'seeds':[42],'max_new_tokens':4,'temperature':.8}))
    config=ModelConfig(vocab_size=tok.vocab_size,width=16,layers=1,heads=2,ff=32,context=512)
    model=LanguageModel(config)
    checkpoint=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'config':config.to_dict(),'tokens':0,
                'fingerprint':{'tokenizer.json':sha256(tmp_path/'tokenizer.json')},'tokenizer_json':(tmp_path/'tokenizer.json').read_text()},checkpoint)
    a=evaluate(checkpoint,tmp_path,prompts,windows=2,generation=True)
    b=evaluate(checkpoint,tmp_path,prompts,windows=2,generation=True)
    assert a['questions']==b['questions'] and a['domains']==b['domains']
    assert a['generation'][0]['text']==b['generation'][0]['text']
    blind_review({'a':a,'b':b},tmp_path/'blind')
    assert len(json.loads((tmp_path/'blind/blind-review.json').read_text())['items'])==2
    b['questions']=[]
    with pytest.raises(ValueError,match='Incomplete'):
        validate_report(b,checkpoint,tmp_path,prompts,'screen','cpu',2,True)
