import math
import pytest
from odin.development import repetition_rate, select_candidate, validate_development_report


def report(web, story, repetition=0.02):
    return {'domains':{'web':{'nll':web},'stories':{'nll':story}},
            'mean_repeated_4gram_rate':repetition,
            'protocol_sha256':'same-protocol','checkpoint_sha256':'a'*64}


def test_repetition_metric_catches_loops_without_penalizing_short_text():
    assert repetition_rate('one two three four five six') == 0
    assert repetition_rate('hello') == 0
    assert repetition_rate('I mean '*30) > 0.9


def test_selection_uses_common_domain_losses_with_web_guard():
    rows={'A':report(3.0,3.0),'B':report(3.09,2.0),'C':report(3.11,1.0)}
    result=select_candidate(rows, report(3.2,3.4))
    assert result['selected']=='B'
    assert 'web regression' in result['rejected']['C']


def test_selection_does_not_reward_repetition_collapse():
    rows={'A':report(3.0,3.0),'B':report(3.0,1.0,0.8),'C':report(3.0,2.0)}
    assert select_candidate(rows, report(3.2,3.4))['selected']=='C'
    for row in rows.values():
        row['mean_repeated_4gram_rate']=0.9
    assert select_candidate(rows, report(3.2,3.4))['selected'] is None


def test_selection_rejects_missing_nonfinite_and_mismatched_measurements():
    good={name:report(3,3) for name in ('A','B','C')}
    for bad in [{'A':report(3,3)}, {**good,'C':report(float('nan'),3)},
                {**good,'C':{**report(3,3),'protocol_sha256':'different'}}]:
        with pytest.raises(ValueError):
            select_candidate(bad, report(3,3))


def test_report_identity_rejects_stale_checkpoint_tokens_or_missing_samples():
    import copy,hashlib,json
    suite={'generation':['A beginning'],'seeds':[42], 'comparison':[]}
    protocol={'context':8,'windows_per_domain':2}
    r={**report(3,3,0),'protocol':protocol,'protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
       'trained_tokens':100,'generation':[{'prompt':'A beginning','seed':42,'text':'one two three four','repeated_4gram_rate':0}],
       'domains':{name:{'nll':3,'windows':2,'tokens':16} for name in ('web','stories')}}
    validate_development_report(r,'a'*64,100,protocol,suite)
    for key,value in [('trained_tokens',99),('checkpoint_sha256','b'*64),('generation',[]),('mean_repeated_4gram_rate',0.2)]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):
            validate_development_report(bad,'a'*64,100,protocol,suite)


def test_development_measurement_repeats_exactly_on_cpu(tmp_path):
    import json,torch
    from odin.data import encode_documents,sha256
    from odin.development import evaluate_development
    from odin.model import ModelConfig,LanguageModel
    from odin.tokenizer import Tokenizer
    tok=Tokenizer.train(['A short story about a cat. '*100],260)
    tok.save(tmp_path/'tokenizer.json')
    for file in ('dev.bin','stories-dev.bin'):
        encode_documents(['A short story about a cat. '*100],tok,tmp_path/file)
    config=ModelConfig(vocab_size=tok.vocab_size,width=16,layers=1,heads=2,ff=32,context=16)
    torch.manual_seed(9);model=LanguageModel(config)
    checkpoint=tmp_path/'test.pt'
    torch.save({'config':config.to_dict(),'model':model.state_dict(),'tokenizer_json':(tmp_path/'tokenizer.json').read_text(),
                'tokens':0,'fingerprint':{'tokenizer.json':sha256(tmp_path/'tokenizer.json')}},checkpoint)
    prompts=tmp_path/'prompts.json'
    prompts.write_text(json.dumps({'generation':['A cat'],'seeds':[42,137],'temperature':0.8,'max_new_tokens':6,
                                  'comparison':[{'prompt':'A','candidates':[' cat',' story'],'expected':0}]}))
    a=evaluate_development(checkpoint,tmp_path,prompts,windows=2)
    b=evaluate_development(checkpoint,tmp_path,prompts,windows=2)
    assert a['domains']==b['domains']
    assert a['comparison']==b['comparison']
    assert [r['text'] for r in a['generation']]==[r['text'] for r in b['generation']]
    validate_development_report(a,sha256(checkpoint),0,a['protocol'],json.loads(prompts.read_text()))
