from odin.corpus import iter_stories, filter_document, verify_refinement_inputs
from odin.data import grams
import hashlib
import json
import pytest


def test_input_audit_rejects_missing_holdout_and_changed_source(tmp_path):
    from odin.data import sha256
    baseline=tmp_path/'baseline';baseline.mkdir()
    for name in ('dev.bin','dev.jsonl','tokenizer.json'):
        (baseline/name).write_text('fixture '+name)
    fineweb=tmp_path/'web.parquet';fineweb.write_bytes(b'web')
    wiki=tmp_path/'wiki.arrow';wiki.write_bytes(b'wiki')
    stories=tmp_path/'TinyStoriesV2-GPT4-train.txt';stories.write_bytes(b'stories')
    protected=[{'file':'heldout.arrow','sha256':'1'*64}]
    manifest={'files':{p.name:sha256(p) for p in baseline.iterdir()},'decontamination':{'protected_files':protected},
              'sources':[{'sha256':sha256(fineweb)},{'files':[{'file':wiki.name,'sha256':sha256(wiki)}]}]}
    (baseline/'manifest.json').write_text(json.dumps(manifest))
    args=(baseline,protected,[wiki],fineweb,stories)
    assert verify_refinement_inputs(*args,story_expected_sha=sha256(stories))['verified']
    with pytest.raises(ValueError,match='Protected'):
        verify_refinement_inputs(baseline,[],[wiki],fineweb,stories,story_expected_sha=sha256(stories))
    wiki.write_bytes(b'changed')
    with pytest.raises(ValueError,match='WikiText'):
        verify_refinement_inputs(*args,story_expected_sha=sha256(stories))


def test_story_separator_preserves_individual_documents(tmp_path):
    p = tmp_path/'stories.txt'
    p.write_text('First story.\n<|endoftext|>\nSecond story.\n<|endoftext|>\nLast story.')
    assert list(iter_stories(p)) == ['First story.', 'Second story.', 'Last story.']


def test_document_filter_excludes_holdout_ngrams_and_short_docs():
    text = ' '.join('word'+str(i) for i in range(50))
    assert filter_document(text, set(grams(text)))['reason'] == 'overlap'
    assert filter_document('too short', set())['reason'] == 'length'
    row = filter_document(text, set())
    assert row['sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert row['split'] in ('train','dev')
