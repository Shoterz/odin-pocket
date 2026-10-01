# Training data

FineWeb-Edu: HuggingFaceFW/fineweb-edu, sample-10BT, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, shard `sample/10BT/000_00000.parquet`, first 150,000 rows considered. License: Open Data Commons Attribution 1.0. Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu . Upstream corpus uses LLM-assisted educational quality filtering; our model receives only next-token targets from the published web text.

WikiText: Salesforce/wikitext, wikitext-103-raw-v1, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Only the training split enters preparation. License: CC-BY-SA-3.0 / GFDL, underlying Wikipedia attribution requirements apply. Source: https://huggingface.co/datasets/Salesforce/wikitext . Article boundaries are preserved before partitioning.

No pretrained weights, imported embeddings, teacher logits or model-generated private targets are used. The tokenizer is trained afresh on a deterministic reservoir sample of training documents.

Filtering: Unicode NFC and newline normalization; 200–200,000 characters per document; SHA-256 exact-document deduplication across sources; exclusion of normalized 13-word overlaps with benchmark validation and test text. All five required benchmark families must be present. Development partition is hash bucket 0 of 100; complete documents stay together. No validation/test data is used as next-token training targets or tokenizer-training input.

Limitations: a single FineWeb shard prefix is not globally representative. Exact hashes miss near duplicates. The 13-gram filter can miss distinctive short prompts, short answer options and paraphrased overlaps. Public web text can contain errors, bias and personal information; the dataset is not certified privacy-clean. Mixing packed documents permits attention across EOS boundaries, as in ordinary causal pretraining; this is documented rather than described as isolated-document attention.

Inspect the bundled [data manifest](../submission/evidence/data-manifest.json) (recreated as `data/prepared/manifest.json`) for the measured number of accepted and rejected documents, token totals, source hashes, tokenizer hash, exact benchmark files and split policy. Corpus redistribution must retain applicable licenses and attribution; generated caches are excluded from source control.
