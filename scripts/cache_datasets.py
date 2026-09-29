"""Fetch pinned official public datasets for training and benchmark exclusions."""
import argparse
from datasets import load_dataset

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',default='.cache/datasets')
    args=parser.parse_args()
    sources=[
        ('Rowan/hellaswag',None,'218ec52e09a7e7462a5400043bb9a69a41d06b76'),
        ('allenai/ai2_arc','ARC-Easy','210d026faf9955653af8916fad021475a3f00453'),
        ('baber/piqa',None,'142f6d7367fd9877f0fb3b5734ea6a545f54cdd1'),
        ('allenai/winogrande','winogrande_xl','01e74176c63542e6b0bcb004dcdea22d94fb67b5'),
        ('Salesforce/wikitext','wikitext-103-raw-v1','b08601e04326c79dfdd32d625aee71d232d685c3'),
    ]
    for name,config,revision in sources:
        print('Caching',name,flush=True)
        load_dataset(name,config,revision=revision,cache_dir=args.cache_dir)
