
#!/usr/bin/env python
from datasets import Dataset
import json, random, argparse, os, pathlib

def main(args):
    raw = json.load(open(args.input, 'r', encoding='utf8'))
    random.seed(42)
    subs = random.sample(raw, args.size)

    def convert(ex):
        return {"text": f"### Instruction:\n{ex['instruction']}\n### Input:\n{ex.get('input','')}\n### Response:\n{ex['output']}"}

    ds = Dataset.from_list([convert(x) for x in subs])
    pathlib.Path(args.outdir).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="../data/alpaca_52k_small.json")
    p.add_argument("--outdir", default="../data/alpaca_prepared")
    p.add_argument("--size", type=int, default=1000)
    main(p.parse_args())
