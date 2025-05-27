
#!/usr/bin/env python
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate, argparse, tqdm

bleu = evaluate.load("bleu"); rouge = evaluate.load("rouge")

def main(args):
    ds = load_from_disk(args.dataset)["test"]
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, device_map="auto")

    preds, refs = [], []
    for ex in tqdm.tqdm(ds):
        prompt, ref = ex["text"].split("### Response:\n")
        out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=128)
        pred = tok.decode(out[0], skip_special_tokens=True).split("### Response:\n")[-1]
        preds.append(pred.strip()); refs.append(ref.strip())

    print("BLEU:", bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"])
    print("ROUGE-L:", rouge.compute(predictions=preds, references=refs)["rougeL"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../checkpoints/qlora_distilgpt2")
    p.add_argument("--dataset", default="../data/alpaca_prepared")
    main(p.parse_args())
