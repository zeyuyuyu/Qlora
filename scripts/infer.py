
#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, textwrap

def gen(prompt, model, tok, max_new=128):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, top_p=0.9, temperature=0.7)
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../checkpoints/qlora_distilgpt2")
    p.add_argument("--prompt", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, device_map="auto")
    print(textwrap.fill(gen(args.prompt, model, tok), width=100))
