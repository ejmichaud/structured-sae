import os
import argparse
import h5py
os.environ['HF_HOME'] = '/om/user/ericjm/.cache/huggingface'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

@torch.inference_mode()
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        output_hidden_states=True,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    model.eval()
    d_model = model.config.n_embd
    model_name = args.model.split('/')[-1] 
    activations_file = os.path.join(args.output_dir, f'{model_name}-layer{args.layer}.h5')
    
    dataset = load_dataset(args.dataset, split='train', streaming=True)
    
    with h5py.File(activations_file, 'w') as f:
        dset = f.create_dataset('activations', shape=(args.tokens, d_model), dtype=args.dtype)
        
        tokens = 0
        pbar = tqdm(total=args.tokens, desc="Processing tokens", unit="tok")
        for document in dataset:
            text = document['text']
            inputs = tokenizer(text, return_tensors='pt', max_length=args.max_context, truncation=True)
            inputs = inputs.to(args.device)
            outputs = model(**inputs)
            activations = outputs.hidden_states[args.layer] # (1, seq_len, d_model)
            
            seq_len = activations.shape[1]
            if tokens + seq_len > args.tokens:
                # If this batch would exceed the desired token count, only take what we need
                seq_len = args.tokens - tokens
                activations = activations[:, :seq_len, :]

            dset[tokens:tokens+seq_len] = activations.squeeze().cpu().numpy()
            tokens += seq_len
            pbar.update(seq_len)
            
            if tokens >= args.tokens:
                break

        pbar.close()

        # In case we didn't reach exactly args.tokens (e.g., if the dataset was exhausted)
        if tokens < args.tokens:
            print(f"\nWarning: Only processed {tokens} tokens. Dataset exhausted before reaching {args.tokens} tokens.")
            # Resize the dataset to match the actual number of tokens processed
            dset.resize((tokens, d_model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openai-community/gpt2', help='model name')
    parser.add_argument('--layer', type=int, default=7, help='block to save residual stream activations')
    parser.add_argument('--dataset', type=str, default='skylion007/openwebtext', help='dataset to run model on')
    parser.add_argument('--tokens', type=int, default=int(2e8), help='number of tokens to run model on')
    parser.add_argument('--output_dir', type=str, default='activations', help='output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run model on')
    parser.add_argument('--dtype', type=str, default='float32', help='data type to save activations as')
    parser.add_argument('--max_context', type=int, default=512, help='max context length')
    args = parser.parse_args()
    main(args)
