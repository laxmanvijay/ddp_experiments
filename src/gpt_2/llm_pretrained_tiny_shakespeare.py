import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import sys
import argparse

sys.stdout.reconfigure(line_buffering=True)

def setup_ddp():
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(backend='gloo', init_method='env://')
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size} initialized")
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def get_shakespeare_dataloader(rank, world_size, batch_size, block_size=128):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token normally
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset('tiny_shakespeare', split='train', trust_remote_code=True)

    def tokenize_function(examples):
        tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=block_size)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader

def get_device(local_rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")

def train(args):
    rank, local_rank, world_size = setup_ddp()

    device = get_device(local_rank)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(model.config.vocab_size)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()  # Initialize gradient scaler

    batch_size = 8 // world_size
    train_loader = get_shakespeare_dataloader(rank, world_size, batch_size)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Use autocast for mixed precision
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              labels=labels)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")

        if rank == 0:
            model.eval()
            with autocast(device_type='cuda', dtype=torch.float16):  # Use autocast for generation too
                prompt = "The king said"
                inputs = tokenizer(prompt, return_tensors='pt') 
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)

                generated_ids = model.module.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
                    eos_token_id=tokenizer.eos_token_id   # Explicitly set eos_token_id
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n=== Sample Generated Text ===\n{generated_text}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=1000)
    args = parser.parse_args()

    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()
