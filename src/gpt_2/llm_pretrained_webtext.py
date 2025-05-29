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

def get_openwebtext_dataloader(rank, world_size, batch_size, block_size=128):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token normally
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset('openwebtext', split='train', streaming=True, trust_remote_code=True)

    def tokenize_function(examples):
        tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=block_size)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    from torch.utils.data import IterableDataset
    
    class OpenWebTextDataset(IterableDataset):
        def __init__(self, dataset, rank, world_size):
            self.dataset = dataset
            self.rank = rank
            self.world_size = world_size
            
        def __iter__(self):
            # Skip samples for other ranks
            for i, sample in enumerate(self.dataset):
                if i % self.world_size == self.rank:
                    yield {
                        'input_ids': torch.tensor(sample['input_ids']),
                        'attention_mask': torch.tensor(sample['attention_mask']),
                        'labels': torch.tensor(sample['labels'])
                    }
    
    dataset_wrapper = OpenWebTextDataset(tokenized_dataset, rank, world_size)
    
    dataloader = DataLoader(
        dataset_wrapper,
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
    scaler = GradScaler()

    batch_size = 8 // world_size
    train_loader = get_openwebtext_dataloader(rank, world_size, batch_size)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

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
            batch_count += 1
            
            if batch_count >= 1000:  # Adjust this number as needed
                break

        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/batch_count}")

        if rank == 0:
            model.eval()
            with autocast(device_type='cuda', dtype=torch.float16):
                prompts = [
                    "The latest research shows that",
                    "In recent years, technology has",
                    "According to experts,",
                    "The company announced that",
                    "Scientists have discovered"
                ]
                
                import random
                prompt = random.choice(prompts)
                
                inputs = tokenizer(prompt, return_tensors='pt') 
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)

                generated_ids = model.module.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id 
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n=== Sample Generated Text (Prompt: '{prompt}') ===\n{generated_text}\n")

    if rank == 0:
        print("Saving model...")
        save_dir = "./saved_model"
        os.makedirs(save_dir, exist_ok=True)
        
        model.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': num_epochs,
        }, os.path.join(save_dir, 'training_checkpoint.pt'))
        
        print(f"Model saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./saved_model', help='Directory to save the trained model')
    args = parser.parse_args()

    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()
