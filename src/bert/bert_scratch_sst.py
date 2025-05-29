import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import sys
import argparse
import time

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

def get_device(local_rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")

def get_sst2_dataloader(rank, world_size, batch_size):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = load_dataset('glue', 'sst2', split='train')

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

def create_bert_from_scratch(num_labels=2, vocab_size=30522):
    """Create BERT model from scratch with custom configuration"""
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,           # Can reduce for smaller model (e.g., 512, 256)
        num_hidden_layers=4,      # Can reduce for smaller model (e.g., 6, 4)
        num_attention_heads=8,    # Should divide hidden_size evenly
        intermediate_size=1024,    # Usually 4x hidden_size
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_labels=num_labels
    )
    
    model = BertForSequenceClassification(config)
    
    model.apply(model._init_weights)
    
    return model, config

def train(args):
    rank, local_rank, world_size = setup_ddp()
    device = get_device(local_rank)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model, config = create_bert_from_scratch(num_labels=args.num_classes, vocab_size=tokenizer.vocab_size)
    if rank == 0:
        print("Training BERT from scratch with configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")
        print(f"  Vocab size: {config.vocab_size}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    lr = args.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    num_training_steps = args.epochs * 1000  # Approximate
    warmup_steps = num_training_steps // 10
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=warmup_steps
    )
    
    scaler = GradScaler()

    # Data loader
    batch_size = args.batch_size // world_size
    train_loader = get_sst2_dataloader(rank, world_size, batch_size)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              labels=labels)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if step < warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            
            model.eval()
            with torch.no_grad():
                test_prompts = [
                    ("This movie is absolutely amazing!", 1),  # (text, true_label)
                    ("I hate this terrible film.", 0),
                    ("The acting was decent but the plot was boring.", 0),
                    ("Best movie I've ever seen!", 1),
                    ("Worst waste of time ever.", 0),
                    ("It was okay, nothing special.", 0),
                    ("Brilliant cinematography and excellent performances.", 1),
                    ("I fell asleep halfway through.", 0)
                ]
                
                correct_predictions = 0
                total_predictions = len(test_prompts)
                
                print(f"\n=== Sentiment Classification Results (Epoch {epoch+1}) ===")
                for prompt, true_label in test_prompts:
                    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', 
                                     truncation=True, max_length=128)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class = torch.argmax(predictions, dim=-1).item()
                        confidence = predictions[0][predicted_class].item()
                    
                    is_correct = predicted_class == true_label
                    if is_correct:
                        correct_predictions += 1
                    
                    sentiment = "Positive" if predicted_class == 1 else "Negative"
                    true_sentiment = "Positive" if true_label == 1 else "Negative"
                    status = "✓" if is_correct else "✗"
                    
                    print(f"Text: '{prompt}'")
                    print(f"True: {true_sentiment} | Predicted: {sentiment} (confidence: {confidence:.3f}) {status}")
                    print("-" * 70)
                
                accuracy = correct_predictions / total_predictions * 100
                print(f"Test Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
                print()
    
    if rank == 0:
        print("Saving BERT model...")
        os.makedirs(args.save_dir, exist_ok=True)
        
        model.module.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': args.epochs,
            'loss': epoch_loss/len(train_loader),
            'config': model.module.config
        }, os.path.join(args.save_dir, 'training_checkpoint.pt'))
        
        print(f"BERT model saved to {args.save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5 * 10)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./saved_bert_model')
    args = parser.parse_args()

    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()