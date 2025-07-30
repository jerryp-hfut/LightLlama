import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

from model.llama import LLaMA, RMSNorm

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dropout_rate=0.1, max_seq_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = LLaMA(d_model, num_heads, num_layers, dropout_rate, max_seq_len)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.gamma)
        
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)

    def forward(self, x, start_pos=0, cache=None):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN/Inf detected in input tokens: {x}")
            
        x = self.embedding(x)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN/Inf detected after embedding")
            print(f"Embedding stats: mean={x.mean():.6f}, std={x.std():.6f}")
            
        x, cache = self.transformer(x, start_pos, cache)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN/Inf detected after transformer")
            print(f"Transformer output stats: mean={x.mean():.6f}, std={x.std():.6f}")
            
        x = self.output(x)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN/Inf detected in final logits")
            print(f"Output layer weight stats: mean={self.output.weight.mean():.6f}, std={self.output.weight.std():.6f}")
            
        return x, cache

def log_metrics_to_tensorboard(writer, metrics, step):
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
            writer.add_scalar(key, value, step)

def log_model_weights_to_tensorboard(writer, model, step):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f'weights/{name}', param.data, step)
            writer.add_scalar(f'weights_stats/{name}_mean', param.data.mean().item(), step)
            writer.add_scalar(f'weights_stats/{name}_std', param.data.std().item(), step)
            writer.add_scalar(f'weights_stats/{name}_norm', param.data.norm().item(), step)
            
            writer.add_histogram(f'gradients/{name}', param.grad.data, step)
            writer.add_scalar(f'gradients_stats/{name}_mean', param.grad.data.mean().item(), step)
            writer.add_scalar(f'gradients_stats/{name}_std', param.grad.data.std().item(), step)
            writer.add_scalar(f'gradients_stats/{name}_norm', param.grad.data.norm().item(), step)

def log_learning_rate_to_tensorboard(writer, lr, step):
    writer.add_scalar('training/learning_rate', lr, step)

def compute_cross_entropy(logits, targets):
    batch_size, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    targets = targets.view(-1)
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"Warning: logits contains NaN or Inf!")
        print(f"logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
        return torch.tensor(float('inf'), device=logits.device)
    
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    logits = logits - logits_max
    
    logits = torch.clamp(logits, min=-20, max=20)
    
    loss = nn.functional.cross_entropy(logits, targets, reduction='mean')
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: loss is NaN or Inf!")
        return torch.tensor(float('inf'), device=logits.device)
    
    return loss

def compute_perplexity(loss):
    return torch.exp(loss)

def get_lr_cosine_schedule(t, max_lr, min_lr, warmup_steps, cosine_steps):
    if t < warmup_steps:
        return max_lr * t / warmup_steps
    elif t < cosine_steps:
        decay_ratio = (t - warmup_steps) / (cosine_steps - warmup_steps)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    else:
        return min_lr

def gradient_clipping(parameters, max_norm=1.0, eps=1e-6):
    # 检查是否有NaN梯度
    has_nan_grad = False
    for p in parameters:
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            has_nan_grad = True
            print(f"Warning: NaN or Inf gradient detected in parameter shape {p.grad.shape}")
            p.grad.zero_()  # 将NaN梯度置零
    
    if has_nan_grad:
        print("Warning: NaN gradients detected and zeroed!")
        return
    
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
        
    total_norm = torch.sqrt(sum(grad.norm(2) ** 2 for grad in grads))
    
    # 检查总范数
    if torch.isnan(total_norm) or torch.isinf(total_norm):
        print(f"Warning: gradient norm is NaN or Inf!")
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()
        return
    
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)

def save_checkpoint(model, optimizer, iteration, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out_path)

def load_checkpoint(src_path, model, optimizer):
    checkpoint = torch.load(src_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

class DataLoader:
    def __init__(self, file_path, batch_size, context_length, device):
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self):
        max_idx = len(self.data) - self.context_length - 1
        indices = np.random.randint(0, max_idx, self.batch_size)
        inputs = torch.zeros((self.batch_size, self.context_length), dtype=torch.long, device=self.device)
        targets = torch.zeros((self.batch_size, self.context_length), dtype=torch.long, device=self.device)
        
        for i, idx in enumerate(indices):
            input_seq = self.data[idx:idx + self.context_length].astype(np.int64)
            target_seq = self.data[idx + 1:idx + self.context_length + 1].astype(np.int64)
            
            if np.any(input_seq < 0) or np.any(target_seq < 0):
                print(f"Warning: negative token indices detected at batch {i}")
                continue
                
            inputs[i] = torch.from_numpy(input_seq)
            targets[i] = torch.from_numpy(target_seq)
            
        inputs = torch.clamp(inputs, 0, 9999)  # vocab_size - 1
        targets = torch.clamp(targets, 0, 9999)
        
        return inputs, targets

def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
        print(f"Using device: CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_float32_matmul_precision('high')  # Apple Silicon
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    vocab_size = 10000
    d_model = 768 
    num_heads = 16  
    num_layers = 8 
    context_length = 256 
    batch_size = args.batch_size
    total_steps = args.total_steps
    max_lr = args.max_lr
    min_lr = args.min_lr
    warmup_steps = args.warmup_steps
    cosine_steps = args.cosine_steps

    model = LanguageModel(vocab_size, d_model, num_heads, num_layers, max_seq_len=context_length).to(device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=max_lr, 
        betas=(0.9, 0.95), 
        eps=1e-8, 
        weight_decay=0.01 
    )
    
    print("Checking model initialization...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: NaN or Inf in parameter {name}")
        param_stats = f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, range=[{param.min().item():.6f}, {param.max().item():.6f}]"
        if 'embedding' in name or 'output' in name:
            print(param_stats)

    train_loader = DataLoader('data/TinyStoriesV2-GPT4-train.npy', batch_size, context_length, device)
    valid_loader = DataLoader('data/TinyStoriesV2-GPT4-valid.npy', batch_size, context_length, device)

    log_dir = f"../../tf-logs/{args.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    hparams = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'context_length': context_length,
        'batch_size': batch_size,
        'max_lr': max_lr,
        'min_lr': min_lr,
        'warmup_steps': warmup_steps,
        'cosine_steps': cosine_steps,
        'weight_decay': 0.01
    }
    
    try:
        sample_input = torch.randint(0, vocab_size, (1, context_length), device=device)
        writer.add_graph(model, sample_input)
    except Exception as e:
        print(f"Could not add model graph to TensorBoard: {e}")

    start_iteration = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_iteration = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Loaded checkpoint from {args.checkpoint_path} at iteration {start_iteration}")

    model.train()
    log_file = open(args.log_file, 'a')
    progress_bar = tqdm(range(start_iteration, total_steps), desc="Training")
    
    recent_train_loss = None
    recent_valid_loss = None
    step_times = []
    tokens_per_second = 0
    
    for step in progress_bar:
        step_start_time = time.time()
        
        lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_steps, cosine_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        log_learning_rate_to_tensorboard(writer, lr, step)

        inputs, targets = train_loader.get_batch()
        
        if inputs.max() >= vocab_size or inputs.min() < 0:
            print(f"Invalid input tokens: range=[{inputs.min()}, {inputs.max()}], expected=[0, {vocab_size-1}]")
            continue
            
        if targets.max() >= vocab_size or targets.min() < 0:
            print(f"Invalid target tokens: range=[{targets.min()}, {targets.max()}], expected=[0, {vocab_size-1}]")
            continue
        
        try:
            logits, _ = model(inputs)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            optimizer.zero_grad()
            continue
            
        loss = compute_cross_entropy(logits, targets)
        
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 50:
            print(f"\nStep {step}: Abnormal loss detected: {loss.item()}")
            print(f"Logits stats: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
            print(f"Logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
            print(f"Targets range: [{targets.min().item()}, {targets.max().item()}]")
            
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(name)
            if nan_params:
                print(f"NaN parameters detected in: {nan_params}")
                print("Re-initializing model...")
                model.init_weights()
                
            optimizer.zero_grad()
            continue
        
        perplexity = compute_perplexity(loss)

        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in model.parameters() if p.grad is not None))
        
        gradient_clipping(model.parameters())
        optimizer.step()

        recent_train_loss = loss.item()
        
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        step_times.append(step_time)
        if len(step_times) > 100:  # 保持最近100步的平均时间
            step_times.pop(0)
        
        avg_step_time = np.mean(step_times)
        tokens_per_second = (batch_size * context_length) / avg_step_time
        
        train_metrics = {
            'training/loss': recent_train_loss,
            'training/perplexity': perplexity.item(),
            'training/gradient_norm': grad_norm.item(),
            'training/tokens_per_second': tokens_per_second,
            'training/step_time': step_time
        }
        log_metrics_to_tensorboard(writer, train_metrics, step)
        
        if step % args.weights_log_interval == 0:
            log_model_weights_to_tensorboard(writer, model, step)

        postfix_dict = {
            'train_loss': f'{recent_train_loss:.4f}',
            'ppl': f'{perplexity.item():.2f}',
            'lr': f'{lr:.6f}',
            'tok/s': f'{tokens_per_second:.0f}'
        }
        if recent_valid_loss is not None:
            postfix_dict['valid_loss'] = f'{recent_valid_loss:.4f}'
        
        progress_bar.set_postfix(postfix_dict)

        if (step + 1) % args.eval_interval == 0:
            model.eval()
            eval_start_time = time.time()
            
            with torch.no_grad():
                valid_loss = 0.0
                valid_perplexity_sum = 0.0
                valid_batches = args.eval_steps
                
                for eval_step in range(valid_batches):
                    val_inputs, val_targets = valid_loader.get_batch()
                    val_logits, _ = model(val_inputs)
                    val_loss = compute_cross_entropy(val_logits, val_targets)
                    
                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        valid_loss += val_loss.item()
                        valid_perplexity_sum += compute_perplexity(val_loss).item()
                    
                valid_loss /= valid_batches
                valid_perplexity = valid_perplexity_sum / valid_batches
                
            eval_time = time.time() - eval_start_time
            model.train()
            
            recent_valid_loss = valid_loss
            
            valid_metrics = {
                'validation/loss': valid_loss,
                'validation/perplexity': valid_perplexity,
                'validation/eval_time': eval_time
            }
            log_metrics_to_tensorboard(writer, valid_metrics, step)
            
            if recent_train_loss is not None:
                writer.add_scalars('loss_comparison', {
                    'train': recent_train_loss,
                    'validation': valid_loss
                }, step)
                
                writer.add_scalars('perplexity_comparison', {
                    'train': perplexity.item(),
                    'validation': valid_perplexity
                }, step)
            
            # 日志记录
            log_entry = {
                'step': step + 1,
                'train_loss': recent_train_loss,
                'train_perplexity': perplexity.item(),
                'valid_loss': valid_loss,
                'valid_perplexity': valid_perplexity,
                'lr': lr,
                'gradient_norm': grad_norm.item(),
                'tokens_per_second': tokens_per_second,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(log_entry, log_file)
            log_file.write('\n')
            log_file.flush()
            
            # 更新进度条显示信息（包含验证损失）
            postfix_dict = {
                'train_loss': f'{recent_train_loss:.4f}',
                'valid_loss': f'{valid_loss:.4f}',
                'ppl': f'{perplexity.item():.2f}',
                'lr': f'{lr:.6f}',
                'tok/s': f'{tokens_per_second:.0f}'
            }
            progress_bar.set_postfix(postfix_dict)

        # 保存检查点
        if (step + 1) % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/checkpoint_step_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")

    # 记录最终超参数和指标
    final_metrics = {}
    if recent_train_loss is not None:
        final_metrics['final_train_loss'] = recent_train_loss
    if recent_valid_loss is not None:
        final_metrics['final_valid_loss'] = recent_valid_loss
    
    writer.add_hparams(hparams, final_metrics)
    
    # 关闭文件和TensorBoard writer
    log_file.close()
    writer.close()
    print(f"\nTraining completed!")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model on TinyStories")
    parser.add_argument('--total_steps', type=int, default=20000, help="Total training steps")
    parser.add_argument('--max_lr', type=float, default=1e-4, help="Maximum learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="Warmup steps")
    parser.add_argument('--cosine_steps', type=int, default=20000, help="Cosine annealing steps")
    parser.add_argument('--eval_interval', type=int, default=500, help="Evaluation interval")
    parser.add_argument('--eval_steps', type=int, default=200, help="Number of validation batches per evaluation")
    parser.add_argument('--checkpoint_interval', type=int, default=2000, help="Checkpoint interval")
    parser.add_argument('--weights_log_interval', type=int, default=200, help="Interval for logging weights/gradients to TensorBoard")
    parser.add_argument('--experiment_name', type=str, default='llama_training', help="Experiment name for TensorBoard logs")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to resume from checkpoint")
    parser.add_argument('--log_file', type=str, default='logs/train_log.jsonl', help="Log file path")
    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('runs', exist_ok=True)  # TensorBoard logs directory
    train(args)

if __name__ == "__main__":
    main()