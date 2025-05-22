import torch
import time

def calculate_orth_loss(W_dec, group_count=32):
    D, N = W_dec.shape
    group_size = D // group_count
    
    perm = torch.randperm(D, device=W_dec.device)
    shuffled = W_dec[perm]
    
    grouped = shuffled.view(group_count, group_size, N)
    cos_sims = torch.bmm(grouped, grouped.transpose(1, 2))
    
    mask = ~torch.eye(group_size, device=W_dec.device).bool()
    max_vals = (cos_sims * mask).max(dim=2).values
    return (max_vals ** 2).mean()

def benchmark():
    sizes = [(1024, 256), (2048, 512), (4096, 1024), (8192, 2048), (16384, 4096), (32768, 8192)]
    with open("pytorch_times.txt", "w") as f:
        for D, N in sizes:
            device = torch.device('cuda')
            W = torch.randn(D, N, device=device)
            
            # Warm-up iteration
            loss = calculate_orth_loss(W)
            torch.cuda.synchronize()
            
            # Timed iteration
            start = time.time()
            loss = calculate_orth_loss(W)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            result = f"Size {D}x{N}: {elapsed*1000:.2f} ms\n"
            print(result.strip())
            f.write(result)

if __name__ == "__main__":
    benchmark()