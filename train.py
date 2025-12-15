import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.video_net import VideoCompressor
from src.datasets.video_dataset import RandomDummyDataset, Vimeo90kDataset
import math

def compute_rate_loss(likelihoods_list):
    """
    Compute the bitrate (bpp) from likelihoods.
    R = -sum(log2(likelihoods))
    """
    bpp_loss = 0
    for v in likelihoods_list:
        # log2(x) = ln(x) / ln(2)
        bpp_loss += torch.log(v).sum() / (-math.log(2))
    return bpp_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = VideoCompressor(use_attention=not args.no_attention).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Use Dummy Dataset if no path provided, else use Vimeo90k
    if args.dataset is None:
        print("No dataset path provided. Using RandomDummyDataset for demonstration.")
        dataset = RandomDummyDataset(length=100)
    else:
        print(f"Loading Vimeo90k dataset from {args.dataset}")
        dataset = Vimeo90kDataset(args.dataset)
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Lambda controls the trade-off between Rate (bpp) and Distortion (MSE)
    # Higher lambda = better quality, higher bitrate
    lmbda = 256 
    
    print(f"Starting training... (Attention: {not args.no_attention})")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (x_cur, x_ref) in enumerate(dataloader):
            x_cur = x_cur.to(device)
            x_ref = x_ref.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = model(x_cur, x_ref)
            
            x_rec = out["x_rec"]
            likelihoods = out["likelihoods"]
            
            # 1. Distortion Loss (MSE)
            # Multiply by 255^2 if you want to match standard literature scale, 
            # or keep as is for normalized [0,1]
            distortion_loss = F.mse_loss(x_cur, x_rec)
            
            # 2. Rate Loss (Bitrate)
            # Combine likelihoods from Motion and Residual
            likelihoods_list = [likelihoods["mv"], likelihoods["res"]]
            
            # Calculate total bits for the batch
            # We normalize by number of pixels to get bpp (bits per pixel)
            N, _, H, W = x_cur.size()
            num_pixels = N * H * W
            rate_loss = compute_rate_loss(likelihoods_list) / num_pixels
            
            # Total Loss = R + lambda * D
            loss = rate_loss + lmbda * distortion_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate PSNR for monitoring
            mse = distortion_loss.item()
            current_psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 100
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (R: {rate_loss.item():.4f}, D: {distortion_loss.item():.4f}, PSNR: {current_psnr:.2f} dB)")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        
        # Learning rate decay
        scheduler.step()
        
        # Save checkpoint
        suffix = "no_attn" if args.no_attention else "attn"
        # Only save every 10 epochs or the last one to save space
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), f"checkpoint_{suffix}_epoch_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--no-attention", action="store_true", help="Disable CBAM attention")
    args = parser.parse_args()
    
    train(args)
