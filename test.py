import torch
from src.models.video_net import VideoCompressor
from src.utils.metrics import psnr
from PIL import Image
from torchvision import transforms
import argparse
import os

def test_one_pair(model_path, img_cur_path, img_ref_path, use_attention=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = VideoCompressor(use_attention=use_attention).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Model checkpoint not found, using random weights for demo.")
        
    model.eval()
    
    # Prepare Data
    # Resize to 256x256 to match training and SPyNet requirements
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    if not os.path.exists(img_cur_path) or not os.path.exists(img_ref_path):
        print("Image files not found. Generating random noise for demo.")
        x_cur = torch.rand(1, 3, 256, 256).to(device)
        x_ref = torch.rand(1, 3, 256, 256).to(device)
    else:
        img_cur = Image.open(img_cur_path).convert('RGB')
        img_ref = Image.open(img_ref_path).convert('RGB')
        x_cur = transform(img_cur).unsqueeze(0).to(device)
        x_ref = transform(img_ref).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = model(x_cur, x_ref)
        x_rec = out["x_rec"]
        
        # Calculate Metrics
        psnr_val = psnr(x_cur, x_rec)
        
        # Calculate BPP (Bits Per Pixel)
        likelihoods = out["likelihoods"]
        likelihoods_list = [likelihoods["mv"], likelihoods["res"]]
        
        import math
        bpp = 0
        for v in likelihoods_list:
            bpp += torch.log(v).sum() / (-math.log(2))
        
        num_pixels = x_cur.size(0) * x_cur.size(2) * x_cur.size(3)
        bpp_val = bpp / num_pixels

    print(f"Results (Attention={use_attention}):")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  BPP:  {bpp_val:.4f}")
    
    # Save output if real images were used
    if os.path.exists(img_cur_path):
        # Create a grid: Original | Reconstructed | Residual
        # Residual needs normalization to be visible
        res = (x_cur - x_rec).abs()
        res = (res - res.min()) / (res.max() - res.min())
        
        orig_img = transforms.ToPILImage()(x_cur.squeeze().cpu())
        rec_img = transforms.ToPILImage()(x_rec.squeeze().cpu())
        res_img = transforms.ToPILImage()(res.squeeze().cpu())
        
        # Concatenate images horizontally
        w, h = orig_img.size
        grid = Image.new('RGB', (w * 3, h))
        grid.paste(orig_img, (0, 0))
        grid.paste(rec_img, (w, 0))
        grid.paste(res_img, (w * 2, 0))
        
        grid.save("comparison.png")
        print("Saved comparison to comparison.png (Original | Reconstructed | Residual)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint_attn_epoch_0.pth")
    parser.add_argument("--cur", type=str, default="frame2.png")
    parser.add_argument("--ref", type=str, default="frame1.png")
    parser.add_argument("--no-attention", action="store_true", help="Use model without attention")
    args = parser.parse_args()
    
    test_one_pair(args.checkpoint, args.cur, args.ref, use_attention=not args.no_attention)
