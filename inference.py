import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from PIL import Image



# -----------------------------------------------------------
# 1. Ricostruzione modello UNet
# -----------------------------------------------------------
def build_model(device):
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    return model


# -----------------------------------------------------------
# 2. Caricamento pesi
# -----------------------------------------------------------
def load_weights(model, ckpt_path, device):
    print(f"\nCarico pesi da: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✔️ Pesi caricati correttamente.\n")


# -----------------------------------------------------------
# 3. Controllo immagine con PILLOW
# -----------------------------------------------------------
def inspect_image(img_path):
    img = Image.open(img_path)
    arr = np.array(img)

    print("\n======================")
    print(" ANALISI IMMAGINE PIL")
    print("======================")
    print("Percorso:", img_path)
    print("Formato:", img.format)
    print("Modalità (color mode):", img.mode)
    print("Dimensioni (W, H):", img.size)
    print("Shape numpy:", arr.shape)
    print("Dtype:", arr.dtype)
    print("Range pixel:", arr.min(), "→", arr.max())
    print("======================\n")

    return img, arr


# -----------------------------------------------------------
# 4. Trasformazioni MONAI per inference
# -----------------------------------------------------------
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    EnsureType,
    Lambda,
)


PATCH_SIZE = 256

def get_infer_transforms(patch_size: int = PATCH_SIZE):
    """
    Trasformazioni MONAI per una singola immagine RGBA:
    - LoadImage: carica PNG → (H, W, 4)
    - EnsureChannelFirst: → (4, H, W)
    - Lambda: tiene solo il primo canale → (1, H, W)
    - Resize: → (1, patch_size, patch_size)
    - ScaleIntensity: normalizza 0–255 → 0–1
    - EnsureType: torch.Tensor
    """
    transforms = Compose(
        [
            LoadImage(image_only=True),           # (H, W, 4)
            EnsureChannelFirst(),                # (4, H, W)
            Lambda(func=lambda x: x[:1, ...]),   # (1, H, W) 1 solo canale
            Resize((patch_size, patch_size)),
            ScaleIntensity(),
            EnsureType(),
        ]
    )
    return transforms

# -----------------------------------------------------------
# 5. Inference vera
# -----------------------------------------------------------
def run_inference_on_image(model, img_path, device, patch_size: int = PATCH_SIZE):
    infer_transforms = get_infer_transforms(patch_size)

    # Applica le trasformazioni alla SOLA immagine
    img = infer_transforms(img_path)      # (1, H, W)
    img = img.unsqueeze(0).to(device)     # (1, 1, H, W) per la UNet

    with torch.no_grad():
        preds = model(img)                # logits
        probs = torch.sigmoid(preds)      # [0,1]
        mask = (probs > 0.5).float()      # binaria

    # Ritorno versioni numpy 2D
    return img[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy()


def show_result(img_np, mask_np):
    """
    Mostra immagine originale preprocessata e maschera predetta.
    """
    plt.figure(figsize=(12, 6))

    # Immagine preprocessata
    plt.subplot(1, 2, 1)
    plt.title("Immagine preprocessata")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    # Mask predetta
    plt.subplot(1, 2, 2)
    plt.title("Mask predetta (threshold 0.5)")
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def save_mask(mask_np, out_path="mask_output.png"):
    """
    Salva la mask predetta come immagine PNG.
    """
    mask_img = (mask_np * 255).astype("uint8")
    Image.fromarray(mask_img).save(out_path)
    print(f"✔️ Mask salvata in: {out_path}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best_metric_model.pth")
    parser.add_argument("--image", type=str, default="Chest_Xray_PA_3-8-2010.png")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Uso device:", device)

    # 1. Crea modello
    model = build_model(device)

    # 2. Carica pesi
    load_weights(model, args.ckpt, device)

    # 3. Analizza immagine
    inspect_image(args.image)

    # 4. Inferenza vera
    img_np, mask_np = run_inference_on_image(model, args.image, device)

    # 5. Visualizza risultati
    show_result(img_np, mask_np)

    # 6. (opzionale) salva mask
    save_mask(mask_np, "mask_output.png")

if __name__ == "__main__":
    main()
