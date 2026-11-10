import torch
import torch.nn.functional as F
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def preprocess_document(pil_image):
    """
    Takes a PIL image (grayscale TIFF from RVL-CDIP) and prepares
    it for ResNet-50 — resizes, converts to 3 channels, normalizes.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(pil_image)  # (3, 224, 224)


def paste_patch(doc_batch, stamp, patch_size=64, position="bottom-right"):
    """
    Pastes a stamp patch onto a batch of document images.

    doc_batch: (B, 3, 224, 224) normalized tensor
    stamp:     (1, 1, H, W) float tensor in [0, 1] from VAE decoder
    returns:   (B, 3, 224, 224) normalized tensor with patch applied
    """
    B, C, H, W = doc_batch.shape

    # Resize stamp to patch_size x patch_size
    stamp_resized = F.interpolate(stamp, size=(patch_size, patch_size),
                                  mode="bilinear", align_corners=False)  # (1,1,ps,ps)

    # Convert grayscale → 3 channels
    stamp_rgb = stamp_resized.repeat(1, 3, 1, 1)  # (1, 3, ps, ps)

    # Normalize stamp to match document normalization
    stamp_norm = normalize(stamp_rgb.squeeze(0)).unsqueeze(0)  # (1, 3, ps, ps)

    # Expand to batch
    stamp_batch = stamp_norm.expand(B, -1, -1, -1)  # (B, 3, ps, ps)

    # Determine position
    ps = patch_size
    positions = {
        "top-left":     (0, 0),
        "top-right":    (0, W - ps),
        "bottom-left":  (H - ps, 0),
        "bottom-right": (H - ps, W - ps),
        "center":       ((H - ps) // 2, (W - ps) // 2),
    }
    row, col = positions[position]

    # Clone to avoid in-place modification issues with autograd
    patched = doc_batch.clone()
    patched[:, :, row:row+ps, col:col+ps] = stamp_batch

    return patched  # (B, 3, 224, 224)