"""Visual Transformer model definition using Linformer"""

from linformer import Linformer
from vit_pytorch.efficient import ViT

def build_vit(dim=128, seq_len=49+1, depth=12, heads=8, k=64, image_size=224, patch_size=32, num_classes=10, channels=3):
    #seq_len = int(image_size/patch_size) + 1
    patch_w = int(image_size/patch_size)
    #seq_len = patch_w * patch_w * patch_w + 1
    efficient_transformer = Linformer(
        dim=dim,
        seq_len=seq_len,  # 7x7 patches + 1 cls-token
        depth=depth,
        heads=heads,
        k=k
    )
    model = ViT(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            transformer=efficient_transformer,
            channels=channels,
        )
    return model

if __name__=="__main__":
    vit = build_vit()
    print(vit.modules())
