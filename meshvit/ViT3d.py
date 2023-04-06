import torch
from torch import nn
from einops import rearrange, repeat
from linformer import Linformer
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT3d(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 1, output_shape=None):
        super().__init__()
        self.output_shape = output_shape
        self.num_classes = num_classes
        if self.output_shape is not None:
            num_classes = num_classes*int(torch.prod(torch.Tensor(self.output_shape)).item())
        image_size_h = image_size
        image_size_w = image_size_h 
        image_size_d = image_size_h
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0 and image_size_d % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size) * (image_size_d // patch_size)
        patch_dim = channels * patch_size ** 3

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1 = patch_size, p2 = patch_size, p3 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.num_patches = num_patches
        self.h = image_size_h
        self.w = image_size_w
        self.d = image_size_d

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        if self.output_shape is not None:
            x = x.reshape(x.shape[0],self.num_classes, *self.output_shape)
        return x

def build_vit(dim=128, seq_len=49*7+1, depth=12, heads=8, k=64, image_size=32, patch_size=8, num_classes=10, channels=1, output_shape=None):
    #seq_len = int(image_size/patch_size) + 1
    patch_w = int(image_size/patch_size)
    seq_len = patch_w * patch_w * patch_w + 1
    seq_len = 344
    efficient_transformer = Linformer(
        dim=dim,
        seq_len=seq_len,  # 7x7x7 patches + 1 cls-token
        depth=depth,
        heads=heads,
        k=k
    )
    model = ViT3d(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            transformer=efficient_transformer,
            channels=channels,
            output_shape=output_shape
        )
    return model
    

if __name__=="__main__":
    image = torch.randn(64, 1, 8, 8, 8)
    model = build_vit(output_shape=[8,8,8], num_classes=104, patch_size=4)
    test = model(image)
    print("ok")
    print(test.shape)