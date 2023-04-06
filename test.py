
import torch
from segmenter.segm.model.decoder3d import MaskTransformer3d
from segmenter.segm.model.segmenter3d import Segmenter3d
from segmenter.segm.model.vit3d import VisionTransformer3d
img_size = 38

device = "cuda"
vit = VisionTransformer3d((img_size, img_size, img_size), 38, 12, 128, 128, 8, 3, channels=1)
decoder = MaskTransformer3d(3, 38, 128, 2, 8, 128, 128, 0.0, 0.1)
net = Segmenter3d(vit, decoder, n_cls=3).to(device)

X = torch.randn(1, 1, img_size, img_size, img_size).to(device)
Y = torch.randn(1, 3, img_size, img_size, img_size)
test = net(X)
#crit = torch.nn.CrossEntropyLoss()
#loss =
print(sum(p.numel() for p in net.parameters() if p.requires_grad)) 