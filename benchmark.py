from sotabench.image_classification import ImageNet
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
import PIL


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

ImageNet.benchmark(
    model=resnext101_32x8d(pretrained=True),
    paper_model_name='ResNext-101',
    paper_arxiv_id='1611.05431',
    paper_pwc_id='aggregated-residual-transformations-for-deep',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)
