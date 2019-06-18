# Optional list of dependencies required by the package
dependencies = ['torch', 'scipy', 'torchvision']

from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from torchvision.models.googlenet import googlenet
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from torchvision.models.mobilenet import mobilenet_v2


from sotabench.image_classification import imagenet
import torchvision.transforms as transforms
import PIL


def benchmark():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_transform = transforms.Compose([
        transforms.Resize(256, PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet.benchmark(
        model=resnext101_32x8d(retrained=True),
        paper_model_name='ResNext-101',
        paper_arxiv_id='1611.05431',
        paper_pwc_id='aggregated-residual-transformations-for-deep',
        input_transform=input_transform,
        batch_size=256,
        num_gpu=1
    )
