from models.resnet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2

def Reduced_ResNet18(num_classes=20, nf=20, bias=True, threshold=0.1, device='cuda:0'):
    # Reduced ResNet18 as in GEM MIR(note that nf=20).
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, bias, threshold, device)

def ResNet18(num_classes=100, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, bias)

def mobilenet_v2(num_classes, threshold=0.1, device='cuda:0'):
    return MobileNetV2(num_classes, 1.0, threshold=threshold, device=device)