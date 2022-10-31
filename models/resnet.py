import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

from models.ncm import NearestClassMean

# ResNet
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, threshold, device):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_features = nf * 8 * block.expansion # 160
        self.num_classes = num_classes
        self.threshold = threshold
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.ncm = NearestClassMean(self.input_features, self.num_classes, device=device)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x, y, ncm_update):
        self.ncm.fit_batch(x, y, ncm_update)
        x = self.ncm.predict(x, ncm_update=ncm_update, return_probas=True)

        return x

    def forward(self, x, y, ncm_update):
        out = self.features(x)
        logit = self.logits(out, y, ncm_update)
        
        return logit

    def ood_logits(self, feature, yul):
        new_feature = torch.zeros(1)
        new_yul = torch.zeros(1)

        for idx in range(feature.size(0)):
            out = feature[idx].view(1, -1)
            out_y = yul[idx].view(1, -1)
            logits = self.ncm.ood_predict(out)
            if torch.max(logits) > self.threshold:
                if torch.count_nonzero(new_feature) == 0:
                    new_feature = out
                    new_yul = out_y
                new_feature = torch.cat([new_feature, out])
                new_yul = torch.cat([new_yul, out_y])
        
        return new_feature, new_yul

    def predict(self, x):
        feature = self.features(x)
        out = self.ncm.predict(feature, ncm_update=True, return_probas=True)

        return feature, out

    # update center
    def ood_update(self, xul, yul, xb, yb):
        feature = self.features(xul)
        feature, yul = self.ood_logits(feature, yul) # pseduo-labeling and filtering noisy data

        if torch.Tensor.dim(feature) < 2:
            self.ncm.fit_batch(xb, yb, ncm_update=True)
            out = self.ncm.predict(xb, ncm_update=True, return_probas=True)
            target = yb
        else:
            ood_x = torch.cat([feature, xb])
            ood_y = torch.cat([yul, yb])
            self.ncm.fit_batch(ood_x, ood_y, ncm_update=True)
            out = self.ncm.predict(ood_x, ncm_update=True, return_probas=True)
            target = ood_y

        return out, target.squeeze(dim=-1)