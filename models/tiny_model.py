import os
from types import new_class
from numpy import outer
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


class NearestClassMean(nn.Module):
    """
    "Online Continual Learning for Embedded Devices"
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    Code from https://github.com/tyler-hayes/Embedded-CL
    """

    def __init__(self, input_shape, num_classes, backbone=None, device='cuda:0'):
        """
        Init function for the NCM model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(NearestClassMean, self).__init__()

        # NCM parameters
        self.device = device
        self.in_features = input_shape
        self.num_classes = num_classes

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for NCM
        self.temp_muK = torch.zeros((self.num_classes, self.in_features)).to(self.device)
        self.temp_cK = torch.zeros(self.num_classes).to(self.device)

        self.muK = torch.zeros((self.num_classes, self.in_features)).to(self.device) # class mean
        self.cK = torch.zeros(self.num_classes).to(self.device) # class count

    @torch.no_grad()
    def init_weights(self, first):
        if first:
            self.temp_muK = torch.zeros((self.num_classes, self.in_features)).to(self.device)
            self.temp_cK = torch.zeros(self.num_classes).to(self.device)
        else:
            self.temp_muK = self.muK
            self.temp_cK = self.cK

    @torch.no_grad()
    def fit(self, x, y, ncm_update):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        # print('x shape', len(x.shape))
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        # print('y shape', len(y.shape))
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means
        if ncm_update:
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
        else:
            self.temp_muK[y, :] += (x - self.temp_muK[y, :]) / (self.temp_cK[y] + 1).unsqueeze(1)
            self.temp_cK[y] += 1

    @torch.no_grad()
    def find_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: indices of closest points in A
        """
        M, d = B.shape
        with torch.no_grad():
            B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
            square_sub = torch.mul(A - B, A - B)  # square all elements
            dist = torch.sum(square_sub, dim=2)
        return -dist

    @torch.no_grad()
    def predict(self, X, ncm_update=False, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        if ncm_update:
            scores = self.find_dists(self.muK, X)

            # mask off predictions for unseen classes
            # visited_ix = torch.where(self.cK != 0)[0]
            # scores = scores[:, visited_ix]
            not_visited_ix = torch.where(self.cK == 0)[0]
            min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
            scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
                len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes
        else:
            scores = self.find_dists(self.temp_muK, X)

            # mask off predictions for unseen classes
            # visited_ix = torch.where(self.temp_cK != 0)[0]
            # scores = scores[:, visited_ix]
            not_visited_ix = torch.where(self.temp_cK == 0)[0]
            min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
            scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
                len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def ood_predict(self, x):
        return self.predict(x, ncm_update=True, return_probas=True)

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, ncm_update):
        # fit NCM one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), ncm_update)

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
        self.threshold = 0.1
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


def Reduced_ResNet18(num_classes=100, nf=20, bias=True, threshold=0.1, device='cuda:0'):
    # Reduced ResNet18 as in GEM MIR(note that nf=20).
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, bias, threshold, device)

def ResNet18(num_classes=100, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, bias)