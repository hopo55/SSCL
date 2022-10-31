import torch
import torch.nn as nn

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