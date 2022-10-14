import torch
import models
import torch.nn as nn
from models.tiny_model import NearestClassMean

import numpy as np

class SSCL():
    def __init__(self, learner_config):
        super(SSCL, self).__init__()
        self.config = learner_config
        device_id = 'cuda:' + str(self.config['device'])
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        self.valid_out_dim = 0
        self.num_classes = self.config['num_classes']

        # num tasks for repeats
        self.tasks = 0
        self.model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']]().to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.ood_criterion = nn.CrossEntropyLoss()

    def add_valid_output_dim(self, dim=0):
        print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        print('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    # def learn_batch(self, train_loader, train_dataset, train_dataset_ul, model_dir, val_loader=None):
    def learn_batch(self, train_loader_l, train_loader_ul, model_dir, val_loader=None):
        self.tasks += 1

        print('Optimizer is reset!')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            print('Epoch:{0}'.format(epoch+1))

            for i, (xl, y)  in enumerate(train_loader_l):
                xl, y = xl.to(self.device), y.to(self.device)

                output = self.model.forward(xl, y).to(self.device)
                print(output.size())
                loss = self.criterion(output, y)

                print(loss)

                optimizer.zero_grad()
                loss.backward()

