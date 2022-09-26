import torch
import models
import torch.nn as nn

import numpy as np

class SSCL():
    def __init__(self, learner_config):
        super(SSCL, self).__init__()
        self.config =learner_config

        self.valid_out_dim = 0
        self.num_classes = self.config['num_classes']

        # num tasks for repeats
        self.tasks = 0
        self.first_task = True
        self.model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']]()
        self.ood_model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']]()

        self.criterion = nn.CrossEntropyLoss()

    def add_valid_output_dim(self, dim=0):
        print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        print('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def learn_batch(self, train_loader, train_dataset, train_dataset_ul, model_dir, val_loader=None):
        self.tasks += 1

        print('Optimizer is reset!')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epoch']):
            print('Epoch:{0}'.format(epoch+1))
            
            self.model.train()
            total_loss = 0
            total_count = 0
            
            for i, (xl, y, xul, yul, task)  in enumerate(train_loader):
                logits = self.model.forward(xl)[:, :self.valid_out_dim]
                loss = self.criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss
                total_count += 1
            print(total_loss / total_count)

