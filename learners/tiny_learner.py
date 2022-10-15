from tkinter import Variable
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
        # self.num_classes = self.config['num_classes']   # not use?

        self.frist_tasks = True

        self.model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']]().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.ood_criterion = nn.CrossEntropyLoss()

        self.model.num_classes = self.config['num_task']
        self.model.threshold = self.config['threshold']
        self.memory = self.config['memory']
        self.replay_buffer = torch.Tensor()
        self.buffer_logits = torch.Tensor()

    def add_valid_output_dim(self, dim=0):
        print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        print('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    # def learn_batch(self, train_loader, train_dataset, train_dataset_ul, model_dir, val_loader=None):
    def learn_batch(self, train_loader_l, train_loader_ul, model_dir, val_loader=None):

        print('Optimizer is reset!')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            print('Epoch:{0}'.format(epoch+1))

            # training labeled dataset
            for i, (xl, y)  in enumerate(train_loader_l):
                xl, y = xl.to(self.device), y.to(self.device)

                output = self.model.forward(xl, y).to(self.device)
                loss = self.criterion(output, y)

                optimizer.zero_grad()
                loss.requires_grad = True
                loss.backward()
                optimizer.step()

        # update replay buffer (exist buffer and new train dataset)
        if self.frist_tasks:
            self.model.eval()

            with torch.no_grad():
                for i, (xl, y)  in enumerate(train_loader_l):
                    xl, y = xl.to(self.device), y.to(self.device)

                    feature, output = self.model.predict(xl)
                    # feature, output, y = feature.cpu(), output.cpu(), y.cpu()

                    for idx, out in enumerate(output):
                        fe = feature[idx].view(1, -1)
                        out = torch.max(out).view(-1)

                        if self.replay_buffer.size(0) < self.memory:
                            if self.replay_buffer.size(0) == 0: 
                                self.replay_buffer = fe
                                self.buffer_logits = out
                            else:
                                self.replay_buffer = torch.cat([self.replay_buffer, fe])
                                self.buffer_logits = torch.cat([self.buffer_logits, out])
                        else:
                            min_value = torch.min(self.buffer_logits)
                            if out[0] > min_value:
                                min_idx = torch.argmin(self.buffer_logits)
                                self.replay_buffer[min_idx] = fe
                                self.buffer_logits[min_idx] = out
                        
        # Fine-tuning NCM using unlabed dataset(pseudo label)
        for i, (xul, yul)  in enumerate(train_loader_ul):
            xul, yul = xul.to(self.device), yul.to(self.device)

            self.model.ood_update(xul, yul, self.replay_buffer)

    def update_buffer(self, x):
        pass


