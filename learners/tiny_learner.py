import torch
import models
import torch.nn as nn
from learners.utils import accuracy, AverageMeter, tensor_logger

import numpy as np

class SSCL():
    def __init__(self, learner_config):
        super(SSCL, self).__init__()
        self.config = learner_config
        device_id = 'cuda:' + str(self.config['device'])
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        self.valid_out_dim = 0
        self.current_tasks = 0
        # self.num_classes = self.config['num_classes']   # not use?

        self.first_tasks = True

        self.model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']]().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.ood_criterion = nn.CrossEntropyLoss()

        self.model.num_classes = self.config['num_task']
        self.model.threshold = self.config['threshold']
        self.memory = self.config['memory']
        self.past_memory = self.memory

        self.buffer_x = torch.Tensor()
        self.buffer_y = torch.Tensor()
        self.buffer_logits = torch.Tensor()

        self.logger = tensor_logger(self.config['logdir'])

    def add_valid_output_dim(self, dim=0):
        print('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        print('Incremental class: New Valid output dimension:', self.valid_out_dim)
        self.current_tasks += 1
        print('Incremental class: Current tasks:', self.current_tasks)


    # def learn_batch(self, train_loader, train_dataset, train_dataset_ul, model_dir, val_loader=None):
    def learn_batch(self, train_loader_l, train_loader_ul, model_dir):

        print('Optimizer is reset!')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            losses = AverageMeter()
            acc = AverageMeter()
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

                losses.update(loss,  y.size(0))
                acc.update(accuracy(output, y), y.size(0))
            
            print(' * Train Loss {loss.avg:.3f}'.format(loss=losses))
            print(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

        self.logger.writer('Training Accuracy', acc.avg, self.current_tasks)

        # Update replay buffer (only first task new train dataset)
        if self.first_tasks:
            self.update_buffer(train_loader_l)

        # Fine-tuning NCM using unlabed dataset(pseudo label) and replay buffer
        self.model.eval()
        ood_losses = AverageMeter()
        ood_acc = AverageMeter()
        
        with torch.no_grad():
            for i, (xul, yul)  in enumerate(train_loader_ul):
                xul, yul = xul.to(self.device), yul.to(self.device)

                self.model.ood_update(xul, yul, self.buffer_x, self.buffer_y)
                
                ood_output = self.model.forward(xul, yul).to(self.device)
                ool_loss = self.criterion(ood_output, yul)

                ood_losses.update(ool_loss,  yul.size(0))
                ood_acc.update(accuracy(ood_output, yul), yul.size(0))
            
        print(' * Train OOD Loss {loss.avg:.3f}'.format(loss=ood_losses))
        print(' * Train OOD Acc {acc.avg:.3f}'.format(acc=ood_acc))
        
        self.logger.writer('Training OOD Accuracy', ood_acc.avg, self.current_tasks)

        # Update replay buffer (exist buffer and new train dataset)
        if not self.first_tasks:
            self.update_buffer(train_loader_ul, self.first_tasks)

        self.first_tasks = False

    def update_buffer(self, dataloader, first=True):
        self.model.eval()

        if first == False:
            new_memory_size, _ = divmod(self.memory, self.current_tasks)

            for idx in range(self.current_tasks - 1):
                if self.current_tasks == 2:
                    start = self.memory * idx
                    end = self.memory * (idx + 1)
                else:
                    start = self.past_memory * idx
                    end = self.past_memory * (idx + 1)

                sort_index = torch.argsort(self.buffer_logits[start:end])
                temp_buffer_x = self.buffer_x[start:end]
                temp_buffer_x = temp_buffer_x[sort_index < new_memory_size].to(self.device)

                temp_buffer_y = self.buffer_y[start:end]
                temp_buffer_y = temp_buffer_y[sort_index < new_memory_size].to(self.device)

                temp_buffer_logits = self.buffer_logits[start:end]
                temp_buffer_logits = temp_buffer_logits[sort_index < new_memory_size]
                    
                if idx == 0:
                    new_buffer_x = temp_buffer_x
                    new_buffer_y = temp_buffer_y
                    new_buffer_logits = temp_buffer_logits
                else:
                    new_buffer_x = torch.cat([new_buffer_x, temp_buffer_x])
                    new_buffer_y = torch.cat([new_buffer_y, temp_buffer_y])
                    new_buffer_logits = torch.cat([new_buffer_logits, temp_buffer_logits])

            self.past_memory = new_memory_size
            self.buffer_x = new_buffer_x
            self.buffer_y = new_buffer_y
            self.buffer_logits = new_buffer_logits

        with torch.no_grad():
            for x, y  in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                feature, output = self.model.predict(x)

                for idx, out in enumerate(output):
                    fe = feature[idx].view(1, -1)
                    fe_y = y[idx].view(1, -1)
                    out = torch.max(out).view(-1)
                    
                    if self.buffer_x.size(0) < self.memory:
                        if self.buffer_x.size(0) == 0 and first: 
                            self.buffer_x = fe
                            self.buffer_y = fe_y
                            self.buffer_logits = out
                        else:
                            self.buffer_x = torch.cat([self.buffer_x, fe])
                            self.buffer_y = torch.cat([self.buffer_y, fe_y])
                            self.buffer_logits = torch.cat([self.buffer_logits, out])
                    else:
                        min_value = torch.min(self.buffer_logits)
                        if out[0] > min_value:
                            min_idx = torch.argmin(self.buffer_logits)
                            self.buffer_x[min_idx] = fe
                            self.buffer_y[min_idx] = fe_y
                            self.buffer_logits[min_idx] = out

    def validatioin(self, test_loader):
        print("Validation test dataset")
        val_acc = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for x, y  in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                _, output = self.model.predict(x)
                val_acc.update(accuracy(output.to(self.device), y), y.size(0))

            print(' * Validation Acc {acc.avg:.3f}'.format(acc=val_acc))
            self.logger.writer('Validation Accuracy', val_acc.avg, self.current_tasks)
