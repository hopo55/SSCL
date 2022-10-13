import torch
import random
import argparse
from dataloaders import loader
import learners
import dataloaders
import numpy as np
from dataloaders.utils import *
from torch.utils.data import DataLoader

def run(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

    if args.dataset == 'CIFAR10':
        Dataset = dataloaders.CIFAR10
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        Dataset = dataloaders.CIFAR100
        num_classes = 100
    else:
        print('Dataset does not select.')
        exit()

    # load tasks
    class_order = np.arange(num_classes).tolist()
    class_order_logits = np.arange(num_classes).tolist()
    if seed > 0 and args.rand_split:
        random.seed(seed)
        random.shuffle(class_order)

    tasks = []
    tasks_logits = []
    p = 0
    first_split_size = 5
    other_split_size = 5

    while p < num_classes:
        inc = other_split_size if p > 0 else first_split_size
        tasks.append(class_order[p:p+inc])
        tasks_logits.append(class_order_logits[p:p+inc])
        p += inc
    num_tasks = len(tasks)
    task_names = [str(i+1) for i in range(num_tasks)]

    # datasets and dataloaders
    train_transform = dataloaders.utils.get_transform(dataset=args.dataset)
    test_transform  = dataloaders.utils.get_transform(dataset=args.dataset)
    
    train_dataset = Dataset(args.dataroot, train=True, label=True, num_label_data=args.labeled_samples, class_type=args.class_type, transform=train_transform, download=True)

    train_dataset_ul = Dataset(args.dataroot, train=True, label=False, num_label_data=args.labeled_samples, class_type=args.class_type, transform=train_transform, download=True)

    test_dataset  = Dataset(args.dataroot, train=False, class_type=args.class_type, transform=train_transform, download=True)

    learner_config = {'num_classes': num_classes,
                      'model_type' : args.model_type,
                      'model_name' : args.model_name,
                      'epoch' : args.epoch,
                      'lr' : args.lr,
                      'momentum' : args.momentum,
                      'weight_decay' : args.weight_decay,
                     }
    learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](learner_config)

    # in case tasks reset...
    tasks = train_dataset.tasks
    max_task = len(tasks)

    for i in range(max_task):
        train_name = task_names[i]
        print('======================', train_name, '=======================')

        # load dataset for task
        task = tasks_logits[i]
        prev = sorted(set([k for task in tasks_logits[:i] for k in task]))

        train_dataset.load_dataset(prev, i, train=True)
        train_dataset_ul.load_dataset(prev, i, train=True)
        out_dim_add = len(task)

        # load dataloader
        train_loader_l = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.workers)
        train_loader_ul = DataLoader(train_dataset_ul, batch_size=args.ul_batch_size, shuffle=True, drop_last=False, num_workers=args.workers)
        train_loader = loader.SSCLDataLoader(train_loader_l, train_loader_ul)

        test_dataset.load_dataset(prev, i, train=False)
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
        
        # add valid class for classifier
        model_save_dir = args.log_dir + '/models/task-'+task_names[i]+'/'
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

        learner.add_valid_output_dim(out_dim_add)
        learner.learn_batch(train_loader, train_dataset, train_dataset_ul, model_save_dir, test_loader)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-Supervised Continual Learning')

    # Standard Args
    parser.add_argument('--seed', type=int, default=0, help='Set seed (default=0)')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="CIFAR10|CIFAR100")
    parser.add_argument('--log_dir', type=str, default="outputs", help="Save experiments results in dir for future plotting!")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--workers', type=int, default=8, help="#Thread for dataloader")
    parser.add_argument('--model_type', type=str, default='tiny_model', help="The type tin_model of backbone network")
    parser.add_argument('--model_name', type=str, default='Reduced_ResNet18', help="The name of actual model for the backbone")
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # SSCL Args
    parser.add_argument('--class_type', type=str, default='super', help="vanilla|super")
    parser.add_argument('--rand_split', default=False, action='store_true', help="Randomize the classes in splits")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate training dataset rather than testing data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learner_type', type=str, default='tiny_learner', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='SSCL', help="The class name of learner")
    parser.add_argument('--ul_batch_size', type=int, default=32)
    parser.add_argument('--labeled_samples', type=int, default=500, help='Number of labeled samples each task in ssl')
    parser.add_argument('--unlabeled_task_samples', type=int, default=-1, help='Number of unlabeled samples in each task in ssl')

    args = parser.parse_args()

    run(args)