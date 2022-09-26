import torch
import random
import argparse
import learners
import dataloaders
import numpy as np
from dataloaders.utils import *

def run(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

    if args.dataset == 'CIFAR10':
        Dataset = dataloaders.iCIFAR10
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
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
    
    train_dataset = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = True,
                            download=True, transform=train_transform, l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    train_dataset_ul = Dataset(args.dataroot, args.dataset, args.labeled_samples, args.unlabeled_task_samples, train=True, lab = False,
                            download=True, transform=train_transform, l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)
    test_dataset  = Dataset(args.dataroot, args.dataset, train=False,
                            download=False, transform=test_transform, l_dist=args.l_dist, ul_dist=args.ul_dist,
                            tasks=tasks, seed=seed, rand_split=args.rand_split, validation=args.validation, kfolds=args.repeat)

    learner_config = {'num_classes': num_classes,}
    learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](learner_config)
    learner.test()

    # in case tasks reset...
    tasks = train_dataset.tasks
    max_task = len(tasks)
    print(max_task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-Supervised Continual Learning')
    parser.add_argument('--seed', type=int, default=0, help='Set seed (default=0)')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="CIFAR10|CIFAR100")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    # labeled_samples init value = 50000
    parser.add_argument('--labeled_samples', type=int, default=10000, help='Number of labeled samples in ssl')
    # unlabeled_task_samples init value = 0
    parser.add_argument('--unlabeled_task_samples', type=int, default=-1, help='Number of unlabeled samples in each task in ssl')
    parser.add_argument('--l_dist', type=str, default='super', help="vanilla|super")
    parser.add_argument('--ul_dist', type=str, default=None, help="none|vanilla|super - if none, copy l dist")
    parser.add_argument('--rand_split', default=False, action='store_true', help="Randomize the classes in splits")
    parser.add_argument('--validation', default=False, action='store_true', help='Evaluate training dataset rather than testing data')
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    # learner_type init value = 'default'
    parser.add_argument('--learner_type', type=str, default='tiny_model', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='SSCL', help="The class name of learner")
    
    args = parser.parse_args()

    run(args)