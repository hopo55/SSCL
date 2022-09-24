import argparse
import dataloaders

def run(args):
    if args.dataset == 'CIFAR10':
        Dataset = dataloaders.iCIFAR10
    elif args.dataset == 'CIFAR100':
        Dataset = dataloaders.iCIFAR100
    else:
        print('Dataset does not select.')
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-Supervised Continual Learning')
    parser.add_argument('--seed', type=int, default=0, help='Set seed (default=0)')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="CIFAR10|CIFAR100")
    
    args = parser.parse_args()

    run(args)