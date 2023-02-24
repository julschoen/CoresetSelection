import argparse
import torch
from torchvision import datasets, transforms
from train import Trainer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # General Training
    parser.add_argument('--batch', type=int, default= 100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--niter', type=int, default=20000, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--lrIms', type=float, default=5e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument("--train_batch", "-tb", default=None, type=int,
                     help="batch size for training, if not specified, it will equal to batch size in argument --batch")
    parser.add_argument("--selection_batch", "-sb", default=None, type=int,
                     help="batch size for selection, if not specified, it will equal to batch size in argument --batch")
    
    # Selecting
    parser.add_argument("--selection_epochs", "-se", default=40, type=int,
                        help="number of epochs whiling performing selection on full dataset")
    parser.add_argument('--selection_momentum', '-sm', default=0.9, type=float, metavar='M',
                        help='momentum whiling performing selection (default: 0.9)')
    parser.add_argument('--selection_weight_decay', '-swd', default=5e-4, type=float,
                        metavar='W', help='weight decay whiling performing selection (default: 5e-4)',
                        dest='selection_weight_decay')
    parser.add_argument('--selection_optimizer', "-so", default="SGD",
                        help='optimizer to use whiling performing selection, e.g. SGD, Adam')
    parser.add_argument("--selection_nesterov", "-sn", default=True, type=bool,
                        help="if set nesterov whiling performing selection")
    parser.add_argument('--selection_lr', '-slr', type=float, default=0.1, help='learning rate for selection')
    parser.add_argument("--selection_test_interval", '-sti', default=1, type=int, help=
    "the number of training epochs to be preformed between two test epochs during selection (default: 1)")
    parser.add_argument("--selection_test_fraction", '-stf', type=float, default=1.,
             help="proportion of test dataset used for evaluating the model while preforming selection (default: 1.)")
    parser.add_argument('--balance', default=True, type=bool,
                        help="whether balance selection is performed per class")
    args = parser.parse_args()

    args.channel, args.im_size, args.num_classes = 3, (32,32), 10
    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ])
    dataset = datasets.CIFAR10('../data/', train=True, download=True, transform=transform)
    print(dataset)
    trainer = Trainer(args, dataset)
    trainer.train()
    

if __name__ == '__main__':
    main()