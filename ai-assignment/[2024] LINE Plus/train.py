import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from infobatch import InfoBatch
from torchvision import transforms
from models import *
import random
import numpy as np
import pickle

# =========================================== Arguments ===========================================
def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='train the face recognition network')
    parser.add_argument('--config_file', default="sample_config", help='name of config file without file extension')

    args = parser.parse_args()

    return args
# ==================================================================================================


# ============================================ Training ============================================
def train(args):
    """ Overall training session based on configurations """
  
    # Device setting for CUDA
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if args.model.lower() == 'r18':
        model = ResNet18(num_classes=10)
    elif args.model.lower() == 'r50':
        model = ResNet50(num_classes=10)
    elif args.model.lower() == 'r101':
        model = ResNet101(num_classes=10)
    else:
        model = ResNet50(num_classes=10)

    # Build model, criterion
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none').to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data loading
    stats = ((0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
           (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, transform=train_transform, download=True)

    # InfoBatch dataset.
    if args.use_info_batch:
        print("InfoBatch is used.")
        trainset = InfoBatch(trainset, args.num_epoch, args.ratio, args.delta)

    else:
        print("InfoBatch is not used.")

    # Set InfoBatch Sampler if InfoBatch is used.
    sampler = None
    train_shuffle = True
    if args.use_info_batch:
        sampler = trainset.sampler
        train_shuffle = False
    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=test_transform)

    # DataLoader
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=0, sampler=sampler)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Optimizer : SGD
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, steps_per_epoch=len(trainloader),
                                                    epochs=args.num_epoch, div_factor=args.div_factor,
                                                    final_div_factor=args.final_div, pct_start=args.pct_start)

    # List for saving the results
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # Scaler to use mix precision training to speed up the training / save memory
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    def train_info_batch(epoch):
        # Train with InfoBatch
        print(f'Epoch: {epoch}, iterations {len(trainloader)}')
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for index, data in trainloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(args.fp16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Set the active indices for InfoBatch
                trainset.cur_batch_index = index
                # Update the weights and scores for current batch samples
                loss = trainset.update(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[Epoch : {epoch}] Training Acc : {correct/total:.2%} Train Loss : {train_loss/len(trainloader):.4f}')
        train_losses.append(train_loss/len(trainloader))
        train_accs.append(100*correct/total)


    def train_normal(epoch):
        # Train without InfoBatch
        print(f'Epoch: {epoch}, iterations {len(trainloader)}')
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(args.fp16):
                outputs = model(inputs)
                loss = torch.mean(criterion(outputs, targets))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[Epoch : {epoch}] Training Acc : {correct/total:.2%} Train Loss : {train_loss/len(trainloader):.4f}')
        train_losses.append(train_loss/len(trainloader))
        train_accs.append(100*correct/total)

    def test(epoch):
        # Test
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        global best_acc
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = test_criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(f'[Epoch : {epoch}] Test Acc : {correct/total:.2%} Test Loss : {test_loss/len(testloader):.4f}')
        if 100*correct/total > best_acc:
            best_acc = 100*correct/total
        test_losses.append(test_loss/len(testloader))
        test_accs.append(100*correct/total)

    # To calculate the total training time
    total_time = 0
    for epoch in range(args.num_epoch):
        # Epoch-based implementation
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                        steps_per_epoch=len(trainloader),
                                                        epochs=args.num_epoch, div_factor=args.div_factor,
                                                        final_div_factor=args.final_div, pct_start=args.pct_start,
                                                        last_epoch=epoch * len(trainloader) - 1)
        end = time.time()
        train_info_batch(epoch) if args.use_info_batch else train_normal(epoch)
        total_time += time.time() - end
        test(epoch)

    if args.use_info_batch:
        print('Total saved sample forwarding: ', trainset.get_pruned_count())
    print(f'Total training time: {total_time}')
    print(f'Best Accuracy : {best_acc}')

    # Save the training results
    save_info = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs,
        'best_acc': best_acc,
        'total_time': total_time
    }
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/results_{args.expname}.pkl", "wb") as f:
        pickle.dump(save_info, f, pickle.HIGHEST_PROTOCOL)

# ============================================== Main ==============================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR10')
    # training configuration
    parser.add_argument('--expname', default="", help='experiment name')
    parser.add_argument('--seed', default=0, type=int, help='random seed for reproduction')
    parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--use_info_batch', action='store_true', help='whether use info batch or not.')
    parser.add_argument('--fp16', action='store_true', help='use mix precision training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N', help='input batch size for testing (default: 128)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W', help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # onecycle scheduling arguments
    parser.add_argument('--max-lr', default=0.05, type=float)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--num_epoch', default=200, type=int, help='training epochs')
    parser.add_argument('--pct-start', default=0.3, type=float)
    parser.add_argument('--shuffle', default=True, action='store_true')
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio')
    parser.add_argument('--delta', default=0.875, type=float)
    parser.add_argument('--model', default='r18', type=str)
    args = parser.parse_args()
   
    # Seed for reproductivity
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(args)
# ==================================================================================================
