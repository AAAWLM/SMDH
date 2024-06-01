import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import random

from torch.optim import Adam
from loss import TripletMarginLoss, DiceLoss
from torch.utils.data import DataLoader
from read_dataset import ChestXrayDataSet
from PK_sampler import PKSampler
from model import Extractor

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train_epoch(model, dataloader, optimizer,criterion1, criterion2, criterion3, device):
    model.train()
    trainloss = 0.0
    # criterion1 = nn.CrossEntropyLoss() ### Classification
    # criterion2 = TripletMarginLoss(margin=args.margin) ### Tripletloss
    # criterion3 = DiceCeLoss() ### Segmentation
    for i, data in enumerate(dataloader):
        samples, masks, targets = data[1].to(device), data[2].to(device), data[3].to(device)
        hash_code, hash_bits, label, segs = model(samples)
        loss1 = criterion1(label, targets)
        loss2 = criterion2(hash_bits, targets)
        loss3 = criterion3(segs, masks)
        loss = 0.6*loss1 + loss2 + loss3

        # Backward pass
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
        trainloss += loss.item()
        Aloss = 100 * trainloss /len(dataloader.dataset)
    return Aloss


def evaluate_epoch(model, dataloader, criterion1, criterion2, criterion3,  device):
    model.eval()
    trainloss = 0.0
    with torch.no_grad():
        for data in dataloader:
            samples, masks, targets = data[1].to(device), data[2].to(device), data[3].to(device)
            hash_code, hash_bits, label, segs = model(samples)
            loss1 = criterion1(label, targets)
            loss2 = criterion2(hash_bits, targets)
            loss3 = criterion3(segs, masks)

            loss = 0.6 * loss1 + loss2  + loss3
            trainloss += loss.item()
        Aloss = 100 * trainloss / len(dataloader.dataset)
    return Aloss


def test(model,dataloader,device):
    model.eval()
    model.to(device)
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()

    with torch.no_grad():
        for i in dataloader:
            inputs, targets = i[1].to(device), i[3].to(device)
            hash_code, hash_bits, label, seg= model(inputs)

            full_batch_output = torch.cat((full_batch_output, hash_bits.data), 0)
            full_batch_label = torch.cat((full_batch_label, targets.data), 0)

        test_binary = torch.sign(full_batch_output)
        test_label = full_batch_label

    test_binary = test_binary.cpu().numpy()
    tst_binary = np.asarray(test_binary, np.int32)
    tst_label = test_label.cpu().numpy()

    query_times = test_binary.shape[0]
    len1 = tst_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, len1 + 1)
    sum_tp = np.zeros(len1)
    sum_r = np.zeros(len1)

    _dists,_labels =[], []
    for i in range(query_times):
        query_label = tst_label[i]
        query_binary = test_binary[i, :]

        query_result = np.count_nonzero(query_binary != tst_binary, axis=1)  # Hamming diatance
        sort_indices = np.argsort(query_result) #sort   ind

        buffer_yes = np.equal(query_label, tst_label[sort_indices]).astype(int) #label equal  ##

        Recall = np.cumsum(buffer_yes)/np.sum(buffer_yes)
        P = np.cumsum(buffer_yes) / Ns
        precision_radius[i] = P[np.where(np.sort(query_result) > 2)[0][0]]
        AP[i] = np.sum(P * buffer_yes) / sum(buffer_yes)
        sum_tp = sum_tp + np.cumsum(buffer_yes)

        _dists.append(query_result)
        _labels.append(query_label)
        sum_r = sum_r+Recall


    _dists = torch.Tensor(_dists)
    _labels = torch.Tensor(_labels)


    kappas = [0, 1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # kappas
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    precision_at_k = sum_tp / Ns / query_times
    recall_at_k = sum_r/query_times
    print('recall=', recall_at_k[kappas])
    print('precision at k:', precision_at_k[kappas])
    map = np.mean(AP)
    print('mAP:', map)


def save(model, epoch, save_dir, margin, lr, batch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = '_' + args.model
    file_name += '_seed_'+str(args.seed)+'_epoch_'+str(epoch)+'_m_'+str(margin) \
                 +'_lr_'+str(args.lr)+'_batch_'+str(batch)+'_ckpt.pth'
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def main(args):
    start_time = time.time()
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p = args.labels_per_batch
    k = args.samples_per_label
    batch_size = p * k

    model = Extractor(num_feature=args.feature, hash_bits=args.hash, type_bits=args.type)
    model.to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = TripletMarginLoss(margin=args.margin)
    criterion3 = DiceLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)


    ### Dataset
    train_dataset = ChestXrayDataSet(data_dir=os.path.join(args.train_dir, 'images'),
                                     masks_dir=os.path.join(args.train_dir, 'masks'),
                                     image_list_file=args.train_image_list)

    targets = train_dataset.labels
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=PKSampler(targets, p, k))

    val_dataset = ChestXrayDataSet(data_dir=os.path.join(args.val_dir, 'images'),
                                   masks_dir=os.path.join(args.val_dir, 'masks'),
                                   image_list_file=args.val_image_list)

    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size,
                            shuffle=False)
    test_dataset = ChestXrayDataSet(data_dir=os.path.join(args.test_dir, 'images'),
                                     masks_dir=os.path.join(args.test_dir, 'masks'),
                                     image_list_file=args.test_image_list)

    test_loader = DataLoader(test_dataset,batch_size=args.eval_batch_size,
                             shuffle=False)

    print('Testing...')
    test(model, test_loader, device)

    _loss_train, _loss_evaluate =[], []
    for epoch in range(1, args.epochs+1):
        print('\n Training...')
        train_loss = train_epoch(model, train_loader, optimizer, criterion1, criterion2, criterion3, device)
        _loss_train.append(train_loss)
        print(' EPOCH {}/{} \t train loss {:.3f}'.format(epoch, args.epochs, train_loss))

        print('Evaluating...')
        evaluate_loss = evaluate_epoch(model, val_loader, criterion1, criterion2, criterion3, device)
        _loss_evaluate.append(evaluate_loss)
        print(' EPOCH {}/{} \t evaluate loss {:.3f}'.format(epoch, args.epochs, evaluate_loss))
        print('Testing_val...')
        test(model, val_loader, device)

        print('Testing...')
        test(model, test_loader, device)


    print('\n Saving...')
    save(model, epoch, args.save_dir, args.margin, args.lr, args.eval_batch_size)
    end_time = time.time()
    run_time = end_time - start_time
    print('>>Time: {:.4f}'.format(run_time))


# 记得要把所有的参数都换成一样的
def parse_args():  #这些都是模型需要的参数
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')
    parser.add_argument('--model', default='SMDH',
                        help='')
    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid)')
    parser.add_argument('--train-dir', default=r'D:\COVID\Train',
                        help='Train dataset directory path')
    parser.add_argument('--test-dir', default=r'D:\COVID\Test',
                        help='Test dataset directory path')
    parser.add_argument('--val-dir', default=r'D:\COVID\Val',
                        help='Val dataset directory path')
    parser.add_argument('--train-image-list', default='./Training.txt',
                        help='Train image list')
    parser.add_argument('--test-image-list', default='./Test.txt',
                        help='Test image list')
    parser.add_argument('--val-image-list', default='./Val.txt',
                        help='Val image list')
    parser.add_argument('-p', '--labels-per-batch', default=3, type=int,
                        help='Number of unique labels/classes per batch')
    parser.add_argument('-k', '--samples-per-label', default=16, type=int,
                        help='Number of samples per label in a batch')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='Number of training epochs to r0un')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='Triplet loss margin')
    parser.add_argument('--lr', default=0.0001,
                        type=float, help='Learning rate')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('--feature',  default=1024, type=int)
    parser.add_argument('--hash', default=128, type=int)
    parser.add_argument('--type', default=3, type=int)
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='Model save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)