import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryRecall, BinarySpecificity, BinaryAccuracy, BinaryAUROC
from torchvision.utils import save_image
from tqdm import tqdm

from models.thick_net import ThickNet
from models.resunet import ResUnet
from models.refine import denosing_module
from tools.dl import fundus_data
from tools.helpfunc import save_checkpoint
from tools.loss import *
from torch import nn


def train(model1, model2, refine, criterion1, criterion2, criterion3,
          optimizer1, optimizer2, optimizer3, device, train_loader,
          test_loader, epochs, root, dataname):

    model1.zero_grad()
    model2.zero_grad()
    refine.zero_grad()
    best_f1 = 0
    best_acc = 0
    best_sen = 0
    best_spe = 1
    best_auc = 0
    running = True
    epoch = 0
    test_writer = SummaryWriter(root + '/' + dataname)

    while epoch <= epochs and running:
        model1.train()
        model2.train()
        refine.train()
        train_loss = 0
        pbar = tqdm(train_loader,
                    colour='#5181D5',
                    desc="Epoch:{}".format(epoch),
                    dynamic_ncols=True,
                    ncols=100)

        for index, (img, thin, thick, mask) in enumerate(pbar):
            #三个图片，原图，细血管，粗血管
            inputs, thin_labels, thick_labels, all_label = img.to(
                device), thin.to(device), thick.to(device), mask.to(device)

            # 粗血管部分进行分割
            thick_pred = model2(inputs)
            thick_loss = criterion2(thick_pred, thick_labels)

            # 细血管的分割
            thin_pred = model1(torch.cat([inputs, thick_pred], dim=1))
            thin_loss = criterion1(thin_pred, thin_labels)

            # 第一阶段的初始预测值
            First_pred = thin_pred + thick_pred

            Second_pred = refine(First_pred)
            all_loss = criterion3(Second_pred, all_label)

            total_loss = thin_loss + thick_loss + all_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            total_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            train_loss += total_loss.item() / len(train_loader)

            pbar.set_postfix({'loss': '{0:1.5f}'.format(train_loss)})
            pbar.update(1)

        # 在测试集合上进行验证
        model1.eval()
        model2.eval()
        refine.eval()

        test_loss = 0
        pbar_test = tqdm(test_loader,
                         colour='#81D551',
                         desc="testing",
                         dynamic_ncols=True)
        for index, (img, thin, thick, mask) in enumerate(pbar_test):
            inputs, thin_labels, thick_labels, all_label = img.to(
                device), thin.to(device), thick.to(device), mask.to(device)

            thick_pred_val = model2(inputs)
            save_image(thick_pred_val,
                       f'visual_output/thick/thick_val_mask_{index + 1}.png')
            thick_loss = criterion2(thick_pred_val, thick_labels)

            thin_pred_val = model1(torch.cat([inputs, thick_pred_val], dim=1))
            save_image(thin_pred_val,
                       f'visual_output/thin/thin_val_mask_{index + 1}.png')
            thin_loss = criterion1(thin_pred_val, thin_labels)

            second_pred_val = thick_pred_val + thin_pred_val

            second_pred_val = refine(second_pred_val)
            save_image(second_pred_val,
                       f'visual_output/all/all_val_mask_{index + 1}.png')
            all_loss = criterion3(second_pred_val, all_label)

            second_pred_val_clone = second_pred_val.clone()
            second_pred_val_clone[second_pred_val_clone > 0.5] = 1
            second_pred_val_clone[second_pred_val_clone <= 0.5] = 0

            total_loss = thin_loss + thick_loss + all_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            total_loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            test_loss += total_loss.item() / len(test_loader)

            spe = BinarySpecificity().to(device)
            f1 = BinaryF1Score().to(device)
            sen = BinaryRecall().to(device)
            acc = BinaryAccuracy().to(device)
            auc = BinaryAUROC().to(device)

        test_writer.add_scalar(tag='Acc',
                               scalar_value=acc(second_pred_val_clone,
                                                all_label).to(device),
                               global_step=epoch)
        test_writer.add_scalar(tag='Sen',
                               scalar_value=sen(second_pred_val_clone,
                                                all_label).to(device),
                               global_step=epoch)
        test_writer.add_scalar(tag='Spe',
                               scalar_value=spe(second_pred_val_clone,
                                                all_label).to(device),
                               global_step=epoch)
        test_writer.add_scalar(tag='F1',
                               scalar_value=f1(second_pred_val_clone,
                                               all_label).to(device),
                               global_step=epoch)
        test_writer.add_scalar(tag='AUC',
                               scalar_value=auc(second_pred_val_clone,
                                                all_label).to(device),
                               global_step=epoch)

        if f1(second_pred_val_clone, all_label).to(device) > best_f1:
            best_f1 = f1(second_pred_val_clone, all_label).to(device)
        if acc(second_pred_val_clone, all_label).to(device) > best_acc:
            best_acc = acc(second_pred_val_clone, all_label).to(device)
        if sen(second_pred_val_clone, all_label).to(device) > best_sen:
            best_sen = sen(second_pred_val_clone, all_label).to(device)
        if spe(second_pred_val_clone, all_label).to(device) < best_spe:
            best_spe = spe(second_pred_val_clone, all_label).to(device)
        if auc(second_pred_val_clone, all_label).to(device) > best_auc:
            best_auc = auc(second_pred_val_clone, all_label).to(device)

        # save_checkpoint(root,
        #                 model_thin=model1,
        #                 model_thick=model2,
        #                 refine_model=refine,
        #                 better=f1(second_pred_val_clone,
        #                           all_label).to(device) == best_f1,
        #                 dataname=dataname)

        print(
            'All_model:\nBest F1:{}---Best ACC:{}---Best Sen:{}---Best Spe:{}---Best AUC:{}'
            .format(best_f1, best_acc, best_sen, best_spe, best_auc))
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_index', help='-1 for cpu', type=int, default=1)
    parser.add_argument('--ckpt_log_dir', type=str, default='ckpts')
    parser.add_argument('--train_data_dir',
                        type=str,
                        default='data/chasedb1/train')
    parser.add_argument('--test_data_dir',
                        type=str,
                        default='data/chasedb1/test')
    parser.add_argument('--test_data_name', type=str, default='chasedb1')
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=140)
    parser.add_argument('--num_works', type=int, default=0)
    args = parser.parse_args()

    if int(args.gpu_index) >= 0 and torch.cuda.is_available():
        device = "cuda:" + str(args.gpu_index)
        print('using device: ', torch.cuda.get_device_name(device))
        print(args)
    else:
        device = "cpu"
        print(args)

    if not os.path.exists(args.ckpt_log_dir + '/' + args.test_data_name + '/'):
        os.makedirs(args.ckpt_log_dir + '/' + args.test_data_name + '/')
        os.makedirs('visual_output/' + args.test_data_name + '/')

    thinmodel = ResUnet(channel=2, device=device)  
    thickmodel = ThickNet(in_channels=1)
    refine = denosing_module(device=device, inchannel=1)

    train_dataset = fundus_data(args.train_data_dir, mode='train',name=args.test_data_name,isnoise=False)
    test_dataset = fundus_data(args.test_data_dir, mode='test',name=args.test_data_name,isnoise=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_works,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             num_workers=args.num_works,
                             shuffle=False)

    optimizer1 = torch.optim.Adam(thinmodel.parameters(),
                                  lr=1e-6,
                                  betas=(0.9, 0.999))
    optimizer2 = torch.optim.Adam(thickmodel.parameters(),
                                  lr=1e-4,
                                  betas=(0.9, 0.999))
    optimizer3 = torch.optim.Adam(refine.parameters(),
                                  lr=1e-4,
                                  betas=(0.9, 0.999))

    
    critetions1 = Focal_IoU(theta=0.5)
    critetions2 = nn.BCELoss()
    critetions3 = EdgeLoss_BCE(device=device, alpha=0.5)

    train(model1=thinmodel.to(device),
          model2=thickmodel.to(device),
          refine=refine.to(device),
          optimizer1=optimizer1,
          optimizer2=optimizer2,
          optimizer3=optimizer3,
          criterion1=critetions1.to(device),
          criterion2=critetions2.to(device),
          criterion3=critetions3.to(device),
          device=device,
          train_loader=train_loader,
          test_loader=test_loader,
          epochs=args.epochs,
          root=args.ckpt_log_dir,
          dataname=args.test_data_name)
