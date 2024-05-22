import argparse
import logging
import os
import gc
import pprint
import yaml

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from prelu_torch_modules import ShallowPreluNet
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import time
import sys
from aux import y_transform
from autoattack import AutoAttack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--exp_name', type=str, default='unnamed_exp')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    return {**config_data, **vars(args)}
    

def run_experiment(config):

    
    img_batch_size = 50

    batch_size = config['train_opt']['batch_size']
    max_epoch = config['train_opt']['max_epoch']
    lr = config['train_opt']['lr']
    
    h = config['train_net']['hidden_width']
    bias = config['train_net']['bias']
    
    p_list = config['train']['p_list']
    n_class = config['train']['n_class']
    rep = config['train']['repeat']
    epsilon_list = np.linspace(*config['train']['epsilon_list'])
    pretrained_net_name = config['train']['pretrained_net_name']
    prelu_prob_name = 'ShallowPreluNet'
    dir = './data'
    dataset_module = datasets.Caltech256
    
    pretrained_net = getattr(models, pretrained_net_name)
    
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # load datasets, preprocessing images
    img_t = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(dim=0) == 1 else x),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])

    dataset = dataset_module(root=dir, transform=img_t)
    datalist = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0))
    n_train = len(datalist[0])
    n_test = len(datalist[1])
    logging.info(f"{n_train} samples in training set and {n_test} samples in test set")

    train_data_img = DataLoader(datalist[0], batch_size=img_batch_size, shuffle=False, num_workers=8)
    test_data_img = DataLoader(datalist[1], batch_size=img_batch_size, shuffle=False, num_workers=8)
    
    resnet_prelu = pretrained_net(weights='DEFAULT')
    for param in resnet_prelu.parameters():
        param.requires_grad = False

    fc_inputs = resnet_prelu.fc.in_features
    resnet_prelu.fc = nn.Sequential(
        nn.Identity()
    )
    
    resnet_prelu = resnet_prelu.to(device)
    
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    
    for xb, yb in train_data_img:
        if x_train == None:
            x_train = resnet_prelu(xb.to(device))
            y_train = y_transform(yb, n_class)
        else:
            x_train = torch.cat((x_train, resnet_prelu(xb.to(device))), dim=0)
            y_train = torch.cat((y_train, y_transform(yb, n_class)), dim=0)
    
    for xb, yb in test_data_img:
        if x_test == None:
            x_test = resnet_prelu(xb.to(device))
            y_test = y_transform(yb, n_class)
        else:
            x_test = torch.cat((x_test, resnet_prelu(xb.to(device))), dim=0)
            y_test = torch.cat((y_test, y_transform(yb, n_class)), dim=0)
    
    center = torch.mean(x_train, dim=0, keepdim=True)
    
    resnet_prelu.cpu()
    del resnet_prelu
    gc.collect()
    torch.cuda.empty_cache()

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    
    train_data_feature = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_data_loader_full = DataLoader(train_data, batch_size=n_train, shuffle=False)

    test_data_feature = DataLoader(test_data, batch_size=n_test, shuffle=False)

    train_record = {keys: torch.zeros(rep, len(p_list), max_epoch)  for keys in ['train_loss','train_acc','valid_acc','hd1_feature_sr','W_sr']}
    train_record['adv_robustness_acc'] = torch.zeros(rep, len(p_list), len(epsilon_list))

    for i, p in enumerate(p_list):

        for r in range(rep):
            
            model = ShallowPreluNet(center=center, input_dim=fc_inputs, hidden_dim=h, output_dim=n_class).to(
                        device)
            
            opt = optim.Adam(model.parameters(), lr=lr)
    
            for epoch in range(max_epoch):
    
                with torch.no_grad():
                    xb, yb = next(iter(train_data_loader_full))
                    pred, hidden_feature = model(xb.to(device), p=p, return_feature=True)
                    hd = hidden_feature.detach().cpu()
                    W = model.prelulayer.W.detach().cpu()
                    
                    loss = F.cross_entropy(pred, yb.to(device))
                    train_acc_val = (torch.argmax(model(xb.to(device), p=p), 1) == yb.to(device)).float().mean()
    
    
                    xb, yb = next(iter(test_data_feature))
                    valid_acc_val = (torch.argmax(model(xb.to(device), p=p), 1) == yb.to(device)).float().mean()
                    
                    train_record['train_loss'][r, i, epoch] = loss.detach().cpu()
                    train_record['train_acc'][r, i, epoch] = train_acc_val.detach().cpu()
                    train_record['valid_acc'][r, i, epoch] = valid_acc_val.detach().cpu()
                    train_record['hd1_feature_sr'][r, i, epoch] = (torch.linalg.norm(hd, ord='fro') / torch.linalg.norm(hd,
                                                                                                      ord=2)) ** 2
                    train_record['W_sr'][r, i, epoch] = (torch.linalg.norm(W, ord='fro') / torch.linalg.norm(W,
                                                                                           ord=2)) ** 2
    
                    if epoch%10==0:
                        logging.info(f"Epoch {epoch}: train acc {train_acc_val.detach().cpu().numpy()}, test acc {valid_acc_val.detach().cpu().numpy()}")
            
                for xb, yb in train_data_feature:
                    pred = model(xb.to(device), p, True)
                    loss = F.cross_entropy(pred, yb.to(device))
            
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0, norm_type=2)
                    opt.step()
                    opt.zero_grad()

    
            for j, eps in enumerate(epsilon_list):
                xb, yb = next(iter(test_data_feature))
                x_max, x_min = torch.max(xb), torch.min(xb)
                scale = (x_max-x_min+10.0)
                xb = (xb - x_min + 5.0)/scale
                adversary = AutoAttack(lambda xb: model((xb*scale+x_min-5.0), p=p), norm='Linf', eps=eps/scale, version='standard', log_path=f'{config["exp_dir"]}/att_log.txt')
                adversary.attacks_to_run = ['apgd-ce']
                x_adv = adversary.run_standard_evaluation(xb.to(device), yb.to(device), bs=n_test)
                train_record['adv_robustness_acc'][r, i, j] = (torch.argmax(model((x_adv*scale+x_min-5.0).to(device), p=p), 1) == yb.to(device)).float().mean()
    
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
    torch.save(train_record, f'{config["save_dir"]}/train_record.th')
    torch.save({'p_list': p_list,
               'epsilon_list': epsilon_list}, f'{config["save_dir"]}/experiment_info.th')

    return

def generate_plot(config):

    train_record = torch.load(f'{config["save_dir"]}/train_record.th')
    experiment_info = torch.load(f'{config["save_dir"]}/experiment_info.th')
    
    p_list = experiment_info['p_list']
    epsilon_list = experiment_info['epsilon_list']
    max_epoch = train_record['train_loss'].size(-1)
    
    train_loss = train_record['train_loss'].numpy()
    train_acc = train_record['train_acc'].numpy()
    valid_acc = train_record['valid_acc'].numpy()
    hd1_feature_sr = train_record['hd1_feature_sr'].numpy()
    adv_robustness_acc = train_record['adv_robustness_acc'].numpy()

    npzfile = np.load('icml24_caltech256_data_corr_prd.npz')
    prd = npzfile['prd']
    
    label_font_size = 26
    title_font_size = 24
    ax_tick_font_size = 26
    ax_label_font_size = 26

    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']

    fig, axs = plt.subplots(1, 4, figsize=(32, 8))
    
    plot_title = ['(b) \n Accuracy, \n Train (Dashed) and Test (Solid)', '(c) \n Stable Rank \n of Hidden Feature', '(d) \n Robust Accuracy \n under L-infinity PGD Attack']
    
    for i, ax in enumerate([axs[1], axs[2], axs[3]]):
        plt.sca(ax)
        plt.rcParams.update({
            "font.size": label_font_size})
        plt.tick_params(labelsize=ax_tick_font_size)
        plt.title(plot_title[i],fontsize=title_font_size)
        # ax.set_xlim(left=0)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, 4))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.5)
    
    train_acc_max = np.max(train_acc)
    valid_acc_max = np.max(valid_acc)
    axs[1].set_ylim(np.min([train_acc_max - 0.2,valid_acc_max - 0.2]), np.min([1, train_acc_max + 0.05]))
    hd1_feature_sr_max = np.max(hd1_feature_sr)
    axs[2].set_ylim(0, hd1_feature_sr_max + 6)
    axs[3].set_ylim(0, 1)
    
    axs[1].set_xlabel('Epoch', fontsize=label_font_size)
    axs[2].set_xlabel('Epoch', fontsize=label_font_size)
    axs[3].set_xlabel('$\ell_\infty$ Attack Radius (in feature space)', fontsize=label_font_size)
    
    for i, p in enumerate(p_list):
    
        axs[1].plot(np.arange(max_epoch), np.mean(train_acc[:,i,:],axis=0), color=color[i], linestyle='--',label="".format(p), linewidth=2.5)
        axs[1].fill_between(np.arange(max_epoch), np.min(train_acc[:,i,:],axis=0), np.max(train_acc[:,i,:],axis=0),
                            color=color[i], alpha=0.2)
    
        axs[1].plot(np.arange(max_epoch), np.mean(valid_acc[:,i,:],axis=0), color=color[i], label="p={:.1f}".format(p), linewidth=2.5)
        axs[1].fill_between(np.arange(max_epoch), np.min(valid_acc[:,i,:],axis=0), np.max(valid_acc[:,i,:],axis=0),
                            color=color[i], alpha=0.2)
    
        axs[3].plot(epsilon_list, np.mean(adv_robustness_acc[:,i,:],axis=0), color=color[i], label="p={:.1f}".format(p), linewidth=2.5)
        axs[3].fill_between(epsilon_list, np.min(adv_robustness_acc[:,i,:],axis=0),
                            np.max(adv_robustness_acc[:,i,:],axis=0),
                            color=color[i], alpha=0.2)
    
        axs[2].plot(np.arange(max_epoch), np.mean(hd1_feature_sr[:,i,:],axis=0), color=color[i],
                    label="p={:.1f}".format(p), linewidth=2.5)
        axs[2].fill_between(np.arange(max_epoch), np.min(hd1_feature_sr[:,i,:],axis=0),
                            np.max(hd1_feature_sr[:,i,:],axis=0),
                            color=color[i], alpha=0.2)
    
    plt.sca(axs[0])
    plt.rcParams.update({
        "font.size": label_font_size})
    plt.tick_params(labelsize=ax_tick_font_size)
    # ax.set_xlim(left=0)
    for axis in ['top', 'bottom', 'left', 'right']:
        axs[0].spines[axis].set_linewidth(2.5)
    img = plt.pcolormesh(prd, cmap='bwr', vmin=-1, vmax=1) # or any cmap (see plt.cm; plt.cm.Blues etc)
    plt.colorbar(img, shrink=0.7)
    plt.title('(a) \n Correlation among extracted feature \n of Caltech256 dataset (centered)', fontsize=title_font_size)
    
    plt.subplots_adjust(left=0.05, right=0.993, bottom=0.136, top=0.852, wspace=0.221, hspace=0.4)
    axs[1].legend()
    plt.savefig(f'{config["save_dir"]}/plot.png', transparent=False, facecolor='white')
    plt.close()

    return

if __name__ == "__main__":
    config = parse_args()
    config['root_dir'] = os.path.dirname(os.path.abspath(__file__))
    config['exp_dir'] = os.path.join(config['root_dir'], 'exps', config['exp_name'])

    os.makedirs(config['exp_dir'], exist_ok=True)
    logging.basicConfig(filename=f'{config["exp_dir"]}/log.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Configurations:\n%s", pprint.pformat(config, indent=4))

    config['save_dir'] = os.path.join(config['exp_dir'], 'saves')
    os.makedirs(config['save_dir'], exist_ok=True)

    if config['train']['run']:
        run_experiment(config)
    if config['plot']['run']:
        generate_plot(config)
