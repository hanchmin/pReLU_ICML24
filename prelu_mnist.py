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
from load_mnist_datasets import load_mnist, load_fmnist, load_kmnist
import time
import sys
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    n_class = config['train']['n_class']
    dataset = config['train']['dataset']
    p_list = config['train']['p_list']
    rep = config['train']['repeat']

    epsilon_list = np.linspace(*config['train']['epsilon_list'])
    if config['train']['epsilon_list_short'] is not None:
        epsilon_list_short = np.linspace(*config['train']['epsilon_list_short'])
        flag = True
    
    batch_size = config['train_opt']['batch_size']
    max_epoch = config['train_opt']['max_epoch']
    lr = config['train_opt']['lr']

    h = config['train_net']['hidden_width']
    bias = config['train_net']['bias']

    
    if dataset == 'mnist':
        load_data = load_mnist
    elif dataset == 'fmnist':
        load_data = load_fmnist
    elif dataset == 'kmnist':
        load_data = load_kmnist
    else:
        raise Exception("unknown dataset")

    train_data_loader_batch, train_data_loader_full, _, test_data_loader_full, centers, n_train, n_test, input_dim = load_data(
        batch_size, n_class)

    train_record = {keys: torch.zeros(rep, len(p_list), max_epoch)  for keys in ['train_loss','train_acc','valid_acc','hd1_feature_sr','W_sr']}
    train_record['adv_robustness_acc'] = torch.zeros(rep, len(p_list), len(epsilon_list))
    if flag:
        robust_acc_list_temp= np.zeros((3, len(epsilon_list_short), rep))
    
    for i, p in enumerate(p_list):

        for r in range(rep):

            logging.info(f"Training pReLU network with p={p}. Repeat: {r+1}")

            model_kwargs = dict(center=centers[0], input_dim=input_dim, hidden_dim=h, output_dim=n_class, bias=bias)
            model = ShallowPreluNet(**model_kwargs).to(device)

            opt = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(max_epoch):

                with torch.no_grad():
                    xb, yb = next(iter(train_data_loader_full))
                    pred, hidden_feature = model(xb.to(device), p=p, return_feature=True)
                    hd = hidden_feature.detach().cpu()
                    W = model.prelulayer.W.detach().cpu()
                    
                    loss = F.cross_entropy(pred, yb.to(device))
                    train_acc_val = (torch.argmax(model(xb.to(device), p=p), 1) == yb.to(device)).float().mean()
    

                    xb, yb = next(iter(test_data_loader_full))
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
                    
                for xb, yb in train_data_loader_batch:
                    loss = F.cross_entropy(model(xb.to(device), p=p), yb.to(device))

                    loss.backward()
                    opt.step()
                    opt.zero_grad()

            for j, eps in enumerate(epsilon_list):
                adversary = AutoAttack(lambda xb: model(xb.to(device), p=p), norm='Linf', eps=eps, version='standard', log_path=f'{config["exp_dir"]}/att_log.txt')
                adversary.attacks_to_run = ['apgd-ce']
                xb, yb = next(iter(test_data_loader_full))
                x_adv = adversary.run_standard_evaluation(xb, yb, bs=10000)
                train_record['adv_robustness_acc'][r, i, j] = (torch.argmax(model(x_adv.to(device), p=p), 1) == yb.to(device)).float().mean()

            if flag:
                for fi, eps in enumerate(epsilon_list_short):
                    for fj, norm in enumerate(['Linf', 'L2' , 'L1']):
                        if norm=='Linf':
                            scale = 1.
                        if norm=='L2':
                            scale = 20.
                        if norm=='L1':
                            scale = 100.
                        adversary = AutoAttack(lambda xb: model(xb.to(device), p=p), norm=norm, eps=eps*scale, version='standard', log_path=f'{config["exp_dir"]}/att_log.txt')
                        adversary.attacks_to_run = ['apgd-ce']
                        xb, yb = next(iter(test_data_loader_full))
                        x_adv = adversary.run_standard_evaluation(xb, yb, bs=10000)
                        robust_acc_list_temp[fj,fi,r] = (torch.argmax(model(x_adv.to(device), p=p), 1) == yb.to(device)).float().mean()

            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()


        if flag:
            with open(f'{config["exp_dir"]}/robust_acc_log.txt', 'a') as f:
                with np.printoptions(precision=3, suppress=True):
                    print(f"robust accuracy for p={p}", file=f)
                    print(f"epsilon_list {epsilon_list_short}",file=f)
                    print(np.mean(robust_acc_list_temp, axis=2), file=f)
                    print(np.std(robust_acc_list_temp, axis=2), file=f)


    torch.save(train_record, f'{config["save_dir"]}/train_record.th')
    torch.save({'p_list': p_list,
               'epsilon_list': epsilon_list,
               'dataset': dataset}, f'{config["save_dir"]}/experiment_info.th')

    return

def generate_plot(config):

    train_record = torch.load(f'{config["save_dir"]}/train_record.th')
    experiment_info = torch.load(f'{config["save_dir"]}/experiment_info.th')
    
    p_list = experiment_info['p_list']
    epsilon_list = experiment_info['epsilon_list']
    dataset = experiment_info['dataset']
    max_epoch = train_record['train_loss'].size(-1)
    
    if dataset == 'mnist':
        load_data = load_mnist
    elif dataset == 'fmnist':
        load_data = load_fmnist
    elif dataset == 'kmnist':
        load_data = load_kmnist
    else:
        raise Exception("unknown dataset")
    
    train_loss = train_record['train_loss'].numpy()
    train_acc = train_record['train_acc'].numpy()
    valid_acc = train_record['valid_acc'].numpy()
    hd1_feature_sr = train_record['hd1_feature_sr'].numpy()
    adv_robustness_acc = train_record['adv_robustness_acc'].numpy()
    
    _, train_data_loader_full, _, test_data_loader_full, centers, n_train, n_test, input_dim = load_data(
        1000, 10)
    xb0, yb0 = next(iter(train_data_loader_full))
    xb, yb = next(iter(train_data_loader_full))
    xb = xb[:10000]
    yb = yb[:10000]
    _, idx = torch.sort(yb)
    xb_sorted = xb[idx]-torch.mean(xb0,dim=0,keepdim=True)
    xb_sorted = xb_sorted /torch.norm(xb_sorted, dim=1, keepdim=True)
    
    prod = torch.matmul(xb_sorted,xb_sorted.t())
    prd = prod.detach().numpy()
    
    label_font_size = 26
    title_font_size = 24
    ax_tick_font_size = 26
    ax_label_font_size = 26
    
    # last layer feature
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']
    
    # first plot, training stats
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
    axs[1].set_ylim(np.min([train_acc_max - 0.05,valid_acc_max - 0.05]), np.min([1, train_acc_max + 0.05]))
    hd1_feature_sr_max = np.max(hd1_feature_sr)
    axs[2].set_ylim(0, hd1_feature_sr_max + 6)
    axs[3].set_ylim(0, 1)
    
    axs[1].set_xlabel('Epoch', fontsize=label_font_size)
    axs[2].set_xlabel('Epoch', fontsize=label_font_size)
    axs[3].set_xlabel('$\ell_\infty$ Attack Radius (in pixel space)', fontsize=label_font_size)
    
    # plt.sca(axs[0])
    # axs[0].pcolormesh(prd, cmap='bwr', vmin=-1, vmax=1)
    # axs[0].colorbar()
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
    plt.title(f'(a) \n Correlation among \n {dataset} data (centered)', fontsize=title_font_size)
    
    plt.subplots_adjust(left=0.05, right=0.993, bottom=0.136, top=0.852, wspace=0.221, hspace=0.4)
    axs[1].legend()
    # plt.show()
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

