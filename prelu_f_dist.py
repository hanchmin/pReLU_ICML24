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
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import math
from prelu_torch_modules import ShallowPreluNet
from torch_adv_atk import estimate_l_inf_dist, adv_atk_batch_bin_class

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--exp_name', type=str, default='unnamed_exp')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    return {**config_data, **vars(args)}


def generate_data(k=10, k1=5, D=None, n=None, sigma=None):
    centers = np.eye(k)
    label = np.random.randint(k, size=(n,))
    X = np.concatenate([centers[label], np.zeros((n, D - k))], axis=1) + np.random.normal(0, sigma, [n, D])
    Y = np.reshape(1 - 2 * (label >= k1), (n,))

    return torch.tensor(X).float(), torch.tensor(Y).float()

class f1(nn.Module):
    def __init__(self, k=None, k1=None, input_dim=None):
        super().__init__()
        self._k1 = torch.tensor(k1)
        self._k2 = torch.tensor(k - k1)
        self._mu1 = nn.Parameter(torch.tensor(np.concatenate([np.ones((k1, 1))/np.sqrt(k1), np.zeros((input_dim - k1, 1))], axis=0)).float())
        self._mu2 = nn.Parameter(torch.tensor(np.concatenate([np.zeros((k1, 1)), np.ones((k-k1, 1))/np.sqrt(k-k1), np.zeros((input_dim - k, 1))], axis=0)).float())
        
    def forward(self, x):
        x = torch.sqrt(self._k1)*F.relu(x @ self._mu1)-torch.sqrt(self._k2)*F.relu(x @ self._mu2)

        return x

class f2(nn.Module):
    def __init__(self, p=1.0, k=None, k1=None, input_dim=None):
        super().__init__()
        self._p = p
        self._k1 = torch.tensor(k1)
        self._k2 = torch.tensor(k - k1)

        self._neurons = nn.Parameter(torch.tensor(np.concatenate([np.eye(k),np.zeros((input_dim - k,k))],axis=0)).float(),requires_grad=False)
        self._head = nn.Parameter(torch.tensor(np.reshape((1-2*(np.arange(k)>=k1)),(k,1))).float(),requires_grad=False)
        
    def forward(self, x):
        x = torch.pow(F.relu(x @ self._neurons), self._p) @ self._head

        return x

def run_experiment(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    D = config['train']['D']
    k = config['train']['k']
    k1 = config['train']['k1']
    save_model = config['train']['save_model']
    n_train = config['train']['n_train']
    n_test = config['train']['n_test']
    rep = config['train']['repeat']
    sigma_list = np.linspace(*config['train']['sigma_list'])
    epsilon_list = np.linspace(*config['train']['epsilon_list'])

    batch_size = config['train_opt']['batch_size']
    max_epoch = config['train_opt']['max_epoch']
    lr = config['train_opt']['lr']

    h = config['train_net']['hidden_width']
    init_scale = config['train_net']['init_scale']

    
    
    # sigma_list = np.linspace(0.01, 0.3, 15)
    # epsilon_list = np.linspace(0.0, 1.5, 15)

    batch_size_dist = config['train_dist']['batch_size_dist']
    batch_num_dist = config['train_dist']['batch_num_dist']
        
    l_inf_diff = torch.zeros(rep, 2, len(sigma_list))
    robust_acc = torch.zeros(rep, 2, len(epsilon_list))

    for i, p in enumerate([1, 3]):
        
        for r in range(rep):

            logging.info(f"Training pReLU network with p={p}. Repeat: {r+1}")
            
            for j, sigma in enumerate(sigma_list):

                logging.info(f"Training with intraclass variance sigma = {sigma}")
            
                X_train, Y_train = generate_data(k=k, k1=k1, D=D, n=n_train, sigma=sigma / np.sqrt(D))
                train_data = TensorDataset(X_train, Y_train)
                train_data_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)

                X_test, Y_test = generate_data(k=k, k1=k1, D=D, n=n_test, sigma=sigma / np.sqrt(D))
                test_data = TensorDataset(X_test, Y_test/2.0+0.5)
                test_data_loader = DataLoader(test_data, batch_size=n_test,shuffle=False)

                model_kwargs = dict(center=None, input_dim=D, hidden_dim=h, output_dim=1, bias=False, init_std=init_scale)
                
                model = ShallowPreluNet(**model_kwargs).to(device)
                opt = optim.SGD(model.parameters(), lr=lr)
                
                for it in range(max_epoch):
                
                    model.eval()
                    
                    if it % 20 == 0:
                        with torch.no_grad():
                            xb, yb = next(iter(train_data_loader))
                            pred = model(xb.to(device), p, normalize_input=False)
                            loss = ((-torch.reshape(pred, (-1,)) * yb.to(device)).exp()).mean()
                            
                            logging.info(f"Epoch {it}: loss of a random single batch {loss.detach().cpu().numpy()}")
                
                
                    for xb, yb in train_data_loader:
                        pred = model(xb.to(device), p, normalize_input=False)
                        loss = ((-torch.reshape(pred, (-1,)) * yb.to(device)).exp()).mean()
                
                        loss.backward()
                        opt.step()
                        opt.zero_grad()

                if j==save_model:
                    torch.save({'state_dict': model.state_dict(),
                               'params': model_kwargs}, f'{config["save_dir"]}/saved_model_{i}.th')
                    logging.info(f"Model saved for p={p}, sigma={sigma}")
 
                    for je, eps in enumerate(epsilon_list):

                        robust_acc[r, i, je] = adv_atk_batch_bin_class(model, device, test_data_loader, eps, 2000, 0.1, 'l2', forward_kwargs=dict(p=p, normalize_input=False))
                        
                    
                logging.info("Estimating linf distance between trained network and corresponding ideal classifier")
                logging.info(f"Running NPGA for {batch_size_dist} samples and repeated for {batch_num_dist} times")
                
                    
                model_f = f1(k=k, k1=k1, input_dim=D).to(device) if p==1 else f2(p=p, k=k, k1=k1, input_dim=D).to(device)

                # esitimating c
                with torch.no_grad():
                    m1 = 0
                    m2 = 0
                    for t in range(batch_num_dist):
                        x, _ = generate_data(k=k, k1=k1, D=D, n=batch_size_dist, sigma=sigma / np.sqrt(D))
                        # x = torch.normal(0, 1/np.sqrt(D), size=(batch_size, D))
                        x = x.to(device)
                        m1 = m1 - (m1-(model(x,p=p, normalize_input=False)*model_f(x)).mean())/(t+1)
                        m2 = m2 - (m2-torch.square(model(x,p=p, normalize_input=False)).mean())/(t+1)
                c=m1/m2

                # NPGA
                max = torch.tensor(0)
                for t in range(batch_num_dist):
                    # x, _ = generate_data(k=k, k1=k1, D=D, n=batch_size, sigma=sigma / np.sqrt(D))
                    x = torch.normal(0, 1/np.sqrt(D), size=(batch_size_dist, D))
                    x = x/torch.linalg.norm(x, dim=1,keepdim=True)
                    m1 = estimate_l_inf_dist(model, model_f, device, c, x, 2000, 0.01, forward_kwargs=dict(p=p, normalize_input=False))
                    max = torch.maximum(max, torch.tensor([m1]))
    
            
                l_inf_diff[r, i, j] = max.detach().cpu()
                
                logging.info(f"Maximum l inf distance is {max}")

            
    torch.save(l_inf_diff, f'{config["save_dir"]}/linf_dist.pt')
    torch.save(robust_acc, f'{config["save_dir"]}/robust_acc.pt')
    torch.save({'sigma_list': sigma_list,
               'epsilon_list': epsilon_list,
               'k': k, 'k1':k1, 'D':D}, f'{config["save_dir"]}/experiment_info.th')
        

def generate_plot(config):
    
    experiment_info = torch.load(f'{config["save_dir"]}/experiment_info.th')
    sigma_list = experiment_info['sigma_list']
    epsilon_list = experiment_info['epsilon_list']
    k = experiment_info['k']
    k1 = experiment_info['k1']
    D = experiment_info['D']
    
    l_inf_diff = torch.load(f'{config["save_dir"]}/linf_dist.pt').numpy()
    robust_acc = torch.load(f'{config["save_dir"]}/robust_acc.pt').numpy()

    print(robust_acc.shape)
    model_infos_p_1 = torch.load(f'{config["save_dir"]}/saved_model_0.th')
    model_infos_p_3 = torch.load(f'{config["save_dir"]}/saved_model_1.th')

    model_kwargs = model_infos_p_1['params']
    model = ShallowPreluNet(**model_kwargs)
    model.load_state_dict(model_infos_p_1['state_dict'])
    W1 = model.prelulayer.W.detach().numpy()
    v1 = model.output_net.weight.detach().numpy()

    model_kwargs = model_infos_p_3['params']
    model = ShallowPreluNet(**model_kwargs)
    model.load_state_dict(model_infos_p_3['state_dict'])
    W3 = model.prelulayer.W.detach().numpy()
    v3 = model.output_net.weight.detach().numpy()


    C1 = W1 * np.abs(v1)
    C3 = W3 * np.abs(v3)
    N1 = np.linalg.norm(C1, axis=0)
    N3 = np.linalg.norm(C3, axis=0)

    idx1 = np.argsort(N1)
    idx3 = np.argsort(N3)

    mu_plus = np.concatenate([np.ones((k1, 1)) / np.sqrt(k1), np.zeros((D - k1, 1))], axis=0)
    mu_minus = np.concatenate([np.zeros((k1, 1)), np.ones((k - k1, 1)) / np.sqrt(k - k1), np.zeros((D - k, 1))], axis=0)
    Mus = np.concatenate([np.eye(k), np.zeros((D - k, k))], axis=0)
    M = np.concatenate([mu_plus, mu_minus, Mus], axis=1)

    label_font_size = 24
    title_font_size = 22
    ax_tick_font_size = 24
    ax_label_font_size = 24

    # last layer feature
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan']

    fig = plt.figure()
    fig.set_size_inches(35, 8, forward=True)

    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.835, wspace=0.25, hspace=0.4)
    ax1 = plt.subplot(2, 5, 1)  # linf_dist p=1
    ax2 = plt.subplot(2, 5, 6)  # linf_dist p=3

    ax3 = plt.subplot(2, 5, 2)  # dom neuron p = 1
    ax4 = plt.subplot(2, 5, 7)  # dom neuron p = 3

    ax5 = plt.subplot(1, 5, 3)  # neuron v.s. centers p = 1
    ax6 = plt.subplot(1, 5, 4)  # neuron v.s. centers p = 3

    ax7 = plt.subplot(1, 5, 5)  # adv

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    for i, ax in enumerate(axes):
        plt.sca(ax)
        plt.rcParams.update({
            "font.size": label_font_size})
        plt.tick_params(labelsize=ax_tick_font_size)
        # plt.title(plot_title[i])
        # ax.set_xlim(left=0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, 4))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.5)

    # ploting axis 1 and 2
    plt.sca(ax1)
    plt.plot(sigma_list, np.mean(l_inf_diff[:,0,:], axis=0), color=color[0], label='dist($f_p$, $F$)', linewidth=2.5)
    plt.fill_between(sigma_list, np.min(l_inf_diff[:,0,:], axis=0), np.max(l_inf_diff[:,0,:], axis=0), color=color[0], alpha=0.2)
    plt.title('(a)\n Estimated $l_\infty$ distance between \n trained network and classifiers $F(x), F^{(p)}(x)$ \n', fontsize=title_font_size)
    plt.ylabel('p=1', fontsize=ax_label_font_size)
    plt.xlabel(r'$\alpha$', fontsize=ax_label_font_size)
    plt.legend()

    plt.sca(ax2)
    plt.plot(sigma_list, np.mean(l_inf_diff[:,1,:], axis=0), color=color[2], label='dist($f_p$, $F^{(p)}$)', linewidth=2.5)
    plt.fill_between(sigma_list, np.min(l_inf_diff[:,1,:], axis=0), np.max(l_inf_diff[:,1,:], axis=0), color=color[2], alpha=0.2)
    plt.ylabel('p=3', fontsize=ax_label_font_size)
    ax2.yaxis.set_label_coords(-0.16, 0.4)
    plt.xlabel(r'$\alpha$', fontsize=ax_label_font_size)
    plt.legend()
    ax2.yaxis.get_offset_text().set_fontsize(24)
    # end of ploting axis 1 and 2

    
    # ploting axis 3 and 4
    plt.sca(ax3)
    idx_neuron = 300
    plt.bar(np.arange(idx_neuron) + 1, np.flip(N1[idx1[-idx_neuron:]]), color=color[0])
    plt.title('(b) \n Neuron contributions \n in trained pReLU networks \n'.format(k + 5), fontsize=title_font_size)
    plt.xlabel('index $j$', fontsize=ax_label_font_size)
    plt.ylabel(r'$|v_j|\Vert w_j\Vert $', fontsize=ax_label_font_size)

    plt.sca(ax4)
    idx_neuron = 50
    plt.bar(np.arange(idx_neuron) + 1, np.flip(N3[idx3[-idx_neuron:]]), color=color[2])
    plt.xlabel('index $j$', fontsize=ax_label_font_size)
    plt.ylabel(r'$|v_j|\Vert w_j\Vert $', fontsize=ax_label_font_size)
    ax4.yaxis.set_label_coords(-0.08, 0.4)
    # end of ploting axis 3 and 4
    

    C1 = C1[:, np.flip(idx1[-300:])]
    C3 = C3[:, np.flip(idx3[-20:])]
    temp1 = np.linalg.norm(C1, axis=0, keepdims=True)
    temp3 = np.linalg.norm(C3, axis=0, keepdims=True)
    C1 = C1 / np.linalg.norm(C1, axis=0, keepdims=True)
    C3 = C3 / np.linalg.norm(C3, axis=0, keepdims=True)
    corr1 = np.matmul(np.transpose(C1), M)

    corr_idx1 = np.argsort(corr1[:, 0])
    corr3 = np.matmul(np.transpose(C3), M)
    corr_idx3 = np.argsort(np.matmul(corr3, np.linspace(1,10000,12)))

    plt.sca(ax5)
    plt.pcolormesh(corr1[np.flip(corr_idx1),:], cmap='binary', vmin=0, vmax=1)
    plt.title('(c) \n Cosine betw. dominant neurons (row) \n and class/subclass centers (column) \n p=1',
              fontsize=title_font_size)
    ax5.xaxis.set_ticklabels([])
    # ax5.yaxis.set_ticklabels([])
    ax5.xaxis.set_ticks([])
    # ax5.yaxis.set_ticks([])

    plt.sca(ax6)
    plt.pcolormesh(corr3[corr_idx3], cmap='binary', vmin=0, vmax=1)
    plt.title('(d) \n Cosine betw. dominant neurons (row) \n and class/subclass centers (column) \n p=3',
              fontsize=title_font_size)
    plt.colorbar()
    ax6.xaxis.set_ticklabels([])
    # ax6.yaxis.set_ticklabels([])
    ax6.xaxis.set_ticks([])
    # ax6.yaxis.set_ticks([])

    plt.sca(ax7)
    plt.title('(e) \n Robust Accuracy of trained networks \n under $l_2$ PGD attack \n', fontsize=title_font_size)
    plt.plot(epsilon_list, np.mean(robust_acc[:,0,:], axis=0), color=color[0], label='p=1', linewidth=2.5)
    plt.fill_between(epsilon_list, np.min(robust_acc[:,0,:], axis=0), np.max(robust_acc[:,0,:], axis=0), color=color[0],
                     alpha=0.2)
    plt.plot(epsilon_list, np.mean(robust_acc[:,1,:], axis=0), color=color[2], label='p=3', linewidth=2.5)
    plt.fill_between(epsilon_list, np.min(robust_acc[:,1,:], axis=0), np.max(robust_acc[:,1,:], axis=0), color=color[0],
                     alpha=0.2)
    plt.plot([1 / np.sqrt(k), 1 / np.sqrt(k)], [0, 1.1], '--', color='gray', linewidth=2.5)
    plt.text(1 / np.sqrt(k) + 0.05, 1 / 2, r'$\frac{1}{\sqrt{k}}$')
    plt.plot([1 / np.sqrt(2), 1 / np.sqrt(2)], [0, 1.1], '--', color='gray', linewidth=2.5)
    plt.text(np.sqrt(2) / 2 + 0.02, 1 / 2 - 0.02, r'$\frac{\sqrt{2}}{2}$')
    plt.ylim(bottom=0, top=1.1)
    plt.xlim(left=0, right=1.5)
    plt.ylabel('Accuracy', fontsize=ax_label_font_size)
    plt.xlabel('$l_2$ Attack Radius', fontsize=ax_label_font_size)
    plt.legend(loc='upper right')

    # plt.show()
    plt.savefig(f'{config["save_dir"]}/plot.png', dpi=fig.dpi, bbox_inches='tight')
    # plt.close()
    

if __name__ == '__main__':

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
    
    
















