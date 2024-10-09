import torch
from prelu_torch_modules import ShallowPreluNet
from load_mnist_datasets import load_mnist
from autoattack import AutoAttack


def load_pretrained_net(p=4):
    if p not in [1, 2, 3, 4]:
        raise ValueError('p must be 1, 2, 3 or 4')
    model_kwargs = dict(center=None, input_dim=784, hidden_dim=500, output_dim=10, bias=True)
    model = ShallowPreluNet(**model_kwargs).to(device)
    model.load_state_dict(torch.load(f'pReLU_pretrained_models/pReLU_{p}_pretrained.pt'))

    return model


def pReLU_forward(xb, p=4):
    if len(xb.shape) == 4 or len(xb.shape) == 3:
        xb = xb.view(-1, 784)

    model = load_pretrained_net(p=p)
    logits = model(xb.to(device), p=p)
    pred_label = torch.argmax(logits, 1)

    return pred_label, logits


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_loader_batch, train_data_loader_full, _, test_data_loader_full, centers, n_train, n_test, input_dim = load_mnist(
        batch_size=1000, n_class=10)

    p = 4  # p can be 1, 2, 3, 4; larger p = more robust

    # forward pass that takes MNIST images (n, 784) and output predicted labels (n, 1)
    # pReLU_forward(xb, p=4)

    # test
    xb, yb = next(iter(test_data_loader_full))
    pred_label, _ = pReLU_forward(xb, p=p)
    print(f'Clean accuracy: {(pred_label == yb).float().mean()}')

    # load a pReLU network that takes MNIST images (n, 784) and output logits (n, 10)
    # model = load_pretrained_net(p=p)

    # # test on AutoAttack

    # adversary = AutoAttack(lambda xb: model(xb.to(device), p=p), norm='Linf', eps=0.1, version='standard')
    # adversary.attacks_to_run = ['apgd-ce']
    # xb, yb = next(iter(test_data_loader_full))
    # x_adv = adversary.run_standard_evaluation(xb, yb, bs=10000)
