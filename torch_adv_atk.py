import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

def adv_atk_step(delta, epsilon, delta_grad, alpha, type):
    if type == 'I-FGSM':
        sign_delta_grad = delta_grad.sign()
        delta.data = (delta + alpha * sign_delta_grad).clamp(-epsilon, epsilon)
    elif type == 'linf':
        delta.data = (delta + alpha * delta_grad).clamp(-epsilon, epsilon)
    elif type == 'l2':
        dir = delta_grad/(torch.linalg.norm(delta_grad, dim=1, keepdim=True)+1e-7)
        # dir = delta_grad.sign()
        l2_norm = torch.norm(delta + alpha * dir, p=2, dim=1, keepdim=True)
        delta.data = (delta + alpha * dir) * torch.minimum(epsilon / l2_norm, torch.ones_like(l2_norm))

    return delta


def adv_atk_batch_bin_class(model, device, test_loader_batch, epsilon, iter, alpha, type, forward_kwargs=dict(), clamp=None):
    # Loop over all examples in test set
    for data, target in test_loader_batch:
        data, target = data.to(device), target.to(device)
        delta = torch.zeros_like(data).to(device)
        delta.requires_grad = True

        for it in range(iter):
            output = model(data + delta, **forward_kwargs)
            loss = ((-torch.reshape(output, (-1,)) * (target - 0.5) * 2).exp()).mean()
            model.zero_grad()

            loss.backward()

            # Call FGSM Attack
            delta = adv_atk_step(delta, epsilon, delta.grad, alpha, type)
            if clamp is None:
                pass
            else:
                delta.data = (data + delta).clamp(clamp[0], clamp[1]) - data
            delta.grad.zero_()

        output = model(data + delta, **forward_kwargs)
        correct = sum(
            (torch.reshape(output, (-1,)) * (target - 0.5) * 2 > 0).float())
        final_acc = correct / float(len(target))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(target)} = {final_acc}")

    return final_acc


def adv_atk_batch_mul_class(model, device, test_loader_full, epsilon, iter, alpha, type, forward_kwargs=dict(), clamp=None):
    for data, target in test_loader_full:
        data, target = data.to(device), target.to(device)
        delta = torch.zeros_like(data).to(device)
        delta.requires_grad = True

        for it in range(iter):
            output = model(data + delta, **forward_kwargs)

            loss = F.cross_entropy(output, target)
            model.zero_grad()

            loss.backward()

            # Call FGSM Attack
            delta = adv_atk_step(delta, epsilon, delta.grad, alpha, type)
            if clamp is None:
                pass
            else:
                delta.data = (data + delta).clamp(clamp[0], clamp[1]) - data
            delta.grad.zero_()

        output = model(data + delta, **forward_kwargs)
        correct = sum(
            (torch.argmax(output, 1) == target).float())
        final_acc = correct / float(len(target))
        print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(target)} = {final_acc}")

    return final_acc


def adv_atk_batch_img(pretrained_model, head, device, test_loader_batch, epsilon, iter, alpha, type, n_class, x_transform,
                      x_inv_transform, y_transform, clamp=None):
    correct = 0
    n_test = 0
    for data, target in test_loader_batch:
        data, target = data.to(device), y_transform(target, n_class).to(device)
        data = x_inv_transform(data, device)
        delta = torch.zeros_like(data).to(device)
        delta.requires_grad = True

        for it in range(iter):
            output = head(pretrained_model(x_transform(data + delta, device)))

            loss = F.cross_entropy(output, target)
            pretrained_model.zero_grad()
            head.zero_grad()

            loss.backward()

            # Call FGSM Attack
            delta = adv_atk_step(delta, epsilon, delta.grad, alpha, type)
            if clamp is None:
                pass
            else:
                delta.data = (data + delta).clamp(clamp[0], clamp[1]) - data
            delta.grad.zero_()

        output = head(pretrained_model(x_transform(data + delta, device)))
        correct += sum(
            (torch.argmax(output, 1) == target).float())
        n_test += len(target)
    final_acc = correct / n_test
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {n_test} = {final_acc}")

    return final_acc

def estimate_l_inf_dist(model1, model2, device, c, x, max_iter, alpha, forward_kwargs=dict()):

    data_batch = x.clone().detach().requires_grad_(True)
    data_batch = data_batch.to(device)
    scale = torch.tensor([c],requires_grad=False).to(device)
    loss = nn.MSELoss()

    for it in range(max_iter):
        output = loss(scale* model1(data_batch, **forward_kwargs), model2(data_batch))

        model1.zero_grad()
        model2.zero_grad()

        data_batch.retain_grad()
        output.backward()

        data_grad = data_batch.grad
        data_batch = data_batch + alpha * data_grad/(torch.linalg.norm(data_grad, dim=1, keepdim=True)+1e-7)
        data_batch = data_batch / torch.linalg.norm(data_batch, dim=1, keepdim=True)

        data_batch = data_batch.clone().detach().requires_grad_(True)


    max_dist = torch.abs(scale*model1(data_batch, **forward_kwargs) - model2(data_batch)).max().detach().item()

    return max_dist

if __name__ == '__main__':
    pass
