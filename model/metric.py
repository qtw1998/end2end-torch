import torch


def accuracy(output, target):
    with torch.no_grad():
        # pred = torch.argmax(output, dim=1)
        pred = output.squeeze()
        assert len(pred) == len(target)
        correct = 0
        correct += torch.sum(
            torch.Tensor(
                [pred[i] >= target[i] - 5.0 and pred[i] <= target[i] + 5.0 for i in range(len(target))])).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
