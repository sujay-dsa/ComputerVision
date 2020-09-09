import torch
import torch.nn.functional as F
incorrect_examples = []
predicted_img = []
actual_targets = []


def test_mnist(model, device, test_loader, test_losses, test_accuracies, regularizer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if (regularizer == 'gbn'):
                _, imgpred = torch.max(output, 1)
                idxs_mask = ((imgpred == target) == False).nonzero()
                incorrect_examples.append(data[idxs_mask].cpu().numpy())
                actual_targets.append(target.cpu().numpy())
                predicted_img.append(imgpred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracies.append(100. * correct / len(test_loader.dataset))