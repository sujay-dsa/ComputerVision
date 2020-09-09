from torchvision import transforms
import torch


class MnistTransforms:
    def __init__(self, strategy):
        if strategy == 1:
            self.dataset_mean = 0.1305
            self.dataset_std = 0.3081
            self.left_rotation = -5.0
            self.right_rotation = 5.0

    def mnist_train_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomRotation((self.left_rotation, self.right_rotation), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((self.dataset_mean,), (self.dataset_std,))
            # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
            # Note the difference between (0.1307) and (0.1307,)
        ])
        return train_transforms

    def mnist_test_transforms(self):
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.dataset_mean,), (self.dataset_std,))
        ])
        return test_transforms


class SDLoader:
    def __init__(self):
        self.ref = 1

    def mnist_data_loaders(self, train, test):
        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        # For reproducibility
        torch.manual_seed(SEED)

        if cuda:
            torch.cuda.manual_seed(SEED)

        # dataloader arguments - something you'll fetch these from cmdprmt
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(
            shuffle=True, batch_size=64)

        # train dataloader
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

        # test dataloader
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

        return train_loader, test_loader
