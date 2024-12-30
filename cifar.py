from torchvision.datasets import CIFAR10, CIFAR100

def get_cifar10(train=True, transform=None, target_transform=None):
    datasets = CIFAR10(root='/home/akie/workplace/mylib/datas/DataSets/cifar10', train=train, transform=transform, target_transform=target_transform)
    return datasets

def get_cifar100(train=True, transform=None, target_transform=None):
    datasets = CIFAR100(root='/home/akie/workplace/mylib/datas/DataSets/cifar100', train=train, transform=transform, target_transform=target_transform)
    return datasets

if __name__ == '__main__':
    pass