cifar_resize = (32, 32)
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform_train = transforms.Compose([
    transforms.Resize(cifar_resize, interpolation=PIL.Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.Resize(cifar_resize, interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
