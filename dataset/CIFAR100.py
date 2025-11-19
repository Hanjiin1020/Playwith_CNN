train_dataset = datasets.CIFAR100(root = './cifar100', train = True, download = True,
                                  transform = transform_train)
test_dataset = datasets.CIFAR100(root = './cifar100', train = False, download = True,
                                  transform = transform_test)
