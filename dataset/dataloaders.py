dataloaders = {}
batch_size = {'train' : 512, 'test' : 256}
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                   drop_last=False, batch_size=batch_size['train'])
dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                   drop_last=False, batch_size=batch_size['test'])
dataset_size = {'train' : len(train_dataset), 'test' : len(test_dataset)}
