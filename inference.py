model = ResNet18_CIFAR().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=0.005, nesterov=True)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

train_epoch = 100
loss_fn = nn.CrossEntropyLoss()

model = train(model, dataloaders, loss_fn, train_epoch, optimizer, scheduler, dataset_size)
