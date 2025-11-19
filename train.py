def train(model, dataloaders, loss_fn, train_epoch, optimizer, scheduler, dataset_size):
    program_start = time.time()

    loss_list = {'train' : [], 'test' : []}
    acc_list = {'train' : [], 'test' : []}

    for epoch in range(train_epoch):
        print(f'\nepoch = {epoch}')
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                model_output = model(inputs)
                _, preds = torch.max(model_output.data, 1)
                loss = loss_fn(model_output, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            running_corrects = running_corrects.float()
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_size[phase]

            print(f'{phase} Loss : {epoch_loss} Acc : {epoch_acc}')
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(100*epoch_acc)

        scheduler.step()

    time_elapsed = time.time() - program_start
    print(f'{(time_elapsed) / 60:.5f}mins')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))

    return loss_list, acc_list


