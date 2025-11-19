1. 데이터셋 사이즈 확인
2. 어떠한 모델을 쓸 것인지 확인
3. 그 모델에 데이터셋 사이즈 맞게 파라미터 변경
4. 적절한 optimizer, scheduler 사용
5. 하이퍼 파라마터 수정하며 train
6. preds 또는 logits 저장

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=200
)
batch_size = 256

transform_train = transforms.Compose([
    transforms.Resize((64,64)),   # 데이터 크기 모르면 64x64 대부분 OK
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

transform_test = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])
