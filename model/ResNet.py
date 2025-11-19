class ResNet18_CIFAR(nn.Module):
    def __init__(self, class_num=100):
        super(ResNet18_CIFAR, self).__init__()
        self.class_num = class_num

        # 기본 pretrained resnet18 불러오기
        model = models.resnet18(pretrained=True)

        # CIFAR-100(32x32)에 맞게 첫 conv 수정
        # 원래: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # CIFAR에서는 maxpool 제거
        model.maxpool = nn.Identity()

        # 마지막 FC 레이어 수정 (512 → class_num)
        model.fc = nn.Linear(512, class_num)

        self.model = model

    def forward(self, x):
        return self.model(x)
