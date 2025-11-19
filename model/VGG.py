class VGG16(nn.Module):
    def __init__(self, class_num = 100):
        super(VGG16, self).__init__()
        self.class_num = class_num
        model = models.vgg16(pretrained = True)
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        model.classifier = nn.Sequential(
            nn.Linear(512,256,bias = True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,256,bias = True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,class_num,bias = True)
        )
        self.model = model

    def forward(self, x):
        x = self.model(x)

        return x
