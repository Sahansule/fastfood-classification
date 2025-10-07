import torch.nn as nn
import torchvision.models as models


class DenseNetCustom(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetCustom, self).__init__()

        # 🔸 Pretrained DenseNet121
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # 🔓 Son 45 katman
        for child in list(self.base_model.features.children())[:-45]:
            for param in child.parameters():
                param.requires_grad = False

        # 🔄 Classifier kısmını daha güçlü hale getir
        self.base_model.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
