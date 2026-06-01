import torch

from transformers.models.mobilenet_v2 import MobileNetV2Model
from transformers.models.mobilenet_v2 import MobileNetV2PreTrainedModel


class MobileNetV2ForFacialExpressionRecognition(MobileNetV2PreTrainedModel):
    """
    from MobileNetV2 for image classification
    """
    def __init__(self, config, multimodal=False):
        super().__init__(config=config)

        self.num_labels = 2
        self.mobilenet_v2 = MobileNetV2Model(config).from_pretrained('google/mobilenet_v2_1.4_224')
        self.multimodal = multimodal

        last_hidden_size = self.mobilenet_v2.conv_1x1.convolution.out_channels

        self.dropout = torch.nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = torch.nn.Linear(last_hidden_size, self.num_labels)
        self.post_init()

    def forward(self, pixel_values=None, output_hidden_states=None, labels=None):
        outputs = self.mobilenet_v2(pixel_values, output_hidden_states=output_hidden_states)

        features = outputs.last_hidden_state
        pooled_output = features.mean(dim=[2,3])

        if self.multimodal:
            return pooled_output
        else:
            logits = self.classifier(self.dropout(pooled_output))

            return logits
