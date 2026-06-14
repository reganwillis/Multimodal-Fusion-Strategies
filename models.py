import torch

from transformers.models.mobilenet_v2 import MobileNetV2Model
from transformers.models.mobilenet_v2 import MobileNetV2PreTrainedModel

from transformers.models.bert import BertModel

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config

from transformers.models.vit import ViTModel
from transformers.models.vit import ViTPreTrainedModel
from transformers.models.vit.configuration_vit import ViTConfig

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class MobileNetV2ForFacialExpressionRecognition(MobileNetV2PreTrainedModel):
    """
    from MobileNetV2 for image classification
    """
    def __init__(self, config, multimodal=False):
        super().__init__(config=config)
        self.multimodal = multimodal

        self.N_CLASSES = 2
        self.mobilenet_v2 = MobileNetV2Model(config).from_pretrained('google/mobilenet_v2_1.4_224')

        last_hidden_size = self.mobilenet_v2.conv_1x1.convolution.out_channels

        self.dropout = torch.nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = torch.nn.Linear(last_hidden_size, self.N_CLASSES)
        self.post_init()

    def forward(self, pixel_values=None, output_hidden_states=None, labels=None):
        outputs = self.mobilenet_v2(pixel_values, output_hidden_states=output_hidden_states)

        features = outputs.last_hidden_state
        pooled_output = features.mean(dim=[2,3])
        x = self.dropout(pooled_output)

        if self.multimodal:
            return x
        else:
            logits = self.classifier(x)

            return logits


class ViTForFacialExpressionRecognition(ViTPreTrainedModel):
    """
    from ViT for image classification
    """
    def __init__(self, config, multimodal=False):
        super().__init__(config)
        self.multimodal = multimodal

        self.N_CLASSES = 2
        self.vit = ViTModel(config, add_pooling_layer=False).from_pretrained('google/vit-base-patch16-224')
        # Classifier head
        self.classifier = torch.nn.Linear(config.hidden_size, self.N_CLASSES)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values = None,
        labels = None,
        interpolate_pos_encoding = None,
        **kwargs,
        ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.vit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        x = sequence_output[:, 0, :]

        if self.multimodal:
            return x
        else:
            logits = self.classifier(x)

            return logits


class BERTForSentimentAnalysis(BertModel):
    """
    from Bert For Sequence Classification
    """
    def __init__(self, config, multimodal=False):
        super().__init__(config=config)
        self.multimodal = multimodal
        self.N_CLASSES = 2
        self.config = config

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(config.hidden_size, self.N_CLASSES)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        x = outputs[1]
        x = self.dropout(x)

        if self.multimodal:
            return x
        else:
            logits = self.classifier(x)

            return logits


class BiDirectionalCrossAttnBlock(torch.nn.Module):

    def __init__(self, hidden_dim, num_heads=12):
        super().__init__()
        self.text_to_vision = torch.nn.MultiheadAttention(hidden_dim, num_heads)

        self.vision_to_text = torch.nn.MultiheadAttention(hidden_dim, num_heads)

        self.text_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.text_norm2 = torch.nn.LayerNorm(hidden_dim)
        self.text_ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.vision_norm1 = torch.nn.LayerNorm(hidden_dim)
        self.vision_norm2 = torch.nn.LayerNorm(hidden_dim)
        self.vision_ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def _to_tokens(self, x):
        return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        elif x.dim() == 3:
            return x
        elif x.dim() == 4:
            return x.flatten(2).transpose(1, 2)

    def forward(self, text_tokens, vision_tokens, text_key_padding_mask=None):
        text_tokens = self._to_tokens(text_tokens)
        vision_tokens = self._to_tokens(vision_tokens)

        # text queries vision
        #print(text_tokens.shape, vision_tokens.shape)
        text_attn, _ = self.text_to_vision(
            query=text_tokens,
            key=vision_tokens,
            value=vision_tokens
        )
        text_tokens = self.text_norm1(text_tokens + text_attn)
        #text_tokens = self.text_norm2(text_tokens + self.text_ffn(text_tokens))

        # vision queries text
        vision_attn, _ = self.vision_to_text(
            query=vision_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding_mask
        )
        vision_tokens = self.vision_norm1(vision_tokens + vision_attn)
        #vision_tokens = self.vision_norm2(vision_tokens + self.vision_ffn(vision_tokens))

        return text_tokens, vision_tokens


class LateFusion(torch.nn.Module):

    def __init__(self, vision_model, cross_attn_fusion):
        super().__init__()
        self.N_CLASSES = 2
        self.vision_model = vision_model
        self.cross_attn_fusion = cross_attn_fusion

        if vision_model == 'mobilenet':
            cfg = MobileNetV2Config()
            self.fer = MobileNetV2ForFacialExpressionRecognition(cfg, True)
        elif vision_model == 'vit':
            cfg = ViTConfig()
            self.fer = ViTForFacialExpressionRecognition(cfg, True)
        cfg = BertConfig()
        self.bert = BERTForSentimentAnalysis(cfg, True)

        self.dim = 768
        self.text_proj = torch.nn.Linear(self.bert.config.hidden_size, self.dim)
        self.vision_proj = torch.nn.LazyLinear(self.dim)
        if self.cross_attn_fusion:
            print('Initializing Cross Attention Block..')
            self.cross_attn = BiDirectionalCrossAttnBlock(self.dim)

        # classification head
        self.linear = torch.nn.LazyLinear(512)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fusion_layer = torch.nn.Linear(256, self.N_CLASSES)

    def forward(self, images=None, input_ids=None, attention_mask=None, token_type_ids=None):
        # call forward pass of fer and bert
        if self.vision_model != 'None': fer_output = self.fer(images)
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        # project to same space
        if self.vision_model != 'None': fer_output = self.vision_proj(fer_output)
        bert_output = self.text_proj(bert_output)

        if self.cross_attn_fusion:
            bert_output, fer_output = self.cross_attn(bert_output, fer_output, None)

        # concat
        if self.vision_model != 'None':
            output = torch.cat((fer_output, bert_output), dim=1)
        else:
            output = bert_output
        x = self.linear(output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        result = self.fusion_layer(x)

        return result


class AttentionFusion(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.fc = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(self.fc(attn_out.squeeze(1)))


class LinearFusion(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = torch.nn.LazyLinear(dim)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.fc(x))


class MidFusion(torch.nn.Module):
    """
    Intermediate fusion of a vision model and BERT.
    """
    def __init__(self, vision_model, cross_attn_fusion, attn_fusion=False, fuse_place=[8,6,4,4]):
        super().__init__()
        self.N_CLASSES = 2
        self.vision_model = vision_model
        self.cross_attn_fusion = cross_attn_fusion
        self.attn_fusion = attn_fusion
        self.fuse_place = fuse_place
        self.TRUNCATE_IDX = self.fuse_place[3]
        self.dim = 768
        self.fusion_layers = torch.nn.ModuleList()

        cfg = BertConfig()
        self.bert = BertModel(cfg).from_pretrained('bert-base-uncased')

        if vision_model == 'mobilenet':
            cfg = MobileNetV2Config()
            self.mobilenet_v2 = MobileNetV2Model(cfg).from_pretrained('google/mobilenet_v2_1.4_224')

            self.pool_layers = torch.nn.ModuleList()
            self.conv_layers = torch.nn.ModuleList()

            self.conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(48, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(28, 28), stride=1)
            ))

            self.conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(88, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
            ))

            self.conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(88, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
            ))
            self.vision_proj = torch.nn.LazyLinear(self.dim)
        elif vision_model == 'vit':
            cfg = ViTConfig()
            self.vit = ViTModel(cfg).from_pretrained('google/vit-base-patch16-224')
        if self.cross_attn_fusion:
            print('Initializing Cross Attention Block..')
            self.cross_attn_layers = torch.nn.ModuleList()
            self.cross_attn_layers.append(BiDirectionalCrossAttnBlock(self.dim))
            self.cross_attn_layers.append(BiDirectionalCrossAttnBlock(self.dim))
            self.cross_attn_layers.append(BiDirectionalCrossAttnBlock(self.dim))
        if self.attn_fusion:
            self.fusion_layers.append(AttentionFusion(1536))
            self.fusion_layers.append(AttentionFusion(1536))
            self.fusion_layers.append(AttentionFusion(1536))
        else:
            # standard
            self.fusion_layers.append(LinearFusion(1536))  # 3
            self.fusion_layers.append(LinearFusion(1536))  # 6
            self.fusion_layers.append(LinearFusion(1536))  # 7

        # classification head
        self.linear1 = torch.nn.LazyLinear(512)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(256, self.N_CLASSES)

    def get_vision_model_hidden_states(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None):
        if self.vision_model == 'mobilenet':
            hidden_states_vision = self.mobilenet_v2.conv_stem(pixel_values)
        elif self.vision_model == 'vit':
            vit_embedding_output = self.vit.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
            hidden_states_vision = vit_embedding_output
        return hidden_states_vision

    def bert_embeddings_and_attention_mask(self, input_ids, position_ids, token_type_ids, inputs_embeds, past_key_values=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if self.bert.config.use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache, EncoderDecoderCache
            past_key_values = (
                    EncoderDecoderCache(DynamicCache(config=self.bert.config), DynamicCache(config=self.bert.config))
                    if encoder_hidden_states is not None or self.bert.config.is_encoder_decoder
                    else DynamicCache(config=self.bert.config)
            )

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask, encoder_attention_mask = self.bert._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )

        return past_key_values, past_key_values_length, embedding_output, attention_mask, encoder_attention_mask

    def forward(
        self,
        pixel_values=None,  # mobilenet val
        bool_masked_pos=None,
        interpolate_pos_encoding=None,
        labels=None,  # mobilenet val
        input_ids=None,  # bert val
        attention_mask=None,  # bert val
        token_type_ids=None,  # bert val
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        **kwargs
    ):
        # VISION MODEL - HIDDEN STATES
        if self.vision_model != 'None':
            hidden_states_vision = self.get_vision_model_hidden_states(pixel_values, bool_masked_pos, interpolate_pos_encoding)

        # BERT - HIDDEN STATES
        past_key_values, past_key_values_length, hidden_states_bert, attention_mask, encoder_attention_mask = self.bert_embeddings_and_attention_mask(input_ids, position_ids, token_type_ids, inputs_embeds, past_key_values, attention_mask, encoder_hidden_states, encoder_attention_mask)

        # JOINT ENCODER
        x = None
        fused_outputs = []
        j = 0
        for i in range(len(self.bert.encoder.layer)-self.TRUNCATE_IDX):
            #print('LAYER', i+1)
            if self.vision_model != 'None':
                if self.vision_model == 'mobilenet':
                    # mobilenet - layer modules (inverted residuals)
                    hidden_states_vision = self.mobilenet_v2.layer[i](hidden_states_vision)
                elif self.vision_model == 'vit':
                    hidden_states_vision = self.vit.layers[i](hidden_states_vision)
                    hidden_states_vision = self.vit.layernorm(hidden_states_vision)

            hidden_states_bert = self.bert.encoder.layer[i](
                hidden_states_bert,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

            # FUSE EMBEDDINGS
            if i == self.fuse_place[0]-1 or i == self.fuse_place[1]-1 or i == self.fuse_place[2]-1:
                if self.vision_model != 'None':
                    if self.vision_model == 'mobilenet':
                        hs_vision = self.conv_layers[j](hidden_states_vision).squeeze(-1).squeeze(-1)
                        hs_vision = self.vision_proj(hs_vision)
                    elif self.vision_model == 'vit':
                        hs_vision = hidden_states_vision[:, 0, :]
                hs_bert = hidden_states_bert[:, 0, :]
                if self.cross_attn_fusion:
                    hs_bert, hs_vision = self.cross_attn_layers[j](hs_bert, hs_vision, None)
                if self.vision_model != 'None':
                    x = torch.cat((hs_vision, hs_bert), dim=1)
                    x = self.fusion_layers[j](x)
                else:
                    x = hs_bert
                fused_outputs.append(x)
                j = j + 1
        x = torch.cat(fused_outputs, dim=1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.out(x)

        return x


class AttentionNet(torch.nn.Module):

    def __init__(self, dim, num_blocks=6):
        super().__init__()
        print('Adding attention net with', num_blocks, 'blocks...')
        self.attn_net = torch.nn.ModuleList()

        for i in range(num_blocks):
            attn_block = torch.nn.ModuleList()
            attn_block.append(torch.nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True))
            attn_block.append(torch.nn.Linear(dim, dim))
            attn_block.append(torch.nn.LayerNorm(dim))
            self.attn_net.append(attn_block)

    def forward(self, x):
        for block in self.attn_net:
            x = x.unsqueeze(1)
            attn_out, _ = block[0](x, x, x)
            x = attn_out.squeeze(1)
            x = block[1](x)
            x = block[2](x)
        return x


class EarlyFusion(torch.nn.Module):
    """
    Early fusion of vision model (MobileNetV2, ViT) and BERT.
    """
    def __init__(self, vision_model, cross_attn_fusion, num_attn_net_blocks=4):
        super().__init__()
        self.N_CLASSES = 2
        self.vision_model = vision_model
        self.cross_attn_fusion = cross_attn_fusion
        self.num_backbone_layers = 6
        self.dim = 768

        cfg = BertConfig()
        self.bert = BertModel(cfg).from_pretrained('bert-base-uncased')

        self.cross_attn = BiDirectionalCrossAttnBlock(self.dim)

        if vision_model == 'mobilenet':
            cfg = MobileNetV2Config()
            self.mobilenet_v2 = MobileNetV2Model(cfg).from_pretrained('google/mobilenet_v2_1.4_224')
            self.num_intermediate_layers = len(self.mobilenet_v2.layer)
            self.pool_layer = torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
            self.vision_proj = torch.nn.LazyLinear(self.dim)
        elif vision_model == 'vit':
            cfg = ViTConfig()
            self.vit = ViTModel(cfg).from_pretrained('google/vit-base-patch16-224')
        if vision_model != 'None':
            self.attn_net = AttentionNet(1536, num_blocks=num_attn_net_blocks)
        else:
            self.attn_net = AttentionNet(768, num_blocks=num_attn_net_blocks)

        # classification head
        self.linear1 = torch.nn.LazyLinear(512)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(256, self.N_CLASSES)

    def get_vision_model_hidden_states(self, pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None):
        if self.vision_model == 'mobilenet':
            hidden_states_vision = self.mobilenet_v2.conv_stem(pixel_values)
        elif self.vision_model == 'vit':
            vit_embedding_output = self.vit.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
            hidden_states_vision = vit_embedding_output
        return hidden_states_vision

    def bert_embeddings_and_attention_mask(self, input_ids, position_ids, token_type_ids, inputs_embeds, past_key_values=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if self.bert.config.use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache, EncoderDecoderCache
            past_key_values = (
                    EncoderDecoderCache(DynamicCache(config=self.bert.config), DynamicCache(config=self.bert.config))
                    if encoder_hidden_states is not None or self.bert.config.is_encoder_decoder
                    else DynamicCache(config=self.bert.config)
            )

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask, encoder_attention_mask = self.bert._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )

        return past_key_values, past_key_values_length, embedding_output, attention_mask, encoder_attention_mask

    def forward(
        self,
        pixel_values=None,  # mobilenet val
        bool_masked_pos=None,
        interpolate_pos_encoding=None,
        labels=None,  # mobilenet val
        input_ids=None,  # bert val
        attention_mask=None,  # bert val
        token_type_ids=None,  # bert val
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        **kwargs
    ):
        # VISION MODEL - HIDDEN STATES
        if self.vision_model != 'None':
            hidden_states_vision = self.get_vision_model_hidden_states(pixel_values, bool_masked_pos, interpolate_pos_encoding)

        # BERT - HIDDEN STATES
        past_key_values, past_key_values_length, hidden_states_bert, attention_mask, encoder_attention_mask = self.bert_embeddings_and_attention_mask(input_ids, position_ids, token_type_ids, inputs_embeds, past_key_values, attention_mask, encoder_hidden_states, encoder_attention_mask)

        # JOINT ENCODER - RUN THROUGH ALL STANDARD LAYERS
        for i in range(self.num_backbone_layers):
            #print('LAYER', i)

            # VISION ENCODER
            if self.vision_model != 'None':
                if self.vision_model == 'mobilenet':
                    # mobilenet - layer modules (inverted residuals)
                    hidden_states_vision = self.mobilenet_v2.layer[i](hidden_states_vision)
                elif self.vision_model == 'vit':
                    hidden_states_vision = self.vit.layers[i](hidden_states_vision)
                    hidden_states_vision = self.vit.layernorm(hidden_states_vision)

            # BERT ENCODER
            hidden_states_bert = self.bert.encoder.layer[i](
                hidden_states_bert,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # FUSE
        if self.vision_model != 'None':
            if self.vision_model == 'mobilenet':
                # early fusion - cnn with mobilenet output + bert output as input
                hidden_states_vision = self.pool_layer(hidden_states_vision).squeeze(-1).squeeze(-1)
                hidden_states_vision = self.vision_proj(hidden_states_vision)
            elif self.vision_model == 'vit':
                hidden_states_vision = hidden_states_vision[:, 0, :]
        hidden_states_bert = hidden_states_bert[:, 0, :]

        if self.cross_attn_fusion:
            hidden_states_bert, hidden_states_vision = self.cross_attn(hidden_states_bert, hidden_states_vision)

        if self.vision_model != 'None':
            concat = torch.cat((hidden_states_vision, hidden_states_bert), dim=1)
        else:
            concat = hidden_states_bert

        x = self.attn_net(concat)

        # classification head
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.out(x)

        return x
