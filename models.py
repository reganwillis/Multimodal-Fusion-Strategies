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

        self.N_CLASSES = 2
        self.mobilenet_v2 = MobileNetV2Model(config).from_pretrained('google/mobilenet_v2_1.4_224')
        self.multimodal = multimodal

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

        self.N_CLASSES = 2
        self.vit = ViTModel(config, add_pooling_layer=False).from_pretrained('google/vit-base-patch16-224')
        self.multimodal=multimodal

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
        self.N_CLASSES = 2
        self.config = config
        self.multimodal = multimodal

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(config.hidden_size, self.N_CLASSES)

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
    def forward(self, text_tokens, vision_tokens, text_key_padding_mask=None):
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
        fer_output = self.fer(images)
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        # project to same space
        fer_output = self.vision_proj(fer_output)
        bert_output = self.text_proj(bert_output)

        if self.cross_attn_fusion:
            bert_output, fer_output = self.cross_attn(bert_output, fer_output, None)
        # pool
        #fer_output = fer_output.pooler_output
        #bert_output = bert_output.pooler_output
        #print(fer_output.shape)
        #print(bert_output.shape)
        #fer_output = fer_output[:, 0, :]
        #bert_output = bert_output[:, 0, :]
        #print(fer_output.shape)
        #print(bert_output.shape)
        #print('exiting..')
        #exit()

        # concat
        output = torch.cat((fer_output, bert_output), dim=1)
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

        self.lin = torch.nn.LazyLinear(self.dim)  # TODO - clean

        cfg = BertConfig()
        self.bert = BertModel(cfg).from_pretrained('bert-base-uncased')
        self.text_proj = torch.nn.Linear(self.bert.config.hidden_size, self.dim)

        if vision_model == 'mobilenet':
            cfg = MobileNetV2Config()
            self.mobilenet_v2 = MobileNetV2Model(cfg).from_pretrained('google/mobilenet_v2_1.4_224')

            self.pool_layers = torch.nn.ModuleList()
            self.conv_layers = torch.nn.ModuleList()
            self.fusion_layers = torch.nn.ModuleList()

            self.vision_proj = torch.nn.LazyLinear(self.dim)

            #self.pool_layers.append(torch.nn.MaxPool2d(kernel_size=(28, 28), stride=1))
            self.pool_layers.append(torch.nn.AdaptiveAvgPool2d((1,1)))

            self.conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(48, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(28, 28), stride=1)
            ))

            #self.pool_layers.append(torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1))
            self.pool_layers.append(torch.nn.AdaptiveAvgPool2d((1,1)))

            self.conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(88, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
            ))

            #self.pool_layers.append(torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1))
            self.pool_layers.append(torch.nn.AdaptiveAvgPool2d((1,1)))

            self.conv_layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(88, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
            ))
        elif vision_model == 'vit':
            cfg = ViTConfig()
            self.vit = ViTModel(cfg).from_pretrained('google/vit-base-patch16-224')
            self.fusion_layers = torch.nn.ModuleList()

            self.vision_proj = torch.nn.Linear(self.vit.config.hidden_size, self.dim)
        if self.cross_attn_fusion:
            print('Initializing Cross Attention Block..')
            self.cross_attn_layers = torch.nn.ModuleList()
            self.cross_attn_layers.append(BiDirectionalCrossAttnBlock(self.dim))
            self.cross_attn_layers.append(BiDirectionalCrossAttnBlock(self.dim))
            self.cross_attn_layers.append(BiDirectionalCrossAttnBlock(self.dim))
        elif self.attn_fusion:
            self.fusion_layers.append(AttentionFusion(self.dim))
            self.fusion_layers.append(AttentionFusion(self.dim))
            self.fusion_layers.append(AttentionFusion(self.dim))
        else:
            # standard
            self.fusion_layers.append(LinearFusion(512))  # 3
            self.fusion_layers.append(LinearFusion(512))  # 6
            self.fusion_layers.append(LinearFusion(512))  # 7

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
        #self.bert.config.use_cache = False  # FIXME: disabling cache due to versioning differences
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
        hidden_states_vision = self.get_vision_model_hidden_states(pixel_values, bool_masked_pos, interpolate_pos_encoding)

        # BERT - HIDDEN STATES
        past_key_values, past_key_values_length, hidden_states_bert, attention_mask, encoder_attention_mask = self.bert_embeddings_and_attention_mask(input_ids, position_ids, token_type_ids, inputs_embeds, past_key_values, attention_mask, encoder_hidden_states, encoder_attention_mask)

        # JOINT ENCODER
        x = None
        fused_outputs = []
        j = 0
        for i in range(len(self.bert.encoder.layer)-self.TRUNCATE_IDX):
            #print('LAYER', i+1)
            if self.vision_model == 'mobilenet':
                # mobilenet - layer modules (inverted residuals)
                hidden_states_vision = self.mobilenet_v2.layer[i](hidden_states_vision)
            elif self.vision_model == 'vit':
                #hidden_states_vision = self.vit.encoder.layer[i](hidden_states_vision)[0]
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
                # project to same space
                #hidden_states_vision = self.vision_proj(hidden_states_vision)
                #hidden_states_bert = self.text_proj(hidden_states_bert)
                #print(hidden_states_vision.shape, hidden_states_bert.shape)

                if self.cross_attn_fusion:
                    #print(hidden_states_vision.shape, hidden_states_bert.shape)
                    hidden_states_bert, hidden_states_vision = self.cross_attn_layers[j](hidden_states_bert, hidden_states_vision, None)
                    #print(hidden_states_vision.shape, hidden_states_bert.shape)
                #else:
                # TODO: may need to flatten somehow
                # project to same space replaces this
                if self.vision_model == 'mobilenet':
                    flat1 = self.conv_layers[j](hidden_states_vision).squeeze(-1).squeeze(-1)
                    #flat1 = self.pool_layers[j](flat1).flatten(1)
                    flat1 = self.lin(flat1)  # project to same space
                elif self.vision_model == 'vit':
                    flat1 = hidden_states_vision[:, 0, :]
                    #flat1 = hidden_states_vision.pooler_output
                flat2 = hidden_states_bert[:, 0, :]
                #flat2 = hidden_states_bert.pooler_output#[:, 0, :]
                #print(flat1.shape, flat2.shape)
                x = torch.cat((flat1, flat2), dim=1)
                #x = torch.cat((hidden_states_vision, hidden_states_bert), dim=1)
                x = self.fusion_layers[j](x)
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

    def forward_old_version(
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
        past_key_values=None
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if self.vision_model == 'mobilenet':
            # mobilenet - conv stem
            hidden_states_vision = self.mobilenet_v2.conv_stem(pixel_values)
        elif self.vision_model == 'vit':
            vit_embedding_output = self.vit.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
            hidden_states_vision = vit_embedding_output

        # bert - init
        output_attentions = self.bert.config.output_attentions

        if self.bert.config.is_decoder:
            self.bert.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.bert.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        # FIXME: with transformers version no attributes attn_implementation and position_embeddinging type
        #use_sdpa_attention_masks = (
        #    self.bert.attn_implementation == "sdpa"
        #    and self.bert.position_embedding_type == "absolute"
        #    and head_mask is None
        #    and not output_attentions
        #)
        use_sdpa_attention_masks = False

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.bert.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # bert - encoding
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        # from foward pass of bert encoder
        # accounting for all vars that change in encoder forward pass
        hidden_states_bert=embedding_output
        attention_mask=extended_attention_mask
        encoder_attention_mask=encoder_extended_attention_mask

        if self.bert.encoder.gradient_checkpointing and self.bert.encoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        truncate_layers = self.fuse_place[3]
        x = None
        fused_outputs = []
        j = 0
        for i in range(len(self.bert.encoder.layer)-truncate_layers):
            #print('LAYER', i+1)
            if self.vision_model == 'mobilenet':
                # mobilenet - layer modules (inverted residuals)
                hidden_states_vision = self.mobilenet_v2.layer[i](hidden_states_vision)
            elif self.vision_model == 'vit':
                hidden_states_vision = self.vit.encoder.layer[i](hidden_states_vision)[0]

            # bert layer
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = self.bert.encoder.layer[i](
                hidden_states_bert,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value
            )
            hidden_states_bert = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # fusion - mobilenet output + bert output
            if i == self.fuse_place[0]-1 or i == self.fuse_place[1]-1 or i == self.fuse_place[2]-1:
                hidden_states_vision = self.vision_proj(hidden_states_vision)
                hidden_states_bert = self.text_proj(hidden_states_bert)
                #if self.cross_attn_fusion:
                #    hidden_states_bert, hidden_states_vision = self.cross_attn(hidden_states_bert, hidden_states_vision, None)
                
                if self.vision_model == 'mobilenet':
                    #flat1 = self.conv_layers[j](hidden_states_vision).squeeze(-1).squeeze(-1)
                    flat1 = self.hidden_states_vision[:, 0, :]  # WARN: may not work
                    #flat1 = hidden_states_vision.pooler_output
                elif self.vision_model == 'vit':
                    flat1 = hidden_states_vision[:, 0, :]
                    #flat1 = hidden_states_vision.pooler_output
                flat2 = hidden_states_bert[:, 0, :]
                #flat2 = hidden_states_bert.pooler_output#[:, 0, :]

                concat = torch.cat((flat1, flat2), dim=1)

                x = self.fusion_layers[j](concat)
                x = torch.nn.ReLU()(x)
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

        self.lin = torch.nn.LazyLinear(self.dim)  # TODO - clean

        cfg = BertConfig()
        self.bert = BertModel(cfg).from_pretrained('bert-base-uncased')
        self.text_proj = torch.nn.Linear(self.bert.config.hidden_size, self.dim)

        self.cross_attn = BiDirectionalCrossAttnBlock(self.dim)

        if vision_model == 'mobilenet':
            cfg = MobileNetV2Config()
            self.mobilenet_v2 = MobileNetV2Model(cfg).from_pretrained('google/mobilenet_v2_1.4_224')
            self.num_intermediate_layers = len(self.mobilenet_v2.layer)
            self.pool_layer = torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
            self.vision_proj = torch.nn.LazyLinear(self.dim)
            #self.attn_net = AttentionNet(856, num_blocks=num_attn_net_blocks)
        elif vision_model == 'vit':
            cfg = ViTConfig()
            self.vit = ViTModel(cfg).from_pretrained('google/vit-base-patch16-224')
            self.vision_proj = torch.nn.Linear(self.vit.config.hidden_size, self.dim)
        self.attn_net = AttentionNet(1536, num_blocks=num_attn_net_blocks)

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
        #self.bert.config.use_cache = False  # FIXME: disabling cache due to versioning differences
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
        hidden_states_vision = self.get_vision_model_hidden_states(pixel_values, bool_masked_pos, interpolate_pos_encoding)

        # BERT - HIDDEN STATES
        past_key_values, past_key_values_length, hidden_states_bert, attention_mask, encoder_attention_mask = self.bert_embeddings_and_attention_mask(input_ids, position_ids, token_type_ids, inputs_embeds, past_key_values, attention_mask, encoder_hidden_states, encoder_attention_mask)

        # JOINT ENCODER - RUN THROUGH ALL STANDARD LAYERS
        for i in range(self.num_backbone_layers):
            #print('LAYER', i)

            # VISION ENCODER
            if self.vision_model == 'mobilenet':
                # mobilenet - layer modules (inverted residuals)
                hidden_states_vision = self.mobilenet_v2.layer[i](hidden_states_vision)
            elif self.vision_model == 'vit':
                #hidden_states_vision = self.vit.encoder.layer[i](hidden_states_vision)[0]
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

        # project to same space
        #hidden_states_vision = self.vision_proj(hidden_states_vision)
        #hidden_states_bert = self.text_proj(hidden_states_bert)

        if self.cross_attn_fusion:
            hidden_states_bert, hidden_states_vision = self.cross_attn(hidden_states_bert, hidden_states_vision)

        if self.vision_model == 'mobilenet':
            # early fusion - cnn with mobilenet output + bert output as input
            flat1 = self.pool_layer(hidden_states_vision).squeeze(-1).squeeze(-1)
            flat1 = self.lin(flat1)
            #hidden_states_vision = self.pool_layer(hidden_states_vision).pooler_output
        elif self.vision_model == 'vit':
            flat1 = hidden_states_vision[:, 0, :]
            #hidden_states_vision = hidden_states_vision.pooler_output
        flat2 = hidden_states_bert[:, 0, :]
        #hidden_states_bert = hidden_states_bert.pooler_output

        concat = torch.cat((flat1, flat2), dim=1)

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

    def forward_old_version(self, pixel_values=None, bool_masked_pos=None, interpolate_pos_encoding=None, labels=None, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if self.vision_model == 'mobilenet':
            # mobilenet - conv stem
            hidden_states_vision = self.mobilenet_v2.conv_stem(pixel_values)
        elif self.vision_model == 'vit':
            # vit - init
            vit_embedding_output = self.vit.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding)
            hidden_states_vision = vit_embedding_output
        # bert - init
        output_attentions = self.bert.config.output_attentions

        if self.bert.config.is_decoder:
            self.bert.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.bert.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        #use_sdpa_attention_masks = (
        #    self.bert.attn_implementation == "sdpa"
        #    and self.bert.position_embedding_type == "absolute"
        #    and head_mask is None
        #    and not output_attentions
        #)
        use_sdpa_attention_masks = False

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.bert.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # bert - encoding
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        # from foward pass of bert encoder
        # accounting for all vars that change in encoder forward pass
        hidden_states_bert=embedding_output
        attention_mask=extended_attention_mask
        encoder_attention_mask=encoder_extended_attention_mask

        if self.bert.encoder.gradient_checkpointing and self.bert.encoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i in range(self.num_backbone_layers):
            #print('LAYER', i)
            hidden_states_vision = self.vision_proj(hidden_states_vision)
            hidden_states_bert = self.text_proj(hidden_states_bert)

            if self.vision_model == 'mobilenet':
                # mobilenet - layer modules (inverted residuals)
                hidden_states_vision = self.mobilenet_v2.layer[i](hidden_states_vision)
            elif self.vision_model == 'vit':
                hidden_states_vision = self.vit.encoder.layer[i](hidden_states_vision)[0]

            # bert layer
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = self.bert.encoder.layer[i](
                hidden_states_bert,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value
            )
            hidden_states_bert = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

        self.cross_attn(hidden_states_bert, hidden_states_vision)

        if self.vision_model == 'mobilenet':
            # early fusion - cnn with mobilenet output + bert output as input
            #flat1 = self.pool_layer(hidden_states_vision).squeeze(-1).squeeze(-1)
            hidden_states_vision = self.pool_layer(hidden_states_vision).pooler_output
        elif self.vision_model == 'vit':
            #flat1 = hidden_states_vision[:, 0, :]
            hidden_states_vision = hidden_states_vision.pooler_output
        #flat2 = hidden_states_bert[:, 0, :]
        hidden_states_bert = hidden_states_bert.pooler_output

        concat = torch.cat((flat1, flat2), dim=1)

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
