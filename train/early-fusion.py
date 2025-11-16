# packages
import os
import torch
import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

from transformers.models.mobilenet_v2 import MobileNetV2Model, MobileNetV2PreTrainedModel
from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config

from typing import Optional, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention
from transformers.models.mobilenet_v2.modeling_mobilenet_v2 import apply_depth_multiplier, MobileNetV2Stem, MobileNetV2InvertedResidual, MobileNetV2ConvLayer

from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa


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


class EarlyFusionModel(torch.nn.Module):
    """
    Early fusion of MobileNetV2 and BERT.
    """
    def __init__(self, num_attn_net_blocks=4):
        super().__init__()
        self.num_labels = 2
        self.num_backbone_layers = 6

        cfg = BertConfig()
        self.bert = BertModel(cfg).from_pretrained('bert-base-uncased')

        cfg = MobileNetV2Config()
        self.mobilenet_v2 = MobileNetV2Model(cfg).from_pretrained('google/mobilenet_v2_1.4_224')
        self.num_intermediate_layers = len(self.mobilenet_v2.layer)

        # attention-based arch
        self.pool_layer = torch.nn.MaxPool2d(kernel_size=(14, 14), stride=1)
        self.attn_net = AttentionNet(856, num_blocks=num_attn_net_blocks)

        # classification head
        self.linear1 = torch.nn.Linear(856, 512)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(256, self.num_labels)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # mobilenet val
        labels=None,  # mobilenet val
        input_ids: Optional[torch.Tensor] = None,  # bert val
        attention_mask: Optional[torch.Tensor] = None,  # bert val
        token_type_ids: Optional[torch.Tensor] = None,  # bert val
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None
    ):  
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # mobilenet - conv stem
        hidden_states_mobilenet = self.mobilenet_v2.conv_stem(pixel_values)

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

        use_sdpa_attention_masks = (
            self.bert.attn_implementation == "sdpa"
            and self.bert.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

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
            # mobilenet - layer modules (inverted residuals)
            hidden_states_mobilenet = self.mobilenet_v2.layer[i](hidden_states_mobilenet)

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
        # early fusion - cnn with mobilenet output + bert output as input
        flat1 = self.pool_layer(hidden_states_mobilenet).squeeze(-1).squeeze(-1)
        flat2 = hidden_states_bert[:, 0, :]
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


if __name__ == "__main__":
    print('Training early fusion...')
    import os 
    os.environ['KAGGLEHUB_CACHE'] = './'
    import kagglehub
    path = kagglehub.dataset_download("a61979992/cmu-mosi")
    print("Path to dataset files:", path)

    # constants
    BATCH_SIZE = 32
    NUM_EPOCHS = 300
    NUM_WORKERS = 4

    if not torch.cuda.is_available():
        print('GPU not available, running script on CPU..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read in cmu mosi data
    import cv2
    import math
    import pandas as pd

    # process csv
    df = pd.read_csv('./datasets/a61979992/cmu-mosi/versions/1/label.csv')
    print(df.head)

    labels = df['annotation']
    mode = df['mode']
    texts = df['text']
    video_ids = df['video_id']
    clip_ids = df['clip_id']

    train_images = []
    train_texts = []
    train_labels = []
    valid_images = []
    valid_texts = []
    valid_labels = []
    test_images = []
    test_texts = []
    test_labels = []
    for i in range(len(labels)):
        file_path = f'./datasets/a61979992/cmu-mosi/versions/1/Raw_peak_frames/Raw_peak_frames/{str(video_ids[i])}/{str(clip_ids[i])}.jpg'
        frame = cv2.imread(file_path)

        if mode[i] == 'train':
            train_images.append(frame)
            train_texts.append(texts[i])
            train_labels.append(labels[i])
        elif mode[i] == 'valid':
            valid_images.append(frame)
            valid_texts.append(texts[i])
            valid_labels.append(labels[i])
        elif mode[i] == 'test':
            test_images.append(frame)
            test_texts.append(texts[i])
            test_labels.append(labels[i])
        else:
            print('error, invalid mode:', mode[i])
    print('...dataset read')

    class MultimodalDataset(torch.utils.data.Dataset):
        def __init__(self, images, texts, labels, val=False):
            self.images = images
            self.texts = texts
            self.labels = labels

            self.tokenizer = TOKENIZER
            
            self.transform = self._transform
            self.text_transform = self._text_transform
            self.target_transform = self._target_transform
            self.val = val

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image = self.images[idx]
            image = self.transform(image)
            text = self.texts[idx]
            text = self.text_transform(text)
            label = self.labels[idx]
            label = self.target_transform(label)

            return image, text, label

        def _transform(self, image):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            pil_image = Image.fromarray(resized_img)

            image = transforms.functional.pil_to_tensor(pil_image)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225])
            image = normalize(image/255.0)

            return image

        def _text_transform(self, text):
            text = str(text)
            text = " ".join(text.split())  # clean whitespace

            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_token_type_ids=True
            )

            return {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long)
            }

        def _target_transform(self, target):
            target = str(target)
            reduced_target = None

            if target == 'Negative':
                reduced_target = 1
            elif target == 'Positive':
                reduced_target = 0
            elif target == 'Neutral':
                reduced_target = 0
            else:
                print(f'ERROR: target {target} not an accepted value')
            
            return reduced_target

    # create dataloaders
    train_dataset = MultimodalDataset(train_images, train_texts, train_labels)
    val_dataset = MultimodalDataset(valid_images, valid_texts, valid_labels)
    test_dataset = MultimodalDataset(test_images, test_texts, test_labels)

    base_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    base_val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    base_test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # loss function
    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)

    # compute accuracy
    def compute_accuracy(outputs, targets):
        predictions = torch.argmax(outputs, dim=1)
        num_predictions = len(predictions)

        predictions = predictions.cpu()
        targets = targets.cpu()
        num_incorrect = 0
        for i in range(len(predictions)):
            if not predictions[i] == targets[i]:
                num_incorrect = num_incorrect + 1
        accuracy = (num_predictions-num_incorrect)/num_predictions

        return accuracy

    # train function
    def train(model, model_type, dataloader, device, epoch, num_epochs, total_steps):
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        for i, (images, texts, labels) in enumerate(dataloader):
            images = images.to(device)
            targets = labels.to(device)

            ids = texts['ids'].to(device, dtype=torch.long)
            mask = texts['mask'].to(device, dtype=torch.long)
            token_type_ids = texts['token_type_ids'].to(device, dtype=torch.long)

            if model_type == 'bert':
                outputs = model(ids, mask, token_type_ids)
            elif model_type == 'mobilenet':
                outputs = model(images)
            elif model_type == 'late_fusion':
                outputs = model(pixel_values=images, input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            
            loss = loss_fn(outputs, targets)
            accuracy = compute_accuracy(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy

            if (i+1) % 50 == 0:
                print(
                    f'TRAINING --> Epoch: {epoch+1}/{num_epochs}, ' +
                    f'Step: {i+1}/{total_steps}, ' +
                    f'Loss: {running_loss / (i+1)}, '
                    f'Accuracy: {running_acc / (i+1)}'
                )
        running_loss = running_loss / total_steps
        running_acc = running_acc / total_steps

        return running_loss, running_acc

    # validate function
    def validate(model, model_type, dataloader, device, epoch, num_epochs, total_steps):
        running_loss = 0.0
        running_acc = 0.0
        model.eval()

        with torch.no_grad():
            for i, (images, texts, labels) in enumerate(dataloader):
                images = images.to(device)
                targets = labels.to(device)

                ids = texts['ids'].to(device, dtype=torch.long)
                mask = texts['mask'].to(device, dtype=torch.long)
                token_type_ids = texts['token_type_ids'].to(device, dtype=torch.long)

                if model_type == 'bert':
                    outputs = model(ids, mask, token_type_ids)
                elif model_type == 'mobilenet':
                    outputs = model(images)
                elif model_type == 'late_fusion':
                    outputs = model(pixel_values=images, input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        
                loss = loss_fn(outputs, targets)
                accuracy = compute_accuracy(outputs, targets)
        
                running_loss += loss.item()
                running_acc += accuracy
        
                if (i+1) % 50 == 0:
                    print(
                        f'VALIDATION --> Epoch: {epoch+1}/{num_epochs}, ' +
                        f'Step: {i+1}/{total_steps}, ' +
                        f'Loss: {running_loss / (i+1)}, '
                        f'Accuracy: {running_acc / (i+1)}'
                    )
        running_loss = running_loss / total_steps
        running_acc = running_acc / total_steps

        return running_loss, running_acc

    def save_best_model(
        model: torch.nn.Module,
        model_save_path,
        val_loss: float,
        val_losses: list,
        epoch: int,
        keep_models: bool = False,
        model_type: str = None
    ):
        """Save the model if it is the first epoch. Subsequently, save the model
        only if a lower validation loss is achieved whilst training.

        :param model: The model to save.
        :type model: torch.nn.Module
        :param model_save_path: The location to save the model to.
        :type model_save_path: Path
        :param val_loss: The current epoch's validation loss.
        :type val_loss: float
        :param val_losses: The history of all other validation losses.
        :type val_losses: list
        :param epoch: The current epoch number.
        :type epoch: int
        :param keep_models: Should all models be saved, defaults to False
        :type keep_models: bool, optional
        """
        # Should we keep all models or just one
        if keep_models:
            model_save_path = model_save_path / f'model_{epoch+1}_{val_loss}.pt'
        else:
            model_save_path = model_save_path / f'{model_type}_state_dict.pt'
        # Save the first model
        if len(val_losses) == 0:
            torch.save(
                model.state_dict(),
                model_save_path
            )
            print(
                'SAVING --> First epoch: \n' +
                f'Val Loss: {val_loss}\n' +
                f'Saving new model to:\n{model_save_path}'
            )
        elif val_loss < min(val_losses):
            # If our new validation loss is less than the previous best save the
            # model
            print(
                'SAVING --> Found model with better validation loss: \n' +
                f'New Best Val Loss: {val_loss}\n' +
                f'Old Best Val Loss: {min(val_losses)}\n'
                f'Saving new model to:\n{model_save_path}'
            )
            torch.save(
                model.state_dict(),
                model_save_path
            )
        return model_save_path

    def plot_epoch_metrics(x, y, data_names, title_prefix, yaxis_label):
        """Plot metrics with the number of epochs on the x axis and the metric of
        interest on the y axis. Note that this function differs based on the input.

        :param x: The values to use on the x-axis.
        :type x: list
        :param y: A list of lists containing len(x) data points to plot. The inner
            lists are the different series to plot.
        :type y: list
        :param data_names: Names of the series to use in the legend.
        :type data_names: str
        :param title_prefix: A prefix to add before everything else in the title.
        :type title_prefix: str
        :param yaxis_label: The label for the y axis.
        :type yaxis_label: str
        """
        # Plot multiple series of data
        for i in y:
            plt.plot(x, i)
        # Set the title
        plt.title(title_prefix + ' ' + ' vs. '.join(data_names) + ' ' + yaxis_label)
        # Set the y axis label
        plt.ylabel(yaxis_label)
        # Enable the legend with the appropriate names
        plt.legend(data_names)

    # train_loop
    def train_loop(model, model_type, train_dataloader, val_dataloader, device, num_epochs, model_save_path=Path('./models-early-fusion')):
        print(f'Models will be saved to: {model_save_path}')
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        if not os.path.isdir(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        train_total_steps = len(train_dataloader)
        val_total_steps = len(val_dataloader)

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train(model, model_type, train_dataloader, device, epoch, num_epochs, train_total_steps)
            print(
                f'TRAINING --> Epoch {epoch+1}/{num_epochs} DONE, ' +
                f'Avg Loss: {train_loss}, Avg Accuracy: {train_accuracy}'
            )

            val_loss, val_accuracy = validate(model, model_type, val_dataloader, device, epoch, num_epochs, val_total_steps)
            print(
                f'VALIDATION --> Epoch {epoch+1}/{num_epochs} DONE, ' +
                f'Avg Loss: {val_loss}, Avg Accuracy: {val_accuracy}'
            )

            new_saved_model_path = save_best_model(model, model_save_path, val_loss, val_losses, epoch, False, model_type)
            
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)
        return (train_losses, train_accs), (val_losses, val_accs), new_saved_model_path

    # evaluate function (test data)
    def evaluate(model, model_type, dataloader, device, total_steps):
        model.eval()
        running_loss = 0.0
        running_acc = 0.0

        with torch.no_grad():
            for i, (images, texts, labels) in enumerate(dataloader):
                images = images.to(device)
                targets = labels.to(device)

                ids = texts['ids'].to(device, dtype=torch.long)
                mask = texts['mask'].to(device, dtype=torch.long)
                token_type_ids = texts['token_type_ids'].to(device, dtype=torch.long)

                if model_type == 'bert':
                    outputs = model(ids, mask, token_type_ids)
                elif model_type == 'mobilenet':
                    outputs = model(images)
                elif model_type == 'late_fusion':
                    outputs = model(pixel_values=images, input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

                loss = loss_fn(outputs, targets)
                accuracy = compute_accuracy(outputs, targets)
        
                running_loss += loss.item()
                running_acc += accuracy
        
                if (i+1) % 256 == 0:
                    print(
                        f'TEST' +
                        f'Step: {i+1}/{total_steps}, ' +
                        f'Loss: {running_loss / (i+1)}, '
                        f'Accuracy: {running_acc / (i+1)}'
                    )
        running_loss = running_loss / total_steps
        running_acc = running_acc / total_steps

        return running_loss, running_acc

    #ablation_study_attn_blocks = [2, 4, 6, 8]
    ablation_study_attn_blocks = [4]

    for idx in ablation_study_attn_blocks:
        model_save_path = Path(f'./models-early-fusion_{str(idx)}-attn-blocks')
        print('...creating early fusion model')
        early_fusion_model = EarlyFusionModel(idx)
        early_fusion_model.to(DEVICE)

        # run training
        print('training early fusion model..')
        optimizer = torch.optim.SGD(early_fusion_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
        (train_losses, train_accs), (val_losses, val_accs), path_to_model = train_loop(early_fusion_model, 'late_fusion', base_train_dataloader, base_val_dataloader, DEVICE, NUM_EPOCHS, model_save_path=model_save_path)
        print(f'Best Validation Loss: {min(val_losses)} after epoch {np.argmin(val_losses) + 1}')
        print(f'Best Validation Acc: {max(val_accs)} after epoch {np.argmax(val_accs) + 1}')

        plot_epoch_metrics(
            np.arange(NUM_EPOCHS),
            [train_losses, val_losses],
            ['Train', 'Validation'],
            'Early Fusion',
            'Loss'
        )

        plot_epoch_metrics(
            np.arange(NUM_EPOCHS),
            [train_accs, val_accs],
            ['Train', 'Validation'],
            'Early Fusion',
            'Accuracy'
        )

        # run evaluation (test)
        early_fusion_model = EarlyFusionModel(idx)
        early_fusion_model.load_state_dict(torch.load(str(model_save_path)+'/late_fusion_state_dict.pt'))
        early_fusion_model.to(DEVICE)

        test_loss, test_accuracy = evaluate(early_fusion_model, 'late_fusion', base_test_dataloader, DEVICE, len(base_test_dataloader))
        print(
            f'TEST (early fusion)--> DONE, ' +
            f'Avg Loss: {test_loss}, Avg Accuracy: {test_accuracy}'
        )

