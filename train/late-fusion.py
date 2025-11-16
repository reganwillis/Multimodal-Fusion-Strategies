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

from transformers.models.mobilenet_v2 import MobileNetV2Model
from transformers.models.mobilenet_v2 import MobileNetV2PreTrainedModel
from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config

from transformers.models.bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)


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


class BERTForSentimentAnalysis(BertModel):
    """ 
    from Bert For Sequence Classification
    """
    def __init__(self, config, multimodal=False):
        super().__init__(config=config)
        self.num_labels = 2
        self.config = config
        self.multimodal = multimodal

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,):         
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        pooled_output = outputs[1]

        if self.multimodal:
            return pooled_output
        else:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            return logits


class LateFusionModel(torch.nn.Module):

    def __init__(self, fer_weights, bert_weights):
        super().__init__()

        fer_config = MobileNetV2Config()
        self.fer = MobileNetV2ForFacialExpressionRecognition(fer_config, multimodal=True)
        self.fer.load_state_dict(torch.load(fer_weights))
        self.fer.multimodal = True
        self.fer.eval()
        for name, param in self.fer.named_parameters():
            param.requires_grad = False

        bert_config = BertConfig()
        self.bert = BERTForSentimentAnalysis(bert_config, multimodal=True)
        self.bert.load_state_dict(torch.load(bert_weights))
        self.bert.multimodal = True
        self.bert.eval()
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        fer_output_size = 1792
        bert_output_size = 768

        self.linear = torch.nn.Linear(fer_output_size+bert_output_size, 512)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fusion_layer = torch.nn.Linear(256, 2)

    def forward(self, images=None, input_ids=None, attention_mask=None, token_type_ids=None):
        # call forward pass of fer and bert
        fer_output = self.fer(images)
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)

        output = torch.cat((fer_output, bert_output), dim=1)
        x = self.linear(output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        result = self.fusion_layer(x)

        return result


if __name__ == "__main__":
    print('Starting late fusion training...')

    import os
    os.environ['KAGGLEHUB_CACHE'] = './'
    import kagglehub
    # Download latest version
    path = kagglehub.dataset_download("reganw/cmu-mosi")
    print("Path to dataset files:", path)

    # constants
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    if not torch.cuda.is_available():
        print('GPU not available, running script on CPU..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read in cmu mosi data
    import cv2
    import math
    import pandas as pd

    # process csv
    df = pd.read_csv('./datasets/reganw/cmu-mosi/versions/2/label.csv')
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
        file_path = f'./datasets/reganw/cmu-mosi/versions/2/Raw_peak_frames/Raw_peak_frames/{str(video_ids[i])}/{str(clip_ids[i])}.jpg'
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


    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from PIL import Image

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

            if not self.val:
                flip = transforms.RandomHorizontalFlip(p=0.5)
                pil_image = flip(pil_image)
                crop = transforms.RandomResizedCrop(size=pil_image.size, scale=(0.08, 1.0), ratio=(0.75, 1.33))
                pil_image = crop(pil_image)
                affine = transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))
                pil_image = affine(pil_image)
                color = transforms.ColorJitter(brightness=(0.5, 1.5), saturation=0.2, hue=0.1)
                pil_image = color(pil_image)

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
                print(f'ERROR: target {target} not in accepted range')
            
            return reduced_target

    # create dataloaders
    train_dataset = MultimodalDataset(train_images, train_texts, train_labels)
    val_dataset = MultimodalDataset(valid_images, valid_texts, valid_labels, True)
    test_dataset = MultimodalDataset(test_images, test_texts, test_labels, True)

    # split for fine-tuning fusion model
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    torch.manual_seed(50)

    base_train_size = int(0.8 * len(train_dataset))
    fine_tune_train_size = len(train_dataset) - base_train_size
    base_train_dataset, fine_tune_train_dataset = random_split(train_dataset, [base_train_size, fine_tune_train_size])

    base_val_size = int(0.8 * len(val_dataset))
    fine_tune_val_size = len(val_dataset) - base_val_size
    base_val_dataset, fine_tune_val_dataset = random_split(val_dataset, [base_val_size, fine_tune_val_size])

    base_test_size = int(0.8 * len(test_dataset))
    fine_tune_test_size = len(test_dataset) - base_test_size
    base_test_dataset, fine_tune_test_dataset = random_split(test_dataset, [base_test_size, fine_tune_test_size])

    torch.set_rng_state(cpu_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)

    base_train_dataloader = torch.utils.data.DataLoader(base_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    base_val_dataloader = torch.utils.data.DataLoader(base_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    base_test_dataloader = torch.utils.data.DataLoader(base_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    fine_tune_train_dataloader = torch.utils.data.DataLoader(fine_tune_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    fine_tune_val_dataloader = torch.utils.data.DataLoader(fine_tune_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    fine_tune_test_dataloader = torch.utils.data.DataLoader(fine_tune_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print('...creating BERT model')
    cfg = BertConfig()
    bert_model = BERTForSentimentAnalysis(cfg)
    bert_model.to(DEVICE)

    print('...creating MobileNetV2 model')
    cfg = MobileNetV2Config()
    mobilenet_model = MobileNetV2ForFacialExpressionRecognition(cfg)
    mobilenet_model.to(DEVICE)

    # loss function
    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)

    # compute accuracy
    def compute_accuracy(outputs, targets):
        predictions = torch.argmax(outputs, 1)
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
                outputs = model(images=images, input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

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
                    outputs = model(images=images, input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
        
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
        model_type: str = None,
        acc_mode: bool = False,
        val_acc = None,
        val_accs = None
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
        if not acc_mode:
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
        else:
            if len(val_accs) == 0:
                torch.save(
                    model.state_dict(),
                    model_save_path
                )
                print(
                    'SAVING --> First epoch: \n' +
                    f'Val Acc: {val_acc}\n' +
                    f'Saving new model to:\n{model_save_path}'
                )
            elif val_acc > max(val_accs):
                print(
                    'SAVING --> Found model with better validation accuracy: \n' +
                    f'New Best Val Acc: {val_acc}\n' +
                    f'Old Best Val Acc: {max(val_accs)}\n'
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
    def train_loop(model, model_type, train_dataloader, val_dataloader, device, num_epochs, model_save_path=Path('./late-fusion-weights')):
        print(f'Models will be saved to: {model_save_path}')  # modelpath
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        if not model_save_path.exists():
            model_save_path.mkdir(exist_ok=True, parents=True)

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

            new_saved_model_path = save_best_model(model, model_save_path, val_loss, val_losses, epoch, False, model_type, True, val_accuracy, val_accs)
            
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)
        return (train_losses, train_accs), (val_losses, val_accs), new_saved_model_path

    fer_config = MobileNetV2Config()
    mobilenet_model = MobileNetV2ForFacialExpressionRecognition(fer_config, multimodal=False)
    mobilenet_model.to(DEVICE)

    # run training
    print('training mobilenet...')
    NUM_EPOCHS = 150
    optimizer = torch.optim.SGD(mobilenet_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    (train_losses, train_accs), (val_losses, val_accs), path_to_mobilenet = train_loop(mobilenet_model, 'mobilenet', base_train_dataloader, base_val_dataloader, DEVICE, NUM_EPOCHS)
    print(f'Best Validation Loss: {min(val_losses)} after epoch {np.argmin(val_losses) + 1}')
    print(f'Best Validation Acc: {max(val_accs)} after epoch {np.argmax(val_accs) + 1}')

    # run training
    print('training bert..')
    NUM_EPOCHS = 12
    optimizer = torch.optim.Adam(params=bert_model.parameters(), lr=1e-05, betas=(0.9, 0.999), eps=1e-08)
    (train_losses, train_accs), (val_losses, val_accs), path_to_bert = train_loop(bert_model, 'bert', base_train_dataloader, base_val_dataloader, DEVICE, NUM_EPOCHS)
    print(f'Best Validation Loss: {min(val_losses)} after epoch {np.argmin(val_losses) + 1}')
    print(f'Best Validation Acc: {max(val_accs)} after epoch {np.argmax(val_accs) + 1}')

    print('...creating LateFusionModel model')
    late_fusion_model = LateFusionModel(path_to_mobilenet, path_to_bert)
    late_fusion_model.to(DEVICE)

    optimizer = torch.optim.SGD(late_fusion_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    # run training
    print('training late fusion model..')
    NUM_EPOCHS = 100
    (train_losses, train_accs), (val_losses, val_accs), path_to_late_fusion_model = train_loop(late_fusion_model, 'late_fusion', fine_tune_train_dataloader, fine_tune_val_dataloader, DEVICE, NUM_EPOCHS)
    print(f'Best Validation Loss: {min(val_losses)} after epoch {np.argmin(val_losses) + 1}')
    print(f'Best Validation Acc: {max(val_accs)} after epoch {np.argmax(val_accs) + 1}')

    plot_epoch_metrics(
        np.arange(NUM_EPOCHS),
        [train_losses, val_losses],
        ['Train', 'Validation'],
        'Late Fusion',
        'Loss'
    )

    plot_epoch_metrics(
        np.arange(NUM_EPOCHS),
        [train_accs, val_accs],
        ['Train', 'Validation'],
        'Late Fusion',
        'Accuracy'
    )

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
                    outputs = model(images=images, input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)

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

    # run evaluation (test)

    bert_config = BertConfig()
    bert_model = BERTForSentimentAnalysis(bert_config, multimodal=True)
    bert_model.load_state_dict(torch.load(path_to_bert))
    bert_model.multimodal = False
    bert_model.to(DEVICE)

    test_loss, test_accuracy = evaluate(bert_model, 'bert', base_test_dataloader, DEVICE, len(base_test_dataloader))
    print(
        f'TEST (bert)--> DONE, ' +
        f'Avg Loss: {test_loss}, Avg Accuracy: {test_accuracy}'
    )

    fer_config = MobileNetV2Config()
    mobilenet_model = MobileNetV2ForFacialExpressionRecognition(fer_config, multimodal=False)
    mobilenet_model.load_state_dict(torch.load(path_to_mobilenet))
    mobilenet_model.multimodal = False
    mobilenet_model.to(DEVICE)

    test_loss, test_accuracy = evaluate(mobilenet_model, 'mobilenet', base_test_dataloader, DEVICE, len(base_test_dataloader))
    print(
        f'TEST (mobilenet)--> DONE, ' +
        f'Avg Loss: {test_loss}, Avg Accuracy: {test_accuracy}'
    )

    late_fusion_model = LateFusionModel(path_to_mobilenet, path_to_bert)
    late_fusion_model.load_state_dict(torch.load(path_to_late_fusion_model))
    late_fusion_model.to(DEVICE)

    test_loss, test_accuracy = evaluate(late_fusion_model, 'late_fusion', fine_tune_test_dataloader, DEVICE, len(fine_tune_test_dataloader))
    print(
        f'TEST (late fusion)--> DONE, ' +
        f'Avg Loss: {test_loss}, Avg Accuracy: {test_accuracy}'
    )


    ############################## ------------------------------- #####################################
    print('LATE FUSION TRAINING (ViT)')
    from transformers.models.vit import ViTModel
    from transformers.models.vit import ViTPreTrainedModel
    from transformers.models.vit.configuration_vit import ViTConfig
    class ViTForFacialExpressionRecognition(ViTPreTrainedModel):
        """
        from ViT for image classification
        """
        def __init__(self, config, multimodal=False):
            super().__init__(config)

            self.num_labels = 2
            self.vit = ViTModel(config, add_pooling_layer=False).from_pretrained('google/vit-base-patch16-224')
            self.multimodal=multimodal

            # Classifier head
            self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

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
            pooled_output = sequence_output[:, 0, :]

            if self.multimodal:
                return pooled_output
            else:
                logits = self.classifier(pooled_output)

                return logits

    print('...creating ViT model')
    cfg = ViTConfig()
    vit_model = ViTForFacialExpressionRecognition(cfg)
    vit_model.to(DEVICE)
    class LateFusionModel_withViT(torch.nn.Module):

        def __init__(self, fer_weights, bert_weights):
            super().__init__()

            fer_config = ViTConfig()
            self.fer = ViTForFacialExpressionRecognition(fer_config, multimodal=True)
            self.fer.load_state_dict(torch.load(fer_weights))
            self.fer.multimodal = True
            self.fer.eval()
            for name, param in self.fer.named_parameters():
                param.requires_grad = False

            bert_config = BertConfig()
            self.bert = BERTForSentimentAnalysis(bert_config, multimodal=True)
            self.bert.load_state_dict(torch.load(bert_weights))
            self.bert.multimodal = True
            self.bert.eval()
            for name, param in self.bert.named_parameters():
                param.requires_grad = False

            fer_output_size = 768
            bert_output_size = 768

            self.linear = torch.nn.Linear(fer_output_size+bert_output_size, 512)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.3)
            self.linear2 = torch.nn.Linear(512, 256)
            self.relu2 = torch.nn.ReLU()
            self.dropout2 = torch.nn.Dropout(0.3)
            self.fusion_layer = torch.nn.Linear(256, 2)

        def forward(self, images=None, input_ids=None, attention_mask=None, token_type_ids=None):
            # call forward pass of fer and bert
            fer_output = self.fer(images)
            bert_output = self.bert(input_ids, attention_mask, token_type_ids)

            output = torch.cat((fer_output, bert_output), dim=1)
            x = self.linear(output)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            result = self.fusion_layer(x)

            return result
    print('training vit...')
    NUM_EPOCHS = 150
    optimizer = torch.optim.SGD(vit_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    (train_losses, train_accs), (val_losses, val_accs), path_to_vit = train_loop(vit_model, 'mobilenet', base_train_dataloader, base_val_dataloader, DEVICE, NUM_EPOCHS, model_save_path=Path('./vit-weights'))
    print(f'Best Validation Loss: {min(val_losses)} after epoch {np.argmin(val_losses) + 1}')
    print(f'Best Validation Acc: {max(val_accs)} after epoch {np.argmax(val_accs) + 1}')
    print('...creating LateFusionModel (with ViT) model')
    late_fusion_model = LateFusionModel_withViT(path_to_vit, path_to_bert)
    late_fusion_model.to(DEVICE)
    optimizer = torch.optim.SGD(late_fusion_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    # run training
    print('training late fusion model..')
    NUM_EPOCHS = 100
    (train_losses, train_accs), (val_losses, val_accs), path_to_late_fusion_model = train_loop(late_fusion_model, 'late_fusion', fine_tune_train_dataloader, fine_tune_val_dataloader, DEVICE, NUM_EPOCHS, model_save_path=Path('./vit-weights'))
    print(f'Best Validation Loss: {min(val_losses)} after epoch {np.argmin(val_losses) + 1}')
    print(f'Best Validation Acc: {max(val_accs)} after epoch {np.argmax(val_accs) + 1}')

    fer_config = ViTConfig()
    vit_model = ViTForFacialExpressionRecognition(fer_config, multimodal=False)
    vit_model.load_state_dict(torch.load(path_to_vit))
    vit_model.multimodal = False
    vit_model.to(DEVICE)

    test_loss, test_accuracy = evaluate(vit_model, 'mobilenet', base_test_dataloader, DEVICE, len(base_test_dataloader))
    print(
        f'TEST (vit)--> DONE, ' +
        f'Avg Loss: {test_loss}, Avg Accuracy: {test_accuracy}'
    )

    late_fusion_model = LateFusionModel_withViT(path_to_vit, path_to_bert)
    late_fusion_model.load_state_dict(torch.load(path_to_late_fusion_model))
    late_fusion_model.to(DEVICE)

    test_loss, test_accuracy = evaluate(late_fusion_model, 'late_fusion', fine_tune_test_dataloader, DEVICE, len(fine_tune_test_dataloader))
    print(
        f'TEST (late fusion)--> DONE, ' +
        f'Avg Loss: {test_loss}, Avg Accuracy: {test_accuracy}'
    )
