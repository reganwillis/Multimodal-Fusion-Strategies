import os
import cv2
import torch
import kagglehub
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import BertTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, images, texts, labels, val=False):
        self.images = images
        self.texts = texts
        self.labels = labels

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)

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


def load_dataset(batch_size, n_workers, finetune=False):
    os.environ['KAGGLEHUB_CACHE'] = './'
    path = kagglehub.dataset_download('a61979992/cmu-mosi')
    print('Loading dataset:', path)

    df = pd.read_csv('./datasets/a61979992/cmu-mosi/versions/1/label.csv')
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

    # create dataloaders
    train_dataset = MultimodalDataset(train_images, train_texts, train_labels)
    val_dataset = MultimodalDataset(valid_images, valid_texts, valid_labels, True)
    test_dataset = MultimodalDataset(test_images, test_texts, test_labels, True)

    if finetune:
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

        base_train_dataloader = torch.utils.data.DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        base_val_dataloader = torch.utils.data.DataLoader(base_val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        base_test_dataloader = torch.utils.data.DataLoader(base_test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        fine_tune_train_dataloader = torch.utils.data.DataLoader(fine_tune_train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        fine_tune_val_dataloader = torch.utils.data.DataLoader(fine_tune_val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        fine_tune_test_dataloader = torch.utils.data.DataLoader(fine_tune_test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        return (
            (base_train_dataloader, base_val_dataloader, base_test_dataloader),
            (fine_tune_train_dataloader, fine_tune_val_dataloader, fine_tune_test_dataloader)
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        return ((train_dataloader, val_dataloader, test_dataloader))
