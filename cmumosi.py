import os
import cv2
import torch
import kagglehub
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def crop_to_face(image, face_det_model, padding=0.15):
    # detect face
    output = face_det_model.predict(source=image)

    if len(output[0].boxes) >= 1:
        # get best match of face
        boxes = output[0].boxes.xyxy.cpu().numpy()
        best_match = output[0].boxes.conf.cpu().numpy().argmax()
        x1, y1, x2, y2 = boxes[best_match]

        # crop to face
        h, w = image.shape[:2]
        bw = x2 - x1
        bh = y2 - y1

        x1 = max(0, int(x1 - bw * padding))
        y1 = max(0, int(y1 - bh * padding))
        x2 = min(w, int(x2 + bw * padding))
        y2 = min(h, int(y2 + bh * padding))

        return image[y1:y2, x1:x2]
    else:
        print('WARN: found {len(output[0].boxes)} faces in image. Returning full image.')
        return image


def preprocess_face_cropping(df, raw_frames_dir, out_dir, recrop=True):
    os.makedirs(out_dir, exist_ok=True)

    # face detection model for cropping to face
    path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    face_det_model = YOLO(path)

    crop_paths = []
    for i in range(len(df)):
        video_id = str(df.loc[i, "video_id"])
        clip_id = str(df.loc[i, "clip_id"])
        mode = df.loc[i, "mode"]

        raw_path = os.path.join(raw_frames_dir, video_id, f"{clip_id}.jpg")
        split_dir = os.path.join(out_dir, mode)
        os.makedirs(split_dir, exist_ok=True)
        crop_path = os.path.join(split_dir, f"{video_id}_{clip_id}.jpg")

        if recrop:
            image = cv2.imread(raw_path)
            cropped = crop_to_face(image, face_det_model)
            cv2.imwrite(crop_path, cropped)
        crop_paths.append(crop_path)

    df = df.copy()
    df["crop_path"] = crop_paths
    return df


class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, images, texts, labels, val=False):
        self.images = images
        self.texts = texts
        self.labels = labels

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

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

        inputs = self.tokenizer(
            text,
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


def load_dataset(batch_size, n_workers, finetune=False, face_crop=True, recrop=False):
    os.environ['KAGGLEHUB_CACHE'] = './'
    path = kagglehub.dataset_download('a61979992/cmu-mosi')
    print('Loading dataset:', path)
    csv_path = './datasets/a61979992/cmu-mosi/versions/1/label.csv'
    file_path = f'./datasets/a61979992/cmu-mosi/versions/1/Raw_peak_frames/Raw_peak_frames/'

    df = pd.read_csv(csv_path)

    if face_crop:
        print('Face crop enabled.')
        face_crops_path = './datasets/a61979992/cmu-mosi/versions/1/face_crops'
        if os.path.exists(face_crops_path) and recrop == False:
            print('Cropped faces already saved, choose recrop option to reprocess.')
            cropped_df = preprocess_face_cropping(df, file_path, face_crops_path, False)
        else:
            print('Cropping faces in dataset..')
            cropped_df = preprocess_face_cropping(df, file_path, face_crops_path)

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
        if face_crop:
            file_path = cropped_df.loc[i, "crop_path"]
        else:
            file_path = file_path + '{str(video_ids[i])}/{str(clip_ids[i])}.jpg'
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


def display_dataset_examples(dataloader, n_samples=5, save_image_locally=True):
    dataset = dataloader.dataset
    label_map = {0: "Non-negative", 1: "Negative"}

    for idx in range(n_samples):
        raw_image = dataset.images[idx]
        raw_text = dataset.texts[idx]
        raw_label = dataset.labels[idx]

        print(f'\nDataset Sample {idx}')
        print(f'Text: {raw_text}')
        print(f'Label: {raw_label}')

        if save_image_locally:
            cv2.imwrite(f'./{idx}.jpg', raw_image)
        else:
            plt.figure(figsize=(5,5))
            plt.imshow(raw_image[:,:,::-1])
            plt.title(f'Dataset Sample {idx}; Label: {raw_label}')
            plt.axis('off')
