import os
import cv2
import time
import threading
import torch
from transformers import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config

from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as rt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import importlib
latefusion = importlib.import_module("train.late-fusion")
midfusion = importlib.import_module("train.mid-fusion")
earlyfusion = importlib.import_module("train.early-fusion")

providers = ['CUDAExecutionProvider', 'CPUExecutionProvier']

if not torch.cuda.is_available():
    print('GPU not available, running script on CPU..')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = './test'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

filename = "beautifulweather"
image_path = os.path.join(save_dir, filename + '.jpg')
text_file = os.path.join(save_dir, filename + '.txt')

print(f"Analyzing image: {image_path}")
print(f"Analyzing text: {text_file}")

print('...loading data')
loaded_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_LINEAR)
pil_image = Image.fromarray(resized_img)
image = transforms.functional.pil_to_tensor(pil_image)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225])
image = normalize(image/255.0)

with open(text_file, 'r') as f:
    text = f.read()
print(f"Text content: {text}")

text = str(text)
text = " ".join(text.split())  # clean whitespace

inputs = TOKENIZER.encode_plus(
    text,
    None,
    add_special_tokens=True,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_token_type_ids=True
)

tokenized_text = {
    'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
    'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
    'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long)
}

def run_inference(model, model_type='early', trials=100, warm_up=10, model_name='default', acc='n/a'):
    total_inf_time = 0.0

    with torch.no_grad():
        for i in range(trials+warm_up):
            start_time = time.time()

            if model_type == 'late':
                out = model(images=image.to(DEVICE).unsqueeze(0),
                            input_ids=tokenized_text['ids'].to(DEVICE, dtype=torch.long).unsqueeze(0),
                            attention_mask=tokenized_text['mask'].to(DEVICE, dtype=torch.long).unsqueeze(0),
                            token_type_ids=tokenized_text['token_type_ids'].to(DEVICE, dtype=torch.long).unsqueeze(0))
            elif model_type == 'early':
                out = model(pixel_values=image.to(DEVICE).unsqueeze(0),
                            input_ids=tokenized_text['ids'].to(DEVICE, dtype=torch.long).unsqueeze(0),
                            attention_mask=tokenized_text['mask'].to(DEVICE, dtype=torch.long).unsqueeze(0),
                            token_type_ids=tokenized_text['token_type_ids'].to(DEVICE, dtype=torch.long).unsqueeze(0))
            elif model_type == 'mobilenet':
                out = model(image.to(DEVICE).unsqueeze(0))
            elif model_type == 'bert':
                out = model(input_ids=tokenized_text['ids'].to(DEVICE, dtype=torch.long).unsqueeze(0),
                            attention_mask=tokenized_text['mask'].to(DEVICE, dtype=torch.long).unsqueeze(0),
                            token_type_ids=tokenized_text['token_type_ids'].to(DEVICE, dtype=torch.long).unsqueeze(0))
            elif model_type == 'mobilenet_onnx':
                inputs = {'input': image.unsqueeze(0).numpy().astype(np.float32)}
                out = model.run(['output'], inputs)
                out = torch.tensor(out[0])
            elif model_type == 'bert_onnx':
                inputs = {'input_ids': tokenized_text['ids'].unsqueeze(0).numpy().astype(np.int64),
                          'attention_mask': tokenized_text['mask'].unsqueeze(0).numpy().astype(np.int64),
                          'token_type_ids': tokenized_text['token_type_ids'].unsqueeze(0).numpy().astype(np.int64)}
                out = model.run(['logits'], inputs)
                out = torch.tensor(out[0])
            elif model_type == 'fusion_onnx':
                inputs = {'images': image.unsqueeze(0).numpy().astype(np.float32),
                          'input_ids': tokenized_text['ids'].unsqueeze(0).numpy().astype(np.int64),
                          'attention_mask': tokenized_text['mask'].unsqueeze(0).numpy().astype(np.int64),
                          'token_type_ids': tokenized_text['token_type_ids'].unsqueeze(0).numpy().astype(np.int64)}
                out = model.run(['logits'], inputs)
                out = torch.tensor(out[0])
            prediction = torch.argmax(out, dim=1).item()

            if prediction == 1:
                pass
            elif prediction == 0:
                pass
            else:
                print('WARN: unexpected prediction:', prediction)

            if i >= warm_up:
                total_inf_time += time.time()-start_time
            #print('running avg inf speed:', total_inf_time/(i+1))
    avg_inf_time = total_inf_time/trials
    print(f'{model_name[:7]}\t| {str(acc)} | {avg_inf_time} s')

    return avg_inf_time


def run_models():
    print('Running inference..')
    mobilenet_onnx = rt.InferenceSession('./late-fusion-weights/mobilenet_state_dict.onnx', providers=providers)
    run_inference(mobilenet_onnx, 'mobilenet_onnx', model_name='Mobilenet-ONNX')
    del mobilenet_onnx

    bert_onnx = rt.InferenceSession('./late-fusion-weights/bert_state_dict.onnx', providers=providers)
    run_inference(bert_onnx, 'bert_onnx', model_name='BERT-ONNX')
    del bert_onnx

    late_onnx = rt.InferenceSession('./late-fusion-weights/late_fusion_state_dict.onnx', providers=providers)
    run_inference(late_onnx, 'fusion_onnx', model_name='Late-ONNX')
    del late_onnx

    mid_onnx = rt.InferenceSession('./models-mid-fusion_8-6-4-4-fuse/late_fusion_state_dict.onnx', providers=providers)
    run_inference(mid_onnx, 'fusion_onnx', model_name='MidF ONNX')
    del mid_onnx

    early_onnx = rt.InferenceSession('./models-early-fusion_4-attn-blocks/late_fusion_state_dict.onnx', providers=providers)
    run_inference(early_onnx, 'fusion_onnx', model_name='EarlF ONNX')
    del early_onnx

if __name__ == "__main__":
    run_models()
