import os
import lib
import csv
import time
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from cmumosi import load_dataset, display_dataset_examples

import warnings
warnings.filterwarnings("ignore", module="huggingface_hub")

def seed(seed=222):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--vision-model', type=str, required=False)
    parser.add_argument('--cross-attn-fusion', action='store_true')
    parser.add_argument('--attn-fusion', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--hyperparam-study', action='store_true')
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()

    arch = args.arch
    args.attn_fusion = True
    DEBUG = args.debug
    BATCH_SIZE = 32
    N_WORKERS = 4

    if args.debug:
        n_trials = 1
        warmup = 0
    else:
        n_trials = 20
        warmup = 5

    seed()

    # load model
    if arch == 'mobilenet':
        from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config
        cfg = MobileNetV2Config()
        from models import MobileNetV2ForFacialExpressionRecognition
        model = MobileNetV2ForFacialExpressionRecognition(cfg)
    elif arch == 'vit':
        from transformers.models.vit.configuration_vit import ViTConfig
        cfg = ViTConfig()
        from models import ViTForFacialExpressionRecognition
        model = ViTForFacialExpressionRecognition(cfg)
    elif arch == 'bert':
        from models import BERTForSentimentAnalysis
        model = BERTForSentimentAnalysis()
    elif arch == 'latefusion':
        from models import LateFusion
        model = LateFusion(args.vision_model, args.cross_attn_fusion)
        arch = arch + args.vision_model
    elif arch == 'midfusion':
        from models import MidFusion
        model = MidFusion(args.vision_model, args.cross_attn_fusion, args.attn_fusion)
        arch = arch + args.vision_model
    elif arch == 'earlyfusion':
        from models import EarlyFusion
        model = EarlyFusion(args.vision_model, args.cross_attn_fusion)
        arch = arch + args.vision_model
    else:
        raise ValueError(f'Invalid model architecture {arch}')

    if not torch.cuda.is_available():
        print('GPU not available, running script on CPU..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    dataloaders = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, face_crop=False)
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    print('Loading model weights for evalution..')
    model.load_state_dict(torch.load(args.weights, weights_only=False, map_location=DEVICE))
    model.to(DEVICE)
    #model_onnx = rt.InferenceSession(args.weights, providers=providers)
    print('Evaluating model..')
    latency = lib.evaluate_latency(model, arch, test_dataloader, DEVICE, n_trials, warmup)
