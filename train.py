import os
import lib
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from cmumosi import load_dataset

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
    parser.add_argument('--path-mobilenet', type=str, required=False)
    parser.add_argument('--path-vit', type=str, required=False)
    parser.add_argument('--path-bert', type=str, required=False)
    parser.add_argument('--debug', type=bool, default=False, required=False)
    args = parser.parse_args()

    arch = args.arch
    # FIXME: args.debug sets to True
    DEBUG = False # args.debug
    BATCH_SIZE = 32
    N_WORKERS = 4

    seed()

    # load model
    if arch == 'mobilenet':
        from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config
        cfg = MobileNetV2Config()
        from models import MobileNetV2ForFacialExpressionRecognition
        model = MobileNetV2ForFacialExpressionRecognition(cfg)
        n_epochs = 150
        dataloaders, _ = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=True)
        train_dataloader, val_dataloader, test_dataloader = dataloaders
    elif arch == 'vit':
        from transformers.models.vit.configuration_vit import ViTConfig
        cfg = ViTConfig()
        from models import ViTForFacialExpressionRecognition
        model = ViTForFacialExpressionRecognition(cfg)
        n_epochs = 150
        dataloaders, _ = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=True)
        train_dataloader, val_dataloader, test_dataloader = dataloaders
    elif arch == 'bert':
        from transformers.models.bert.configuration_bert import BertConfig
        cfg = BertConfig()
        from models import BERTForSentimentAnalysis
        model = BERTForSentimentAnalysis(cfg)
        n_epochs = 12
        dataloaders, _ = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=True)
        train_dataloader, val_dataloader, test_dataloader = dataloaders
    elif arch == 'latefusion':
        if args.path_mobilenet == None and args.path_vit == None:
            raise ValueError('Path to existing vision model needed to train LateFusion. Enter path or \"scratch\" to train from scratch now.')
        elif args.path_mobilenet == 'scratch':
            print('ERR: training mobilenet from scratch not implemented yet.')
            exit()
        elif args.path_vit == 'scratch':
            print('ERR: training vit from scratch not implemented yet.')
            exit()
        if args.path_bert == None:
            raise ValueError('Path to existing bert model needed to train LateFusion. Enter path or \"scratch\" to train from scratch now.')
        elif args.path_bert == 'scratch':
            print('ERR: training bert from scratch not implemented yet.')
            exit()
        from models import LateFusion
        # vision model
        if args.path_mobilenet:
            model = LateFusion('mobilenet', args.path_mobilenet, args.path_bert)
            arch = arch + 'mobilenet'
        elif args.path_vit:
            model = LateFusion('vit', args.path_vit, args.path_bert)
            arch = arch + 'vit'
        _, dataloaders = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=True)
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        n_epochs = 100
    elif arch == 'midfusion':
        from models import MidFusion
        model = MidFusion(args.vision_model)
        arch = arch + args.vision_model
        dataloaders = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=False)
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        n_epochs = 100
    elif arch == 'earlyfusion':
        from models import EarlyFusion
        model = EarlyFusion(args.vision_model)
        arch = arch + args.vision_model
        dataloaders = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=False)
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        n_epochs = 300
    else:
        raise ValueError(f'Invalid model architecture {arch}')

    if not torch.cuda.is_available():
        print('GPU not available, running script on CPU..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    if DEBUG:
        n_epochs = 1

    # train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

    def loss_fn(outputs, targets): return torch.nn.CrossEntropyLoss()(outputs, targets)

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
    print('Training model..')
    (train_losses, train_accs), (val_losses, val_accs), path = lib.train_loop(model, arch, train_dataloader, val_dataloader, DEVICE, optimizer, n_epochs, loss_fn, compute_accuracy, save_path=Path(arch+'_out'))

    # evaluate
    print('Loading best model from training for evalution..')
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    print('Evaluating model..')
    test_loss, test_accuracy = lib.evaluate(model, arch, test_dataloader, DEVICE, len(test_dataloader), loss_fn, compute_accuracy)

    # TODO: save metrics to CSV
