import os
import lib
import csv
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from cmumosi import load_dataset, display_dataset_examples

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
    parser.add_argument('--cross-attn-fusion', action="store_true")
    parser.add_argument('--attn-fusion', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--path-mobilenet', type=str, required=False)
    parser.add_argument('--path-vit', type=str, required=False)
    parser.add_argument('--path-bert', type=str, required=False)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    arch = args.arch
    DEBUG = args.debug
    BATCH_SIZE = 32
    N_WORKERS = 4

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
        from transformers.models.bert.configuration_bert import BertConfig
        cfg = BertConfig()
        from models import BERTForSentimentAnalysis
        model = BERTForSentimentAnalysis(cfg)
    elif arch == 'latefusion':
        from models import LateFusion
        if args.freeze:
            if args.vision_model == 'mobilenet':
                model = LateFusion(args.vision_model, args.cross_attn_fusion, args.freeze, args.path_mobilenet, args.path_bert)
            elif args.vision_model == 'vit':
                model = LateFusion(args.vision_model, args.cross_attn_fusion, args.freeze, args.path_vit, args.path_bert)
        else:
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
    n_epochs = 100

    if DEBUG:
        n_epochs = 1

        # display data
        display_dataset_examples(train_dataloader)

        # display data (no face crop)
        """
        dataloaders = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=False, face_crop=False)
        train_dataloader, _, _ = dataloaders
        display_dataset_examples(train_dataloader)
        """

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

    lib.plot_epoch_metrics(np.arange(n_epochs), [train_losses, val_losses],
                           ['Train', 'Validation'], arch, 'Loss',
                           arch+'_out/loss.png')
    lib.plot_epoch_metrics(np.arange(n_epochs), [train_accs, val_accs],
                           ['Train', 'Validation'], arch, 'Accuracy',
                           arch+'_out/acc.png')

    # evaluate
    print('Loading best model from training for evalution..')
    model.load_state_dict(torch.load(path, weights_only=False))
    model.to(DEVICE)
    print('Evaluating model..')
    test_loss, test_accuracy = lib.evaluate(model, arch, test_dataloader, DEVICE, len(test_dataloader), loss_fn, compute_accuracy)

    # save metrics to CSV
    path = arch+'_out/perf.csv'
    header = ["Architecture", "Epochs", "Test Loss", "Test Accuracy"]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([arch, n_epochs, test_loss, test_accuracy])
    print('Performance results saved:', path)

