import lib
import torch
import argparse
from cmumosi import load_dataset


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--path-mobilenet', type=str, required=False)
    parser.add_argument('--path-bert', type=str, required=False)
    parser.add_argument('--debug', type=bool, default=False, required=False)
    args = parser.parse_args()

    arch = args.arch
    DEBUG = args.debug
    BATCH_SIZE = 32
    N_WORKERS = 4

    # load model
    if arch == 'MobileNetV2':
        from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config
        cfg = MobileNetV2Config()
        from models import MobileNetV2ForFacialExpressionRecognition
        model = MobileNetV2ForFacialExpressionRecognition(cfg)
        n_epochs = 150
        base_dataloaders, _ = load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=True)
        train_dataloader, val_dataloader, _ = base_dataloaders
    elif arch == 'BERT':
        from transformers.models.bert.configuration_bert import BertConfig
        cfg = BertConfig()
        from models import BERTForSentimentAnalysis
        model = BERTForSentimentAnalysis(cfg)
        n_epochs = 12
    elif arch == 'LateFusion':
        if args.path_mobilenet == None:
            raise ValueError('Path to existing mobilenet model needed to train LateFusion. Enter path or \"scratch\" to train from scratch now.')
        elif args.path_mobilenet == 'scratch':
            print('WARN: training from scratch not implemented yet.')
            pass
        if args.path_bert == None:
            raise ValueError('Path to existing bert model needed to train LateFusion. Enter path or \"scratch\" to train from scratch now.')
        elif args.path_bert == 'scratch':
            print('WARN: training from scratch not implemented yet.')
            pass
        model = LateFusionModel(args.path_mobilenet, args.path_bert)
        # TODO how to handle that this uses fine tune dataset not base dataset
        #   load datasets here?
        load_dataset(batch_size=BATCH_SIZE, n_workers=N_WORKERS, finetune=True)
        n_epochs = 100
    else:
        raise ValueError(f'Invalid model architecture {arch}')

    if not torch.cuda.is_available():
        print('GPU not available, running script on CPU..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)

    if DEBUG:
        n_epochs = 3

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
    (train_losses, train_accs), (val_losses, val_accs), path = lib.train_loop(model, arch, train_dataloader, val_dataloader, device=DEVICE, optimizer=optimizer, loss_fn=loss_fn, compute_accuracy=compute_accuracy, num_epochs=n_epochs, save_path=arch+'_out')
    # TODO: evaluate
