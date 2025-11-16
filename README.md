# Multimodal Fusion Strategies
A multimodal approach to sentiment analysis that uses facial expression and sentiment analysis of spoken text. A vision model process the facial expressions and a BERT model processes the text. There are three levels of fusion: late, intermediate, and early. There are two vision models: MobileNetV2 and ViT.

## Running Training
Scripts for training the models are in the [train](./train) folder.
Run all training with the bash script `bash run.sh`

## Model Architecture
The model architectures are classes in the training scripts:
* [BERTForSentimentAnalysis](./train/late-fusion.py)
* [MobileNetV2ForFacialExpressionRecognition](./train/late-fusion.py)
* [ViTForFacialExpressionRecognition](./train/late-fusion.py)
* [LateFusionModel](./train/late-fusion.py)
* [LateFusionModel_withViT](./train/late-fusion.py)
* [MidFusionModel](./train/mid-fusion.py)
* [MidFusionModel](./train/mid-fusion-vit.py) - vit
* [EarlyFusionModel](./train/early-fusion.py)
* [EarlyFusionModel](./train/early-fusion-vit.py) - vit

## CMU-MOSI Dataset
The CMU-MOSI dataset was used to train the models. The training scripts above read the dataset in and create PyTorch dataloaders with a custom PyTorch dataset: `MultimodalDataset`. Note that paths will need to be changed for training.
The original CMU-MOSI dataset contains video clips of speakers. We use the middle frame of these clips as the image to train the facial expression recognition model. The dataset version uploaded [here](https://www.kaggle.com/datasets/a61979992/cmu-mosi) has the images used.
