import onnx
import torch
from transformers import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.mobilenet_v2.configuration_mobilenet_v2 import MobileNetV2Config
from onnx.defs import onnx_opset_version
import onnxscript

import importlib
latefusion = importlib.import_module("train.late-fusion")
midfusion = importlib.import_module("train.mid-fusion")
earlyfusion = importlib.import_module("train.early-fusion")

def export_to_onnx(path_to_weights, path, arch, device, path2=None, path3=None):
    model_id = path.split('.')[0]
    path = path_to_weights + '/' + path
    print('deploying to onnx:', model_id)
    if arch == 'mobilenetv2':
        cfg = MobileNetV2Config()
        model = latefusion.MobileNetV2ForFacialExpressionRecognition(cfg)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = None
        opset_version = 13
        dynamo=False
    elif arch == 'bert':
        # https://huggingface.co/blog/convert-transformers-to-onnx
        cfg = BertConfig()
        model = latefusion.BERTForSentimentAnalysis(cfg)
        tokenizer = TOKENIZER
        text = str("that is crazy!")
        text = " ".join(text.split())  # clean whitespace
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_token_type_ids=True,
            return_tensors = 'pt'
        ).to(device)
        dummy_input = tuple(inputs.values())

        input_names = ['input_ids', 'attention_mask', 'token_type_ids']
        output_names=['logits']
        dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                        'logits': {0: 'batch_size', 1: 'sequence'}}
        opset_version=14
        dynamo=False
    elif arch == 'late_fusion':
        model = latefusion.LateFusionModel(path2, path3)

        mobilenet_input = torch.randn(1, 3, 224, 224).to(device)

        tokenizer = TOKENIZER
        text = str("that is crazy!")
        text = " ".join(text.split())  # clean whitespace
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_token_type_ids=True,
            return_tensors = 'pt'
        ).to(device)
        bert_input = tuple(inputs.values())
        dummy_input = (mobilenet_input, bert_input[0], bert_input[1], bert_input[2])
        input_names = ['images', 'input_ids', 'attention_mask', 'token_type_ids']
        output_names=['logits']
        dynamic_axes = {'images': {0: 'batch_size', 1: 'sequence'},
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                        'logits': {0: 'batch_size', 1: 'sequence'}}
        opset_version=14
        dynamo=False
    elif arch == 'early_fusion':
        model = earlyfusion.EarlyFusionModel()

        mobilenet_input = torch.randn(1, 3, 224, 224).to(device)

        tokenizer = TOKENIZER
        text = str("that is crazy!")
        text = " ".join(text.split())  # clean whitespace
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_token_type_ids=True,
            return_tensors = 'pt'
        ).to(device)
        bert_input = tuple(inputs.values())
        dummy_input = (mobilenet_input, bert_input[0], bert_input[1], bert_input[2])
        input_names = ['images', 'input_ids', 'attention_mask', 'token_type_ids']
        output_names=['logits']
        dynamic_axes = {'images': {0: 'batch_size', 1: 'sequence'},
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                        'logits': {0: 'batch_size', 1: 'sequence'}}
        dynamic_axes = None
        opset_version=20
        dynamo=True
    elif arch == 'mid_fusion':
        model = midfusion.MidFusionModel()

        mobilenet_input = torch.randn(1, 3, 224, 224).to(device)

        tokenizer = TOKENIZER
        text = str("that is crazy!")
        text = " ".join(text.split())  # clean whitespace
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_token_type_ids=True,
            return_tensors = 'pt'
        ).to(device)
        bert_input = tuple(inputs.values())
        dummy_input = (mobilenet_input, bert_input[0], bert_input[1], bert_input[2])
        input_names = ['images', 'input_ids', 'attention_mask', 'token_type_ids']
        output_names=['logits']
        dynamic_axes = {'images': {0: 'batch_size', 1: 'sequence'},
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                        'logits': {0: 'batch_size', 1: 'sequence'}}
        dynamic_axes = None
        opset_version=20
        dynamo=True
    else:
        print('ERROR: model architecture not supported:', arch)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

    # Export the model to ONNX
    torch.onnx.export(
        model,  # The loaded PyTorch model
        dummy_input,  # Example input tensor
        f"./{path_to_weights}/{model_id}.onnx",  # Output ONNX file name
        dynamo=dynamo,
        export_params=True,  # Store trained parameters
        opset_version=opset_version,  # ONNX version (adjust as needed)
        do_constant_folding=False,  # Optimize by folding constants
        input_names=input_names,  # Naming input tensor
        output_names=output_names,  # Naming output tensor
        dynamic_axes=dynamic_axes
    )

    print("...Model successfully exported to ONNX!\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('GPU not available, running script on CPU..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    export_to_onnx('late-fusion-weights', 'mobilenet_state_dict.pt', 'mobilenetv2', DEVICE)
    export_to_onnx('late-fusion-weights', 'bert_state_dict.pt', 'bert', DEVICE)
    export_to_onnx('late-fusion-weights', 'late_fusion_state_dict.pt', 'late_fusion', DEVICE, 'late-fusion-weights/mobilenet_state_dict.pt', 'late-fusion-weights/bert_state_dict.pt')

    export_to_onnx('models-mid-fusion_8-6-4-4-fuse','late_fusion_state_dict.pt', 'mid_fusion', DEVICE)
    export_to_onnx('models-early-fusion_4-attn-blocks','late_fusion_state_dict.pt', 'early_fusion', DEVICE)

