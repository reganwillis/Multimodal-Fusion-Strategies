#!/bin/bash

export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error

GPU=0
DEBUG= 
#--debug
# use attention fusion layers for mid fusion
ATTN_FUSION=--attn-fusion

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv2
source venv2/bin/activate
#pip install -r requirements.txt

# scripts to run
BERTBASE=false
BERTBASE_XATTN=true
BERTLARGE=true
BERTLARGE_XATTN=true

# BERT BASE - NO X ATTN
if $BERTBASE; then
	echo "Evaluating BERT Base"
	WEIGHTS_DIR='./trials/trial2'
	CROSS_ATTN_FUSION= 
	BERT_PRETRAINED_PATH=bert-base-uncased
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'mobilenet' $DEBUG --weights $WEIGHTS_DIR/mobilenet_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'bert' --bert-model $BERT_PRETRAINED_PATH $DEBUG --weights $WEIGHTS_DIR/bert_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionmobilenet_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'vit' $DEBUG --weights $WEIGHTS_DIR/vit_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionvit_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionmobilenet_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionvit_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'earlyfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/earlyfusionmobilenet_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'earlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/earlyfusionvit_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/veryearlyfusionvit_out/model_state_dict.pt
fi

# BERT BASE - X ATTN
if $BERTBASE_XATTN; then
	echo "Evaluating BERT Base X-attn"
	WEIGHTS_DIR='./trials/trial9_crossattn'
	CROSS_ATTN_FUSION=--cross-attn-fusion
	BERT_PRETRAINED_PATH=bert-base-uncased
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionmobilenet_xattn_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionvit_xattn_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionmobilenet_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionvit_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./trials/trial9_crossattn'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'earlyfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/earlyfusionmobilenet_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/veryearlyfusionvit_xattn_out/model_state_dict.pt
fi

# BERT LARGE - NO X ATTN
if $BERTLARGE; then
	echo "Evaluating BERT Large"
	WEIGHTS_DIR='./new_archs'
	CROSS_ATTN_FUSION= 
	BERT_PRETRAINED_PATH=bert-large-uncased
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionmobilenet_bertlarge_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionvit_bertlarge_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionmobilenet_bertlarge_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionvit_bertlarge_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'earlyfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/earlyfusionmobilenet_bertlarge_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/veryearlyfusionvit_bertlarge_out/model_state_dict.pt
fi

# BERT LARGE - X ATTN
if $BERTLARGE_XATTN; then
	echo "Evaluating BERT Large - X ATTN"
	WEIGHTS_DIR='./new_archs2'
	#./new_archs2/latefusionmobilenet_bertlarge_xattn_out/model_state_dict.pt
	CROSS_ATTN_FUSION=--cross-attn-fusion
	BERT_PRETRAINED_PATH=bert-large-uncased
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionmobilenet_bertlarge_xattn_out/model_state_dict.pt
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'latefusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/latefusionvit_bertlarge_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./trials/trial10_bertlargecrossattn'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionmobilenet_bertlarge_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/midfusionvit_bertlarge_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./trials/trial10_bertlargecrossattn'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'earlyfusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/earlyfusionmobilenet_bertlarge_xattn_out/model_state_dict.pt
	WEIGHTS_DIR='./new_archs2'
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG \
		--weights $WEIGHTS_DIR/veryearlyfusionvit_bertlarge_xattn_out/model_state_dict.pt
fi
exit

# run evaluations scripts
#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'mobilenet' $DEBUG --weights $WEIGHTS_DIR/mobilenet_out/model_state_dict.pt
#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'bert' $DEBUG --weights $WEIGHTS_DIR/bert_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'latefusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/latefusionmobilenet_out/model_state_dict.pt
exit
#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py --arch 'vit' $DEBUG --weights $WEIGHTS_DIR/vit_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'latefusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/latefusionvit_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'midfusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/midfusionmobilenet_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'midfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/midfusionvit_out/model_state_dict.pt
#WEIGHTS_DIR='./latency_test_models'
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'earlyfusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/earlyfusionmobilenet_out/model_state_dict.pt
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 evaluate.py \
	--arch 'earlyfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$DEBUG \
	--weights $WEIGHTS_DIR/earlyfusionvit_out/model_state_dict.pt
exit
