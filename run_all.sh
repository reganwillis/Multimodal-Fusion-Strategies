#!/bin/bash

GPU=2
DEBUG= 
#--debug
# use attention fusion layers for mid fusion
ATTN_FUSION=--attn-fusion

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv2
source venv2/bin/activate
#pip install -r requirements.txt

# clear output dirs
rm -r *_out

# No vision model
BERT_PRETRAINED_PATH=bert-base-uncased
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model None \
	--bert-model $BERT_PRETRAINED_PATH \
	$DEBUG > out.log
mv out.log latefusionNone_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'midfusion' \
	--vision-model None \
	--bert-model $BERT_PRETRAINED_PATH \
	$ATTN_FUSION \
	$DEBUG > out.log
mv out.log midfusionNone_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'earlyfusion' \
	--vision-model None \
	--bert-model $BERT_PRETRAINED_PATH \
	$DEBUG > out.log
mv out.log earlyfusionNone_out
exit

# dataset validation
#CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
#	--arch 'latefusion' --vision-model 'mobilenet' --debug
#exit

# hyperparam study
#HYPERPARAM_STUDY=--hyperparam-study
BERT_PRETRAINED_PATH=bert-large-uncased
#CROSS_ATTN_FUSION=--cross-attn-fusion
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model 'vit' \
	--bert-model $BERT_PRETRAINED_PATH \
	$CROSS_ATTN_FUSION \
	$HYPERPARAM_STUDY \
	$DEBUG > out.log
mv out.log latefusionvit_bertlarge_out

HYPERPARAM_STUDY= 
exit


# scripts to run
BERTBASE=true
BERTBASE_XATTN=true
BERTLARGE=true
BERTLARGE_XATTN=true

# run training scripts -- BERT-BASE, NO X ATTN
if $BERTBASE; then
	CROSS_ATTN_FUSION= 
	BERT_PRETRAINED_PATH=bert-base-uncased
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'mobilenet' $DEBUG > out.log
	#mv out.log mobilenet_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'bert' --bert-model $BERT_PRETRAINED_PATH $DEBUG > out.log
	#mv out.log bert_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'latefusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$HYPERPARAM_STUDY \
	#	$DEBUG > out.log
	#mv out.log latefusionmobilenet_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'vit' $DEBUG > out.log
	#mv out.log vit_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'latefusion' \
	#	--vision-model 'vit' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$HYPERPARAM_STUDY \
	#	$DEBUG > out.log
	#mv out.log latefusionvit_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'midfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log midfusionmobilenet_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log midfusionvit_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'earlyfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log earlyfusionmobilenet_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'earlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log earlyfusionvit_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'veryearlyfusion' \
	#	--vision-model 'vit' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log veryearlyfusionvit_out
fi

# run training scripts -- BERT-BASE, X ATTN
if $BERTBASE_XATTN; then
	CROSS_ATTN_FUSION=--cross-attn-fusion
	BERT_PRETRAINED_PATH=bert-base-uncased
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'latefusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$HYPERPARAM_STUDY \
	#	$DEBUG > out.log
	#mv out.log latefusionmobilenet_xattn_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'latefusion' \
	#	--vision-model 'vit' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$HYPERPARAM_STUDY \
	#	$DEBUG > out.log
	#mv out.log latefusionvit_xattn_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'midfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log midfusionmobilenet_xattn_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log midfusionvit_xattn_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'earlyfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log earlyfusionmobilenet_xattn_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log veryearlyfusionvit_xattn_out
fi

# run training scripts -- BERT-LARGE, NO X ATTN
if $BERTLARGE; then
	CROSS_ATTN_FUSION= 
	BERT_PRETRAINED_PATH=bert-large-uncased
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'latefusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$HYPERPARAM_STUDY \
	#	$DEBUG > out.log
	#mv out.log latefusionmobilenet_bertlarge_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'latefusion' \
	#	--vision-model 'vit' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$HYPERPARAM_STUDY \
	#	$DEBUG > out.log
	#mv out.log latefusionvit_bertlarge_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'midfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log midfusionmobilenet_bertlarge_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log midfusionvit_bertlarge_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'earlyfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log earlyfusionmobilenet_bertlarge_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log veryearlyfusionvit_bertlarge_out
fi

# run training scripts -- BERT-LARGE, X ATTN
if $BERTLARGE_XATTN; then
	CROSS_ATTN_FUSION=--cross-attn-fusion
	BERT_PRETRAINED_PATH=bert-large-uncased
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'latefusion' \
		--vision-model 'mobilenet' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$HYPERPARAM_STUDY \
		$DEBUG > out2.log
	mv out2.log latefusionmobilenet_bertlarge_xattn_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'latefusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$HYPERPARAM_STUDY \
		$DEBUG > out2.log
	mv out2.log latefusionvit_bertlarge_xattn_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'midfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log midfusionmobilenet_bertlarge_xattn_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'midfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log midfusionvit_bertlarge_xattn_out
	#CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	#	--arch 'earlyfusion' \
	#	--vision-model 'mobilenet' \
	#	--bert-model $BERT_PRETRAINED_PATH \
	#	$CROSS_ATTN_FUSION \
	#	$DEBUG > out.log
	#mv out.log earlyfusionmobilenet_bertlarge_xattn_out
	CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
		--arch 'veryearlyfusion' \
		--vision-model 'vit' \
		--bert-model $BERT_PRETRAINED_PATH \
		$CROSS_ATTN_FUSION \
		$DEBUG > out2.log
	mv out2.log veryearlyfusionvit_bertlarge_xattn_out
fi

echo "DONE"
exit

# No vision model
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model None \
	$DEBUG > out.log
mv out.log latefusionNone_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'midfusion' \
	--vision-model None \
	$DEBUG > out.log
mv out.log midfusionNone_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'earlyfusion' \
	--vision-model None \
	$DEBUG > out.log
mv out.log earlyfusionNone_out
exit

# freezing weights for late fusion
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model 'mobilenet' \
	--freeze --path-mobilenet trial2/mobilenet_out/model_state_dict.pt \
	--path-bert trial2/bert_out/model_state_dict.pt \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log latefusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model 'vit' \
	--freeze --path-vit trial2/vit_out/model_state_dict.pt \
	--path-bert trial2/bert_out/model_state_dict.pt \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log latefusionvit_out
exit
