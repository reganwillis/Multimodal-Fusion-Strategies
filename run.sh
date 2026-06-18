#!/bin/bash

DEBUG= 
#--debug
CROSS_ATTN_FUSION= 
#--cross-attn-fusion
# use attention fusion layers for mid fusion
ATTN_FUSION=--attn-fusion

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv
source venv2/bin/activate
#pip install -r requirements.txt

# clear output dirs
rm -r *_out

# dataset validation
#CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
#	--arch 'latefusion' --vision-model 'mobilenet' --debug
#exit

# run training scripts
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'mobilenet' $DEBUG > out.log
mv out.log mobilenet_out
#CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'bert' $DEBUG > out.log
#mv out.log bert_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log latefusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'vit' $DEBUG > out.log
mv out.log vit_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log latefusionvit_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'midfusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$ATTN_FUSION \
	$DEBUG > out.log
mv out.log midfusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'midfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$ATTN_FUSION \
	$DEBUG > out.log
mv out.log midfusionvit_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'earlyfusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log earlyfusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'earlyfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log earlyfusionvit_out
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
