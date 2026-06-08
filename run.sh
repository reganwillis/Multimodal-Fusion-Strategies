#!/bin/bash

DEBUG=--debug
CROSS_ATTN_FUSION=--cross-attn-fusion

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv
source venv/bin/activate
#pip install -r requirements.txt

# clear output dirs
rm -r *_out 

# run training scripts
#CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'mobilenet' $DEBUG > out.log
#mv out.log mobilenet_out
#CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'bert' $DEBUG > out.log
#mv out.log bert_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model 'mobilenet' \
	$CROSS_ATTN_FUSION \
	$DEBUG > out.log
mv out.log latefusionmobilenet_out
#CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py --arch 'vit' --debug $DEBUG > out.log
#mv out.log vit_out
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
	$DEBUG > out.log
mv out.log midfusionmobilenet_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'midfusion' \
	--vision-model 'vit' \
	$CROSS_ATTN_FUSION \
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

# TODO: No vision model
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'latefusion' \
	--vision-model None \
	$DEBUG > out.log
mv out.log latefusionnone_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'midfusion' \
	--vision-model None \
	$DEBUG > out.log
mv out.log midfusionnone_out
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 train.py \
	--arch 'earlyfusion' \
	--vision-model None \
	$DEBUG > out.log
mv out.log earlyfusionnone_out
