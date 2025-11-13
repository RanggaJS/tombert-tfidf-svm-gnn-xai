# run_multimodal_classifier.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# ULTRA OPTIMIZED VERSION - Target 95%+ Accuracy with 3+ Days Training
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""BERT finetuning runner - ULTRA OPTIMIZED VERSION."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import sys
import logging
import argparse
import random
import time
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# Ensure project root is on sys.path so 'my_bert' and other local packages can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_bert.tokenization import BertTokenizer
from my_bert.mm_modeling import (ResBertForMMSequenceClassification, MBertForMMSequenceClassification,
                          MBertNoPoolingForMMSequenceClassification, TomBertForMMSequenceClassification,
                          TomBertNoPoolingForMMSequenceClassification)
from my_bert.optimization import BertAdam
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from torchvision import transforms
from torchvision.models import resnet152 as tv_resnet152
try:
    # Torchvision >= 0.13
    from torchvision.models import ResNet152_Weights
    TORCHVISION_HAS_WEIGHTS = True
except Exception:
    TORCHVISION_HAS_WEIGHTS = False
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support

# Optimization imports
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

__author__ = "Ultra Optimized for 95%+ Accuracy"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ULTRA OPTIMIZATION UTILITIES
# ============================================================================

class AdaptiveLabelSmoothingCrossEntropy(torch.nn.Module):
    """Adaptive label smoothing that reduces as training progresses"""
    def __init__(self, initial_smoothing=0.2, final_smoothing=0.05):
        super(AdaptiveLabelSmoothingCrossEntropy, self).__init__()
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.current_smoothing = initial_smoothing
    
    def update_smoothing(self, progress):
        """Update smoothing based on training progress (0 to 1)"""
        self.current_smoothing = self.initial_smoothing - progress * (self.initial_smoothing - self.final_smoothing)
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.current_smoothing) + (1 - one_hot) * self.current_smoothing / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

class EnhancedEMA:
    """Enhanced Exponential Moving Average with different decay for different layers"""
    def __init__(self, model, decay=0.9999, decay_bert=0.9995, decay_cnn=0.999):
        self.model = model
        self.decay = decay
        self.decay_bert = decay_bert
        self.decay_cnn = decay_cnn
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Different decay rates for different components
                if 'bert' in name.lower():
                    decay = self.decay_bert
                elif 'resnet' in name.lower() or 'cnn' in name.lower():
                    decay = self.decay_cnn
                else:
                    decay = self.decay
                
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

class CombinedLoss(torch.nn.Module):
    """Combination of different losses for better training"""
    def __init__(self, label_smoothing=0.1, focal_alpha=1, focal_gamma=2, 
                 ls_weight=0.7, focal_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.label_smoothing_loss = AdaptiveLabelSmoothingCrossEntropy(label_smoothing, label_smoothing * 0.5)
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.ls_weight = ls_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target, progress=0.0):
        self.label_smoothing_loss.update_smoothing(progress)
        ls_loss = self.label_smoothing_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        return self.ls_weight * ls_loss + self.focal_weight * focal_loss

def create_ultra_optimizer_grouped_parameters(model, encoder, learning_rate, weight_decay=0.01):
    """Create ultra-optimized parameter groups with layer-wise learning rates"""
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    seen_params = set()
    param_groups = []

    def filter_params(named_params, predicate):
        filtered = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if not predicate(name, param):
                continue
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))
            filtered.append(param)
        return filtered

    named_model_params = list(model.named_parameters())
    named_encoder_params = list(encoder.named_parameters())

    # BERT layers with different learning rates (lower for earlier layers)
    for layer_idx in range(12):  # BERT has 12 layers
        layer_lr = learning_rate * (0.8 + 0.2 * layer_idx / 11)

        layer_params_decay = filter_params(
            named_model_params,
            lambda n, _: f'layer.{layer_idx}' in n and not any(nd in n for nd in no_decay)
        )
        layer_params_no_decay = filter_params(
            named_model_params,
            lambda n, _: f'layer.{layer_idx}' in n and any(nd in n for nd in no_decay)
        )

        if layer_params_decay:
            param_groups.append({
                'params': layer_params_decay,
                'weight_decay': weight_decay,
                'lr': layer_lr
            })
        if layer_params_no_decay:
            param_groups.append({
                'params': layer_params_no_decay,
                'weight_decay': 0.0,
                'lr': layer_lr
            })

    # Other BERT parameters
    other_bert_params_decay = filter_params(
        named_model_params,
        lambda n, _: 'bert' in n.lower() and not any(f'layer.{i}' in n for i in range(12)) and not any(nd in n for nd in no_decay)
    )
    other_bert_params_no_decay = filter_params(
        named_model_params,
        lambda n, _: 'bert' in n.lower() and not any(f'layer.{i}' in n for i in range(12)) and any(nd in n for nd in no_decay)
    )

    if other_bert_params_decay:
        param_groups.append({
            'params': other_bert_params_decay,
            'weight_decay': weight_decay,
            'lr': learning_rate
        })
    if other_bert_params_no_decay:
        param_groups.append({
            'params': other_bert_params_no_decay,
            'weight_decay': 0.0,
            'lr': learning_rate
        })

    # Image encoder parameters with much lower learning rate
    encoder_params_decay = filter_params(
        named_encoder_params,
        lambda n, _: not any(nd in n for nd in no_decay)
    )
    encoder_params_no_decay = filter_params(
        named_encoder_params,
        lambda n, _: any(nd in n for nd in no_decay)
    )

    if encoder_params_decay:
        param_groups.append({
            'params': encoder_params_decay,
            'weight_decay': weight_decay * 0.5,
            'lr': learning_rate * 0.05  # Much lower for pre-trained CNN
        })
    if encoder_params_no_decay:
        param_groups.append({
            'params': encoder_params_no_decay,
            'weight_decay': 0.0,
            'lr': learning_rate * 0.05
        })

    # Classifier parameters with higher learning rate
    classifier_params_decay = filter_params(
        named_model_params,
        lambda n, _: 'classifier' in n.lower() and not any(nd in n for nd in no_decay)
    )
    classifier_params_no_decay = filter_params(
        named_model_params,
        lambda n, _: 'classifier' in n.lower() and any(nd in n for nd in no_decay)
    )

    if classifier_params_decay:
        param_groups.append({
            'params': classifier_params_decay,
            'weight_decay': weight_decay,
            'lr': learning_rate * 2.0  # Higher for classifier
        })
    if classifier_params_no_decay:
        param_groups.append({
            'params': classifier_params_no_decay,
            'weight_decay': 0.0,
            'lr': learning_rate * 2.0
        })

    # Catch-all: include any remaining trainable params not already covered
    remaining_params = [
        p for _, p in named_model_params + named_encoder_params
        if p.requires_grad and id(p) not in seen_params
    ]
    if remaining_params:
        param_groups.append({
            'params': remaining_params,
            'weight_decay': weight_decay,
            'lr': learning_rate
        })

    return param_groups

def ultra_cosine_schedule_with_restarts(step, total_steps, warmup_steps, restart_cycles=3):
    """Cosine schedule with warm restarts for better convergence"""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    
    # Calculate cycle length
    cycle_length = (total_steps - warmup_steps) // restart_cycles
    current_cycle = (step - warmup_steps) // cycle_length
    cycle_step = (step - warmup_steps) % cycle_length
    
    if current_cycle >= restart_cycles:
        current_cycle = restart_cycles - 1
        cycle_step = cycle_length - 1
    
    # Cosine annealing within cycle
    progress = float(cycle_step) / float(max(1, cycle_length))
    cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
    
    # Decay amplitude with each restart
    amplitude = 1.0 / (2 ** current_cycle)
    
    return amplitude * cosine_factor

# ============================================================================
# EXISTING CODE (keep as is but with modifications)
# ============================================================================

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class MMInputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b, img_id, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class MMInputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, \
                 s2_segment_ids, img_feat, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.s2_input_ids = s2_input_ids
        self.s2_input_mask = s2_input_mask
        self.s2_segment_ids = s2_segment_ids
        self.img_feat = img_feat
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class AbmsaProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3].lower()
            text_b = line[4].lower()
            img_id = line[2]
            label = line[1]
            examples.append(
                MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label))
        return examples

def convert_mm_examples_to_features(examples, label_list, max_seq_length, max_entity_length, tokenizer, crop_size, path_img):
    """Loads a data file into a list of `InputBatch`s - ULTRA OPTIMIZED."""
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    count = 0

    # ULTRA OPTIMIZED: Advanced image augmentation for better generalization
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ])

    sent_length_a = 0
    entity_length_b = 0
    total_length = 0
    
    total_examples = len(examples)
    
    for (ex_index, example) in enumerate(examples):
        # Progress tracking
        if ex_index % 1000 == 0:
            progress = (ex_index / total_examples) * 100
            print(f"\rProcessing examples: {progress:.1f}% ({ex_index}/{total_examples})", end="", flush=True)
        
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        
        if len(tokens_b) >= entity_length_b:
            entity_length_b = len(tokens_b)
        if len(tokens_a) >= sent_length_a:
            sent_length_a = len(tokens_a)

        if len(tokens_b) > max_entity_length - 2:
            s2_tokens = tokens_b[:(max_entity_length - 2)]
        else:
            s2_tokens = tokens_b
        s2_tokens = ["[CLS]"] + s2_tokens + ["[SEP]"]
        s2_segment_ids = [0] * len(s2_tokens)
        s2_input_ids = tokenizer.convert_tokens_to_ids(s2_tokens)
        s2_input_mask = [1] * len(s2_input_ids)

        s2_padding = [0] * (max_entity_length - len(s2_input_ids))
        s2_input_ids += s2_padding
        s2_input_mask += s2_padding
        s2_segment_ids += s2_padding

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) >= total_length:
            total_length = len(tokens)

        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids)+49)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        added_input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path, transform_train)
        except:
            count += 1
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform_train)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                              segment_ids=segment_ids,
                              s2_input_ids=s2_input_ids, s2_input_mask=s2_input_mask, s2_segment_ids=s2_segment_ids,
                              img_feat = image,
                              label_id=label_id))

    print(f"\rProcessing examples: 100.0% ({total_examples}/{total_examples})")
    print('the number of problematic samples: ' + str(count))
    print('the max length of sentence a: '+str(sent_length_a+2) + ' entity b: '+str(entity_length_b+2) + \
          ' total length: '+str(total_length+3))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


# ============================================================================
# MAIN FUNCTION - ULTRA OPTIMIZED FOR 95%+ ACCURACY
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='../absa_data/twitter',
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model")
    parser.add_argument("--task_name",
                        default='twitter',
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory")

    ## Other parameters - ULTRA OPTIMIZED
    parser.add_argument("--max_seq_length",
                        default=128,  # ULTRA OPTIMIZED: Increased for better context
                        type=int,
                        help="The maximum total input sequence length")
    parser.add_argument("--max_entity_length",
                        default=32,  # ULTRA OPTIMIZED: Increased for better entity representation
                        type=int,
                        help="The maximum entity input sequence length")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,  # ULTRA OPTIMIZED: Smaller batch for better gradient estimates
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,  # ULTRA OPTIMIZED: Consistent with train batch
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,  # ULTRA OPTIMIZED: Lower for stability
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=200.0,  # ULTRA OPTIMIZED: Extended epochs for ultra training
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.25,  # ULTRA OPTIMIZED: Extended warmup
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,  # ULTRA OPTIMIZED: Accumulate gradients
                        help="Number of updates steps to accumulate")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling")
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default='../pytorch-pretrained-BERT/twitter_subimages/', help='path to images')
    parser.add_argument('--mm_model', default='TomBert', help='model name')
    parser.add_argument('--pooling', default='concat', help='pooling method')
    parser.add_argument('--bertlayer', action='store_true', help='whether to add another bert layer')
    parser.add_argument('--tfn', action='store_true', help='whether to use TFN')
    
    # ULTRA OPTIMIZED: New arguments for 95%+ accuracy
    parser.add_argument('--label_smoothing', type=float, default=0.2, help='Initial label smoothing factor')
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use enhanced exponential moving average')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--use_cosine_schedule', action='store_true', default=True, help='Use cosine learning rate schedule with restarts')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_focal_loss', action='store_true', default=True, help='Use focal loss for class imbalance')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--target_accuracy', type=float, default=0.95, help='Target accuracy to achieve')
    parser.add_argument('--max_training_hours', type=int, default=72, help='Maximum training hours (default: 72 hours = 3 days)')

    args = parser.parse_args()

    # ULTRA OPTIMIZED: Enhanced logging and progress tracking
    print("="*100)
    print("🚀 ULTRA OPTIMIZED TOMBERT TRAINING FOR 95%+ ACCURACY")
    print("="*100)
    print(f"🎯 Target Accuracy: {args.target_accuracy*100:.1f}%")
    print(f"⏱️  Maximum Training Time: {args.max_training_hours} hours ({args.max_training_hours/24:.1f} days)")
    print(f"📊 Model: {args.mm_model}")
    print(f"🔧 Pooling: {args.pooling}")
    print(f"📏 Max Seq Length: {args.max_seq_length}")
    print(f"🏷️  Max Entity Length: {args.max_entity_length}")
    print(f"📦 Batch Size: {args.train_batch_size}")
    print(f"📈 Learning Rate: {args.learning_rate}")
    print(f"🔄 Epochs: {args.num_train_epochs}")
    print(f"🌡️  Warmup: {args.warmup_proportion}")
    print(f"✨ Label Smoothing: {args.label_smoothing}")
    print(f"📊 Use EMA: {args.use_ema}")
    print(f"🎯 Use Focal Loss: {args.use_focal_loss}")
    print(f"🔄 Gradient Accumulation: {args.gradient_accumulation_steps}")
    print("="*100)

    if args.task_name == "twitter":
        args.path_image = os.path.join(os.path.dirname(os.path.dirname(__file__)), "IJCAI2019_data", "twitter2017_images")
    elif args.task_name == "twitter2015":
        args.path_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "absa_data", "twitter2015_images")
    else:
        print("The task name is not right!")

    processors = {
        "twitter2015": AbmsaProcessor,
        "twitter": AbmsaProcessor
    }

    num_labels_task = {
        "twitter2015": 3,
        "twitter": 3
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # ULTRA OPTIMIZED: Enhanced GPU setup with memory optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()  # Clear cache
        
        # Print GPU info
        for i in range(n_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        # Auto-rename to a unique timestamped directory instead of failing
        ts_suffix = time.strftime("%Y%m%d_%H%M%S")
        new_output_dir = f"{args.output_dir.rstrip('/')}" + f"_{ts_suffix}"
        logger.info(f"Output directory exists and not empty. Using new directory: {new_output_dir}")
        args.output_dir = new_output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.mm_model == 'ResBert':
        model = ResBertForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                   cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                       args.local_rank),
                                                                   num_labels=num_labels,
                                                                   bert_layer=args.bertlayer,
                                                                   tfn=args.tfn)
    elif args.mm_model == 'MBert':
        model = MBertForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                    args.local_rank),
                                                                num_labels=num_labels,
                                                                pooling=args.pooling)
    elif args.mm_model == 'MBertNoPooling':
        model = MBertNoPoolingForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                    args.local_rank),
                                                                num_labels=num_labels)
    elif args.mm_model == 'TomBertNoPooling':
        model = TomBertNoPoolingForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                    args.local_rank),
                                                                num_labels=num_labels)
    else:  # TomBert by default
        model = TomBertForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                    args.local_rank),
                                                                num_labels=num_labels,
                                                                pooling=args.pooling)
    
    # Initialize image encoder (ResNet-152). Fallback to torchvision pretrained if local weights missing.
    resnet_weights_path = os.path.join(args.resnet_root, 'resnet152.pth')
    if os.path.isfile(resnet_weights_path):
        net = getattr(resnet, 'resnet152')()
        net.load_state_dict(torch.load(resnet_weights_path, weights_only=False))
        logger.info(f"Loaded ResNet-152 weights from {resnet_weights_path}")
    else:
        logger.warning(f"ResNet weights not found at {resnet_weights_path}. Falling back to torchvision pretrained ResNet-152.")
        if TORCHVISION_HAS_WEIGHTS:
            net = tv_resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        else:
            # Older torchvision API
            net = tv_resnet152(pretrained=True)
    encoder = myResnet(net, args.fine_tune_cnn, device)
    
    # Mixed precision: only enable half() if Apex is available; otherwise keep float32
    apex_installed = False
    if args.fp16:
        try:
            import apex  # noqa: F401
            apex_installed = True
        except ImportError:
            apex_installed = False
            logger.warning("Apex is not installed. Disabling fp16 and using float32.")
            args.fp16 = False
    if args.fp16 and apex_installed:
        model.half()
        encoder.half()
    else:
        model.float()
        encoder.float()
    
    model.to(device)
    encoder.to(device)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex")
        model = DDP(model)
        encoder = DDP(encoder)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = torch.nn.DataParallel(encoder)

    # ULTRA OPTIMIZED: Prepare ultra optimizer with layer-wise learning rates
    optimizer_grouped_parameters = create_ultra_optimizer_grouped_parameters(
        model, encoder, args.learning_rate, weight_decay=0.01
    )
    
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            logger.info("Using Apex FP16 optimizer (FusedAdam).")
        except ImportError:
            logger.warning("Apex is not installed. Disabling fp16 and continuing with standard optimizer.")
            args.fp16 = False
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    # ULTRA OPTIMIZED: Initialize enhanced EMA
    ema = None
    if args.use_ema:
        ema = EnhancedEMA(model, decay=args.ema_decay, decay_bert=0.9995, decay_cnn=0.999)
        logger.info(f"Using Enhanced EMA with decay {args.ema_decay}")

    # ULTRA OPTIMIZED: Combined loss with adaptive label smoothing and focal loss
    if args.use_focal_loss:
        criterion = CombinedLoss(
            label_smoothing=args.label_smoothing,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            ls_weight=0.6,
            focal_weight=0.4
        )
        logger.info(f"Using Combined Loss (Label Smoothing + Focal Loss)")
    elif args.label_smoothing > 0:
        criterion = AdaptiveLabelSmoothingCrossEntropy(
            initial_smoothing=args.label_smoothing,
            final_smoothing=args.label_smoothing * 0.5
        )
        logger.info(f"Using Adaptive Label Smoothing with factor {args.label_smoothing}")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")
    
    # Training tracking
    training_start_time = time.time()
    max_training_seconds = args.max_training_hours * 3600
    
    if args.do_train:
        print("\n🔄 Converting training examples to features...")
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, args.max_seq_length, args.max_entity_length, 
            tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
        all_s2_input_mask = torch.tensor([f.s2_input_mask for f in train_features], dtype=torch.long)
        all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                   all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                                   all_img_feats, all_label_ids)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        # Prepare dev set for evaluation
        print("\n🔄 Converting dev examples to features...")
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, args.max_seq_length, args.max_entity_length, 
            tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
        all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
        all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                  all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                                  all_img_feats, all_label_ids)
        
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # ULTRA OPTIMIZED: Training loop with comprehensive improvements
        max_f1 = 0.0
        max_accuracy = 0.0
        patience_counter = 0
        target_reached = False
        
        logger.info("🚀 *************** Running ULTRA OPTIMIZED training ***************")
        logger.info(f"🎯 Target: {args.target_accuracy*100:.1f}% accuracy")
        logger.info(f"⏱️  Max time: {args.max_training_hours} hours")
        logger.info(f"📊 Total epochs: {int(args.num_train_epochs)}")
        
        for train_idx in range(int(args.num_train_epochs)):
            # Check time limit
            elapsed_time = time.time() - training_start_time
            if elapsed_time > max_training_seconds:
                logger.info(f"⏰ Maximum training time ({args.max_training_hours} hours) reached!")
                break
            
            remaining_time = max_training_seconds - elapsed_time
            epoch_progress = (train_idx + 1) / args.num_train_epochs * 100
            
            logger.info("="*80)
            logger.info(f"🔄 Epoch: {train_idx + 1}/{int(args.num_train_epochs)} ({epoch_progress:.1f}%)")
            logger.info(f"⏰ Elapsed: {elapsed_time/3600:.2f}h | Remaining: {remaining_time/3600:.2f}h")
            logger.info(f"🎯 Current best accuracy: {max_accuracy*100:.3f}%")
            logger.info("="*80)
            
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            
            model.train()
            encoder.train()
            encoder.zero_grad()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            epoch_start_time = time.time()
            total_batches = len(train_dataloader)
            
            for step, batch in enumerate(train_dataloader):
                # Batch progress
                batch_progress = (step + 1) / total_batches * 100
                overall_progress = ((train_idx * total_batches + step + 1) / (args.num_train_epochs * total_batches)) * 100
                
                if step % 50 == 0:  # Update every 50 steps
                    elapsed_epoch = time.time() - epoch_start_time
                    estimated_epoch_time = elapsed_epoch / (step + 1) * total_batches
                    remaining_epoch_time = estimated_epoch_time - elapsed_epoch
                    
                    print(f"\r🔄 Epoch {train_idx+1}: {batch_progress:.1f}% | Overall: {overall_progress:.2f}% | "
                          f"ETA: {remaining_epoch_time/60:.1f}min", end="", flush=True)
                
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
                img_feats, label_ids = batch
                
                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                
                # Forward pass
                logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, 
                             input_mask, s2_input_mask, added_input_mask)
                
                # ULTRA OPTIMIZED: Use combined loss with progress tracking
                training_progress = global_step / t_total
                if hasattr(criterion, 'forward') and 'progress' in criterion.forward.__code__.co_varnames:
                    loss = criterion(logits, label_ids, progress=training_progress)
                else:
                    loss = criterion(logits, label_ids)
                
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # ULTRA OPTIMIZED: Use cosine schedule with restarts
                    if args.use_cosine_schedule:
                        lr_this_step = args.learning_rate * ultra_cosine_schedule_with_restarts(
                            global_step, t_total, int(t_total * args.warmup_proportion), restart_cycles=3
                        )
                    else:
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step/t_total, args.warmup_proportion
                        )
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # ULTRA OPTIMIZED: Update enhanced EMA
                    if ema is not None:
                        ema.update()

            print()  # New line after progress updates

            # ULTRA OPTIMIZED: Comprehensive evaluation after each epoch
            logger.info("📊 ***** Running evaluation on Dev Set *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            
            # Apply EMA if used
            if ema is not None:
                ema.apply_shadow()
            
            model.eval()
            encoder.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            true_label_list = []
            pred_label_list = []

            eval_start_time = time.time()
            total_eval_batches = len(eval_dataloader)

            for eval_step, (input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
                img_feats, label_ids) in enumerate(eval_dataloader):
                
                # Evaluation progress
                eval_progress = (eval_step + 1) / total_eval_batches * 100
                if eval_step % 20 == 0:
                    print(f"\r📊 Evaluation: {eval_progress:.1f}%", end="", flush=True)
                
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                s2_input_ids = s2_input_ids.to(device)
                s2_input_mask = s2_input_mask.to(device)
                s2_segment_ids = s2_segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                    logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, 
                                 input_mask, s2_input_mask, added_input_mask)
                    
                    # Use simple cross-entropy for evaluation
                    tmp_eval_loss = torch.nn.functional.cross_entropy(logits, label_ids)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            print()  # New line after evaluation progress

            # Restore original weights if EMA was used
            if ema is not None:
                ema.restore()

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = macro_f1(true_label, pred_outputs)
            
            # Calculate elapsed time
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - training_start_time
            
            result = {
                'epoch': train_idx + 1,
                'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'precision': precision,
                'recall': recall,
                'f_score': F_score,
                'global_step': global_step,
                'loss': loss,
                'epoch_time_minutes': epoch_time / 60,
                'total_time_hours': total_elapsed / 3600
            }

            # ULTRA OPTIMIZED: Enhanced result logging with progress
            print("\n" + "="*80)
            print(f"📊 ***** EPOCH {train_idx + 1} RESULTS *****")
            print("="*80)
            print(f"🎯 Accuracy:     {eval_accuracy*100:.3f}% (Target: {args.target_accuracy*100:.1f}%)")
            print(f"📈 F1-Score:     {F_score:.4f}")
            print(f"🔍 Precision:    {precision:.4f}")
            print(f"🎯 Recall:       {recall:.4f}")
            print(f"📉 Loss:         {eval_loss:.4f}")
            print(f"⏱️  Epoch Time:   {epoch_time/60:.2f} minutes")
            print(f"⏰ Total Time:   {total_elapsed/3600:.2f} hours")
            print(f"📊 Progress:     {epoch_progress:.1f}% of epochs")
            
            # Target achievement check
            if eval_accuracy >= args.target_accuracy:
                print(f"🎉 🎉 🎉 TARGET ACHIEVED! 🎉 🎉 🎉")
                print(f"✅ Accuracy {eval_accuracy*100:.3f}% >= Target {args.target_accuracy*100:.1f}%")
                target_reached = True
            
            print("="*80 + "\n")

            logger.info("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            # ULTRA OPTIMIZED: Save best model and implement sophisticated early stopping
            improvement = False
            if eval_accuracy > max_accuracy:
                max_accuracy = eval_accuracy
                improvement = True
                
            if F_score > max_f1:
                max_f1 = F_score
                improvement = True
            
            if improvement:
                patience_counter = 0
                
                # Save best model
                model_to_save = model.module if hasattr(model, 'module') else model
                encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
                
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                
                # Save detailed results
                result_file = os.path.join(args.output_dir, f'best_results_epoch_{train_idx+1}.json')
                with open(result_file, 'w') as f:
                    import json
                    json.dump(result, f, indent=2)
                
                logger.info(f"🌟 *** NEW BEST! Accuracy: {max_accuracy*100:.3f}%, F1: {max_f1:.4f} - Model saved! ***")
                
                # Check if target is reached
                if target_reached:
                    logger.info(f"🎯 *** TARGET ACCURACY {args.target_accuracy*100:.1f}% REACHED! ***")
                    # Continue training for a few more epochs to ensure stability
                    if patience_counter >= 2:  # Allow 2 more epochs after target
                        logger.info("🏁 Target achieved and stabilized. Stopping training.")
                        break
            else:
                patience_counter += 1
                logger.info(f"⏳ No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
                
                # If target reached, be more patient
                patience_limit = args.early_stopping_patience * 2 if target_reached else args.early_stopping_patience
                
                if patience_counter >= patience_limit:
                    logger.info("🛑 Early stopping triggered!")
                    break

        # Training completed
        total_training_time = time.time() - training_start_time
        logger.info("="*100)
        logger.info("🏁 TRAINING COMPLETED!")
        logger.info("="*100)
        logger.info(f"⏰ Total training time: {total_training_time/3600:.2f} hours ({total_training_time/86400:.2f} days)")
        logger.info(f"🎯 Best accuracy achieved: {max_accuracy*100:.3f}%")
        logger.info(f"📈 Best F1-score achieved: {max_f1:.4f}")
        logger.info(f"✅ Target {args.target_accuracy*100:.1f}% {'ACHIEVED' if target_reached else 'NOT ACHIEVED'}")
        logger.info("="*100)

    # Load best model for final evaluation
    logger.info("📥 Loading best model for final evaluation...")
    model_state_dict = torch.load(output_model_file)
    if args.mm_model == 'ResBert':
        model = ResBertForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                   state_dict=model_state_dict,
                                                                   num_labels=num_labels,
                                                                   bert_layer=args.bertlayer,
                                                                   tfn=args.tfn)
    elif args.mm_model == 'MBert':
        model = MBertForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                state_dict=model_state_dict,
                                                                num_labels=num_labels,
                                                                pooling=args.pooling)
    elif args.mm_model == 'MBertNoPooling':
        model = MBertNoPoolingForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                state_dict=model_state_dict,
                                                                num_labels=num_labels)
    elif args.mm_model == 'TomBertNoPooling':
        model = TomBertNoPoolingForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                state_dict=model_state_dict,
                                                                num_labels=num_labels)
    else:
        model = TomBertForMMSequenceClassification.from_pretrained(args.bert_model,
                                                                state_dict=model_state_dict,
                                                                num_labels=num_labels,
                                                                pooling=args.pooling)
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        print("\n🔄 Converting test examples to features...")
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, args.max_seq_length, args.max_entity_length, 
            tokenizer, args.crop_size, args.path_image)
        
        logger.info("🧪 ***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
        all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
        all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                  all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
                                  all_img_feats, all_label_ids)
        
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        encoder.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        true_label_list = []
        pred_label_list = []
        
        total_test_batches = len(eval_dataloader)
 
        for test_step, (input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
            img_feats, label_ids) in enumerate(eval_dataloader):
            
            # Test progress
            test_progress = (test_step + 1) / total_test_batches * 100
            if test_step % 20 == 0:
                print(f"\r🧪 Testing: {test_progress:.1f}%", end="", flush=True)
            
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            s2_input_ids = s2_input_ids.to(device)
            s2_input_mask = s2_input_mask.to(device)
            s2_segment_ids = s2_segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, 
                             input_mask, s2_input_mask, added_input_mask)
                tmp_eval_loss = torch.nn.functional.cross_entropy(logits, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        print()  # New line after test progress

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)
        precision, recall, F_score = macro_f1(true_label, pred_outputs)
        
        result = {
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'precision': precision,
            'recall': recall,
            'f_score': F_score,
            'global_step': global_step,
            'loss': loss
        }

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
        fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + '\n')
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + '\n')

        fout_p.close()
        fout_t.close()

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("🏆 ***** FINAL TEST RESULTS *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        
        # ULTRA OPTIMIZED: Final comprehensive results summary
        print("\n" + "="*100)
        print("🏆 FINAL TEST RESULTS - ULTRA OPTIMIZED TOMBERT")
        print("="*100)
        print(f"🎯 Test Accuracy:  {eval_accuracy*100:.3f}% {'✅ TARGET ACHIEVED!' if eval_accuracy >= args.target_accuracy else '❌ Below target'}")
        print(f"📈 Test F1-Score:  {F_score:.4f}")
        print(f"🔍 Test Precision: {precision:.4f}")
        print(f"🎯 Test Recall:    {recall:.4f}")
        print(f"📉 Test Loss:      {eval_loss:.4f}")
        print("="*100)
        print(f"🎯 Target Accuracy: {args.target_accuracy*100:.1f}%")
        print(f"✅ Best Dev Accuracy: {max_accuracy*100:.3f}%")
        print(f"📈 Best F1-Score: {max_f1:.4f}")
        if hasattr(locals(), 'total_training_time'):
            print(f"⏰ Total Training Time: {total_training_time/3600:.2f} hours ({total_training_time/86400:.2f} days)")
        print("="*100 + "\n")

if __name__ == "__main__":
    main()