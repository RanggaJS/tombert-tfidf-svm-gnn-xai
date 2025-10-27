# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# OPTIMIZED VERSION - Enhanced for better performance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""BERT finetuning runner - OPTIMIZED VERSION."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from my_bert.tokenization import BertTokenizer
from my_bert.mm_modeling import (ResBertForMMSequenceClassification, MBertForMMSequenceClassification,
                          MBertNoPoolingForMMSequenceClassification, TomBertForMMSequenceClassification,
                          TomBertNoPoolingForMMSequenceClassification)
from my_bert.optimization import BertAdam
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from torchvision import transforms
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support

# Optimization imports
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

__author__ = "Jianfei - Optimized"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# OPTIMIZATION UTILITIES
# ============================================================================

class LabelSmoothingCrossEntropy(torch.nn.Module):
    """Label smoothing for better generalization"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
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
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def create_optimizer_grouped_parameters(model, encoder, learning_rate, weight_decay=0.01):
    """Create optimized parameter groups with different learning rates"""
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    # BERT parameters
    bert_params = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': learning_rate
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': learning_rate
        }
    ]
    
    # Image encoder parameters with lower learning rate
    encoder_params = [
        {
            'params': [p for n, p in encoder.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': learning_rate * 0.1  # 10x lower for pre-trained CNN
        },
        {
            'params': [p for n, p in encoder.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
            'lr': learning_rate * 0.1
        }
    ]
    
    return bert_params + encoder_params

def warmup_cosine_schedule(step, total_steps, warmup_steps):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

# ============================================================================
# EXISTING CODE (keep as is)
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
    """Loads a data file into a list of `InputBatch`s - OPTIMIZED."""
    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    count = 0

    # OPTIMIZED: Better image augmentation
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize first
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Add color jitter
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    sent_length_a = 0
    entity_length_b = 0
    total_length = 0
    
    for (ex_index, example) in enumerate(examples):
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
            image = image_process(image_path, transform)
        except:
            count += 1
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)

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
# MAIN FUNCTION - OPTIMIZED
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

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=80,  # OPTIMIZED: Increased from 64
                        type=int,
                        help="The maximum total input sequence length")
    parser.add_argument("--max_entity_length",
                        default=20,  # OPTIMIZED: Increased from 16
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
                        default=32,  # OPTIMIZED: Increased from 16
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,  # OPTIMIZED: Increased from 16
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,  # OPTIMIZED: Changed from 5e-5
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=12.0,  # OPTIMIZED: Increased from 8.0
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.15,  # OPTIMIZED: Increased from 0.1
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
                        default=1,
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
    parser.add_argument('--pooling', default='concat', help='pooling method')  # OPTIMIZED: Changed default to 'concat'
    parser.add_argument('--bertlayer', action='store_true', help='whether to add another bert layer')
    parser.add_argument('--tfn', action='store_true', help='whether to use TFN')
    
    # OPTIMIZED: New arguments
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--use_ema', action='store_true', help='Use exponential moving average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--use_cosine_schedule', action='store_true', help='Use cosine learning rate schedule')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Early stopping patience')

    args = parser.parse_args()

    # OPTIMIZED: Enhanced logging
    print("="*80)
    print("OPTIMIZED TOMBERT TRAINING")
    print("="*80)
    print(f"Model: {args.mm_model}")
    print(f"Pooling: {args.pooling}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Max Entity Length: {args.max_entity_length}")
    print(f"Batch Size: {args.train_batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Warmup: {args.warmup_proportion}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Use EMA: {args.use_ema}")
    print("="*80)

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

    # OPTIMIZED: Enhanced GPU setup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
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
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
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
    
    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth'), weights_only=False))
    encoder = myResnet(net, args.fine_tune_cnn, device)
    
    if args.fp16:
        model.half()
        encoder.half()
    
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

    # OPTIMIZED: Prepare optimizer with grouped parameters
    optimizer_grouped_parameters = create_optimizer_grouped_parameters(
        model, encoder, args.learning_rate, weight_decay=0.01
    )
    
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    # OPTIMIZED: Initialize EMA if requested
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        logger.info(f"Using EMA with decay {args.ema_decay}")

    # OPTIMIZED: Label smoothing loss
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        logger.info(f"Using label smoothing with factor {args.label_smoothing}")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")
    
    if args.do_train:
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
        
        # Prepare dev set for early stopping
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

        # OPTIMIZED: Training loop with improvements
        max_f1 = 0.0
        patience_counter = 0
        
        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: "+ str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            
            model.train()
            encoder.train()
            encoder.zero_grad()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
                img_feats, label_ids = batch
                
                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                
                # Forward pass
                logits = model(input_ids, s2_input_ids, img_att, segment_ids, s2_segment_ids, 
                             input_mask, s2_input_mask, added_input_mask)
                
                # OPTIMIZED: Use label smoothing loss
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
                    # OPTIMIZED: Use cosine schedule if requested
                    if args.use_cosine_schedule:
                        lr_this_step = args.learning_rate * warmup_cosine_schedule(
                            global_step, t_total, int(t_total * args.warmup_proportion)
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

                    # OPTIMIZED: Update EMA
                    if ema is not None:
                        ema.update()

            # OPTIMIZED: Evaluation after each epoch
            logger.info("***** Running evaluation on Dev Set*****")
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

            for input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
                img_feats, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
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
                    tmp_eval_loss = criterion(logits, label_ids)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                true_label_list.append(label_ids)
                pred_label_list.append(logits)
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            # Restore original weights if EMA was used
            if ema is not None:
                ema.restore()

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            loss = tr_loss / nb_tr_steps if args.do_train else None
            true_label = np.concatenate(true_label_list)
            pred_outputs = np.concatenate(pred_label_list)
            precision, recall, F_score = macro_f1(true_label, pred_outputs)
            
            result = {
                'epoch': train_idx,
                'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                'precision': precision,
                'recall': recall,
                      'f_score': F_score,
                      'global_step': global_step,
                'loss': loss
            }

            logger.info("***** Dev Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            # OPTIMIZED: Save best model based on F1 score and early stopping
            if F_score > max_f1:
                max_f1 = F_score
                patience_counter = 0
                
                # Save best model
                model_to_save = model.module if hasattr(model, 'module') else model
                encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder
                
                    torch.save(model_to_save.state_dict(), output_model_file)
                    torch.save(encoder_to_save.state_dict(), output_encoder_file)
                
                logger.info(f"*** New best F1: {max_f1:.4f} - Model saved! ***")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")
                
                if patience_counter >= args.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break

    # Load best model for final evaluation
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
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, args.max_seq_length, args.max_entity_length, 
            tokenizer, args.crop_size, args.path_image)
        
        logger.info("***** Running evaluation on Test Set*****")
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
 
        for input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, s2_segment_ids, \
            img_feats, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
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
                tmp_eval_loss = criterion(logits, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

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
            logger.info("***** Test Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        
        # OPTIMIZED: Print final summary
        print("\n" + "="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        print(f"Accuracy:  {eval_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {F_score:.4f}")
        print("="*80)

if __name__ == "__main__":
    main()
