# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Generate token graph
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

from os.path import join, exists
from os import listdir, makedirs
import random
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from simpletransformers.classification import ClassificationModel
from simpletransformers.classification.classification_utils import InputExample, convert_examples_to_features

from config import configuration as cfg, platform as plat, username as user
from Logger.logger import logger
from File_Handlers.json_handler import json_keys2df, read_labelled_json
from Layers.BERT_multilabel_classifier import format_inputs

## Setting up the device for GPU usage
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


# Preparing for TPU usage
# import torch_xla
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()


class BERT_pretrained_simpletransformers(ClassificationModel):
    def __init__(self, model_type, model_name, num_labels=None, use_cuda=True,
                 cuda_device=-1, seed=42, n_gpu=2, **kwargs):
        """
        Initializes a MultiLabelClassification model.

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (
            pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            pos_weight (optional): A list of length num_labels containing the weights to assign to each label for
            loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a
            dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options
            specific to the 'from_pretrained' implementation where this will be supplied.
        """
        super().__init__(model_type, model_name, num_labels, use_cuda, cuda_device, **kwargs)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if n_gpu > 0 and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' is True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.args.model_name = model_name
        self.args.model_type = model_type

    def prepare_dataset(self, train_df, output_dir=None, verbose=True, **kwargs):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header,
            it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text,
            and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being logger.infoed to console.
            Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the
            model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training
            is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of
            metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels,
                        and the second parameter will be the predictions.

        Returns:
            None
        """

        if not output_dir:
            output_dir = self.args.output_dir

        if exists(output_dir) and listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        if "text" in train_df.columns and "labels" in train_df.columns:
            train_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(zip(train_df["text"].astype(str), train_df["labels"]))
            ]
        elif "text_a" in train_df.columns and "text_b" in train_df.columns:
            train_examples = [
                InputExample(i, text_a, text_b, label)
                for i, (text_a, text_b, label) in enumerate(
                    zip(train_df["text_a"].astype(str), train_df["text_b"].astype(str), train_df["labels"])
                )
            ]
        else:
            logger.warn(
                "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
            )
            train_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(zip(train_df.iloc[:, 0], train_df.iloc[:, 1]))
            ]
        train_dataset = self.load_and_cache_examples(train_examples, verbose=verbose)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, num_workers=14
        )

        makedirs(output_dir, exist_ok=True)

    def load_and_cache_examples(
            self, examples, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"

        makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        cached_features_file = join(
            args.cache_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode, args.model_type, args.max_seq_length, self.num_labels, len(examples),
            ),
        )

        if exists(cached_features_file) and (
                (not args.reprocess_input_data and not no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not no_cache)
        ):
            features = torch.load(cached_features_file)
            if verbose:
                logger.info(f" Features loaded from cache at {cached_features_file}")
        else:
            if verbose:
                logger.info(" Converting to features started. Cache is not used.")
                if args.sliding_window:
                    logger.info(" Sliding window enabled")

            # If labels_map is defined, then labels need to be replaced with ints
            if self.args.labels_map:
                for example in examples:
                    if multi_label:
                        example.label = [self.args.labels_map[label] for label in example.label]
                    else:
                        example.label = self.args.labels_map[example.label]

            features = convert_examples_to_features(
                examples,
                args.max_seq_length,
                tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences,
                # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args.model_type in ["xlnet"]),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args.silent or silent,
                use_multiprocessing=args.use_multiprocessing,
                sliding_window=args.sliding_window,
                flatten=not evaluate,
                stride=args.stride,
                add_prefix_space=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
                args=args,
            )
            if verbose and args.sliding_window:
                logger.info(f" {len(features)} features created from {len(examples)} samples.")

            if not no_cache:
                torch.save(features, cached_features_file)

        if args.sliding_window and evaluate:
            features = [
                [feature_set] if not isinstance(feature_set, list) else feature_set for feature_set in features
            ]
            window_counts = [len(sample) for sample in features]
            features = [feature for feature_set in features for feature in feature_set]

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    @staticmethod
    def load_and_cache_examples(examples, evaluate=False, no_cache=False, multi_label=True, verbose=True, silent=False):
        return super().load_and_cache_examples(
            examples, evaluate=evaluate, no_cache=no_cache, multi_label=multi_label, verbose=verbose, silent=silent
        )


# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05
cuda_device = -1


class TransformerDataset(Dataset):

    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.text = df.text
        self.targets = df.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=self.max_len,
                                            pad_to_max_length=True, return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids':            torch.tensor(ids, dtype=torch.long),
            'mask':           torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets':        torch.tensor(self.targets[index], dtype=torch.float)
        }


## Creating model, by adding a dropout and a linear (dense) layer on top of transformer to get final model:
class TransformerClassifier(torch.nn.Module):
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad', num_layers=2, dropout=0.3, hid_size=768,
                 out_size=4, pool_output=False):
        super(TransformerClassifier, self).__init__()
        self.model_name = model_name
        self.pool_output = pool_output
        self.transformer = AutoModel.from_pretrained(model_name)

        ## Add multiple layers based on param [num_layers]:
        self.dropout = torch.nn.Dropout(dropout)
        self.pre_classifier = nn.Linear(hid_size, hid_size)

        self.linear_layers = []
        for _ in range(num_layers - 2):
            self.linear_layers.append(nn.Linear(hid_size, hid_size))
        self.linear_layers = nn.ModuleList(self.linear_layers)

        self.classifier = nn.Linear(hid_size, out_size)

        # self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, token_type_ids=None, pool_output=None):
        ## Recheck if pooled output to be taken:
        self.pool_output = pool_output if pool_output is not None else self.pool_output

        if self.model_name.startswith('bert'):
            _, transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                                     token_type_ids=token_type_ids)
        elif self.model_name.startswith('distilbert'):
            transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                                  head_mask=head_mask)  # (bs, seq_len, dim)
        else:
            raise NotImplementedError(f'Unknown model: [{self.model_name}]')

        outputs = transformer_output[0]

        ## Use pooled (sentence) output instead of token embeddings:
        if pool_output:
            outputs = outputs[:, 0]  # (bs, dim)
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.pre_classifier(outputs)

        for layer in self.linear_layers:
            outputs = layer(outputs)
            outputs = F.relu(outputs)
            outputs = F.dropout(outputs, training=self.training)

        outputs = self.classifier(outputs)

        ## Add hidden states and attention if they are here
        outputs = (outputs,) + transformer_output[1:]
        return outputs


class TransformerPretrain():
    def __init__(self, MODEL_NAME="distilbert-base-uncased-distilled-squad", train_df=None, test_df=None, n_gpu=2,
                 use_cuda=True, seed=42):
        self.seed = seed

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if n_gpu > 0 and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if train_df is None:
            train_df = self.read_data()

        if test_df is None:
            train_df, test_df = self.split_data(train_df)
        self.train_df, self.test_df = train_df, test_df

        ## Get the length of second column to get number of classes:
        num_classes = len(train_df[train_df.columns[1]][0])

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_dataset = TransformerDataset(self.train_df, self.tokenizer, MAX_LEN)
        test_dataset = TransformerDataset(self.test_df, self.tokenizer, MAX_LEN)

        train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
        test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

        self.train_loader = DataLoader(train_dataset, **train_params)
        self.test_loader = DataLoader(test_dataset, **test_params)

        self.model = TransformerClassifier(MODEL_NAME, out_size=num_classes)
        logger.info(self.model)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)

    def freeze_weights(self, except_layers=tuple('classifier')):
        for name, param in self.model.named_parameters():
            # for layer in except_layers:
            #     if layer not in name:
            if name not in except_layers:  ## Except specified layers
                param.requires_grad = False
                logger.info(f'Froze layer: {name}')

        # if 'embedding' in except_layers:
        #     for param in list(self.model.bert.embeddings.parameters()):
        #         param.requires_grad = False
        #     logger.info("Froze Embedding Layer")
        #
        # # freeze_layers is a list [1,2,3] representing layer number
        # if freeze_layers is not None:
        #     layer_indexes = [x for x in freeze_layers]
        #     for layer_idx in layer_indexes:
        #         for param in list(self.model.bert.encoder.layer[layer_idx].parameters()):
        #             param.requires_grad = False
        #         logger.info(f"Froze Layer: {layer_idx}")

    def read_data(self, data_dir=cfg["paths"]["dataset_dir"][plat][user], filename=cfg["data"]["source"]['labelled']):
        new_df = read_labelled_json(data_dir, filename)
        new_df = format_inputs(new_df)
        return new_df

    def split_data(self, df, train_size=0.8):
        train_df = df.sample(frac=train_size, random_state=self.seed)
        test_df = df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        logger.info("FULL Dataset: {}".format(df.shape))
        logger.info("TRAIN Dataset: {}".format(train_df.shape))
        logger.info("TEST Dataset: {}".format(test_df.shape))
        return train_df, test_df

    def train(self):
        self.model.to(device)
        self.model.train()
        epoch_loss = 0
        stores = {'preds': [], 'trues': [], 'ids': []}
        for _, data in enumerate(self.train_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            ## Set pool_output = True for classification:
            outputs = self.model(ids, mask, token_type_ids=token_type_ids, pool_output=True)

            self.optimizer.zero_grad()
            loss = self.loss_fn(outputs[0], targets)

            stores['preds'].append(outputs[0])
            stores['trues'].append(targets)

            if _ % 20 == 0:
                logger.info(f'Batch: {_}, Loss:  {loss.item()}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        stores['preds'] = torch.cat(stores['preds'])
        stores['trues'] = torch.cat(stores['trues'])
        # stores['ids'] = torch.cat(stores['ids'])
        return epoch_loss / len(self.train_loader), stores  # , epoch_acc / len(iterator)

    def validation(self):
        self.model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(self.test_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    @staticmethod
    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def prepare_data(self, df):
        dataset_pt = []
        for txt, targets in df:
            tokens = self.tokenizer.tokenize(txt)
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens_ids = self.tokenizer.build_inputs_with_special_tokens(tokens_ids)
            tokens_pt = torch.tensor([tokens_ids])
            dataset_pt.append(tokens_pt)

        return dataset_pt

    def trainer(self):
        for epoch in range(EPOCHS):
            epoch_loss, stores = self.train()
            logger.info(f'Epoch: {epoch}, Loss:  {epoch_loss}')

        # for input in dataset:
        #     outputs, pooled = self.model(input)
        #     logger.info("Token wise output: {}, Pooled output: {}".format(outputs.shape, pooled.shape))

    def validate_model(self):
        for epoch in range(EPOCHS):
            outputs, targets = self.validation()
            outputs = np.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            logger.info(f"Accuracy Score = {accuracy}")
            logger.info(f"F1 Score (Micro) = {f1_score_micro}")
            logger.info(f"F1 Score (Macro) = {f1_score_macro}")


if __name__ == "__main__":
    pretrain_model = TransformerPretrain(use_cuda=False)
    pretrain_model.trainer()

    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("-d", "--dataset_name",
    #                     default=cfg['data']['dataset_name'], type=str)
    # parser.add_argument("-m", "--model_name",
    #                     default=cfg['model']['model_name'], type=str)
    # parser.add_argument("-mt", "--model_type",
    #                     default=cfg['model']['model_type'], type=str)
    # parser.add_argument("-ne", "--num_train_epochs",
    #                     default=cfg['sampling']['num_train_epoch'], type=int)
    # parser.add_argument("-c", "--use_cuda",
    #                     default=cfg['model']['use_cuda'], action='store_true')
    #
    # args = parser.parse_args()
    #
    # main(args.dataset_name, args.model_name, args.model_type,
    #      args.num_train_epochs, args.use_cuda)
