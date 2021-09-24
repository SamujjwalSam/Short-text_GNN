# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : GNN Wrapper to combine token token graph with instance graphs.
__description__ : GNN Wrapper to combine token token graph with instance graphs written in DGL library
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "20/10/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

from torch import nn, Tensor, mean, cat, stack, save

from Layers.gcn_classifiers import GCN
from Layers.bilstm_classifiers import BiLSTM_Classifier
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

from simpletransformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset,
    ClassificationDataset,
    convert_examples_to_features,
    # load_hf_dataset,
)


class BertForGLEN(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, 
            sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the 
            self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"

    def __init__(self, config):
        super(BertForGLEN, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask)
        # Complains if input_embeds is kept

        # pooled_output = outputs.pooler_output

        output = self.dropout(outputs.last_hidden_state)

        return output


class GLEN_BERT_Classifier(BertPreTrainedModel):
    def __init__(self, config, in_dim, hid_dim, num_heads, out_dim, num_classes,
                 combine='concat', state=None):
        super(GLEN_BERT_Classifier, self).__init__(config)
        self.combine = combine
        self.token_gcn = GCN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)

        ## Load model state and parameters for the GCN part only:
        if state is not None:
            self.token_gcn.load_state_dict(state['state_dict'])

        ## Define BERT here
        self.bert = BertForGLEN(config)

        if combine == 'concat':
            final_dim = 2 * out_dim
        elif combine == 'avg':
            final_dim = out_dim
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{combine}] provided.')
        self.bilstm_classifier = BiLSTM_Classifier(final_dim, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids,
                head_mask, instance_batch_global_token_ids: list, A: Tensor,
                X: Tensor, save_gcn_embs=False) -> Tensor:
        """ Combines embeddings of tokens from token graph and bert by concatenating.

        Take embeddings from token graph for tokens present in the bert batch.

        :param head_mask:
        :param position_ids:
        :param token_type_ids:
        :param attention_mask:
        :param input_ids:
        :param save_gcn_embs:
        :param instance_batch_local_token_ids: local node ids for a instance graph to repeat the node embeddings.
        :param node_counts: As batching graphs lose information about the number of nodes in each instance graph,
        node_counts is a list of nodes (unique) in each instance graph.
        :param combine: How to combine two embeddings (Default: concatenate)
        :param instance_batch_global_token_ids: Ordered set of token ids present in the current instance batch
        :param X: Embeddings from token GCN
        :param A: token graph
        """
        ## Fetch embeddings from BERT:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        # Complains if input_embeds is kept
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        ## Fetch embeddings from token graph
        X = self.token_gcn(A=A, X=X)

        if save_gcn_embs:
            save(X, 'X_gcn.pt')

        ## Fetch embeddings from token graph of tokens present in instance batch only:
        ## Fetch consecutive instance embs and global idx from token_embs:
        preds = []
        for token_global_ids in zip(instance_batch_global_token_ids):
            ## Combine both embeddings:
            if self.combine == 'concat':
                combined_emb = cat([X[token_global_ids], pooled_output], dim=1)
            elif self.combine == 'avg':
                combined_emb = mean(stack([X[token_global_ids], pooled_output]), dim=0)
            else:
                raise NotImplementedError(f'combine supports either concat or avg.'
                                          f' [{self.combine}] provided.')

            pred = self.bilstm_classifier(combined_emb.unsqueeze(0))
            preds.append(pred)

        return stack(preds).squeeze()


def glen_bert_model(bert, gcn, bilstm_classifier, A, X, inputs, global_token_ids,
                    combine='concat'):
    bert_outputs = bert(**inputs)
    X = gcn(A, X)
    preds = []
    for token_global_ids in zip(global_token_ids):
        ## Combine both embeddings:
        if combine == 'concat':
            combined_emb = cat([X[token_global_ids], bert_outputs], dim=1)
        elif combine == 'avg':
            combined_emb = mean(stack([X[token_global_ids], bert_outputs]), dim=0)
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{combine}] provided.')
        pred = bilstm_classifier(combined_emb.unsqueeze(0))

        outputs = bilstm_classifier(**inputs)



if __name__ == "__main__":
    test = GLEN_BERT_Classifier(in_dim=5, hid_dim=3, num_heads=2, out_dim=4, num_classes=2)

    test()
