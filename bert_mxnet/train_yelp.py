import random
import os
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, autograd
import gluonnlp as nlp
from bert import *
from gluonnlp.data import TSVDataset
from glob import glob
from os.path import expanduser
import json
import shutil
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, zero_one_loss

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# import holoviews as hv
# from holoviews import opts, dim
# hv.extension('bokeh')

# seeding all randomizers
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(100)
random.seed(100)
mx.random.seed(100)

# use GPU when available otherwise use CPU
ctx = mx.gpu(0) if mx.test_utils.list_gpus() else mx.cpu()
import warnings
warnings.filterwarnings('ignore')

HOME = expanduser("~")
DATADIR = '{}/BERTvsULMFIT/data_yelp/'.format(HOME)
filename = '{}/output/net.params'.format(HOME)
max_epochs = 3

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
# print(bert_base)

nclasses = {"yelp":5, "20ng":20, "imdb":2, "r8":8, "r52":52, "ohsumed_all": 23, "ohsumed_first": 23}

dataset_name = 'yelp'
# network params
# maximum sequence length
num_discard_samples = 0
max_len = 150
# number of classes
n_classes = nclasses[dataset_name]
all_labels = [str(_) for _ in range(n_classes)]
# batch size
batch_size = 32
# initial learning rate
lr = 5e-6
# gradient clipping value
grad_clip = 1
# log to screen every 50 batch
log_interval = 50
# train until we fail to beat the current best validation loss for 5 consecutive epochs
max_patience = 5
pair = False

def print_results(y_true, y_pred):
    print("Error Rate: {:2.2f}".format(100*zero_one_loss(y_true, y_pred)))
    print("Accuracy: {:2.2f}".format(100*accuracy_score(y_true, y_pred)))
    print("F1-Score: {:2.2f}".format(100*f1_score(y_true, y_pred, average="macro")))
    print("Precision: {:2.2f}".format(100*precision_score(y_true, y_pred, average="macro")))
    print("Recall: {:2.2f}".format(100*recall_score(y_true, y_pred, average="macro")))
    fig, ax = plt.subplots(figsize=(18, 10))
    x = sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, ax=ax)
    x.invert_yaxis()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    return fig

class Dataset(TSVDataset):
    """Train dataset.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    root : str, default 'dir containing train/dev/test datasets'
    """
    def __init__(self, segment='train', root='.', n_classes=2):
        self._supported_segments = ['train', 'dev', 'test']
#         self.field_separator = nlp.data.Splitter(',')
        assert segment in self._supported_segments, 'Unsupported segment: %s'%segment
        path = os.path.join(root, '%s.tsv'%segment)
        A_IDX, LABEL_IDX = 2, 3
        fields = [A_IDX, LABEL_IDX]
        self.n_classes=n_classes
        super(Dataset, self).__init__(path, field_indices=fields, num_discard_samples=num_discard_samples)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return [str(_) for _ in range(self.n_classes)]

if __name__=='__main__':
    data_train = Dataset(root=DATADIR, segment='train', n_classes=n_classes)
    data_dev = Dataset(root=DATADIR, segment='dev', n_classes=n_classes)
    data_test = Dataset(root=DATADIR, segment='test', n_classes=n_classes)

    sample_id = np.random.randint(0, len(data_train))
    print('<<<<TEXT>>>>')
    print(data_train[sample_id][0])
    print("<<<<LABEL>>>>")
    print(data_train[sample_id][1])

    bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                                 dataset_name='book_corpus_wiki_en_uncased',
                                                 pretrained=True, ctx=ctx, use_pooler=True,
                                                 use_decoder=False, use_classifier=False)
    #print(bert_base)

    # use the vocabulary from pre-trained model for tokenization
    tokenizer = tokenization.FullTokenizer(vocabulary, do_lower_case=True)
    transform = dataset.ClassificationTransform(tokenizer, all_labels, max_len, pair=False)


    data_train = data_train.transform(transform)
    data_dev = data_dev.transform(transform)
    data_test = data_test.transform(transform)
    print('token ids = \n%s'%data_train[sample_id][0])

    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, last_batch='rollover')
    dev_dataloader = mx.gluon.data.DataLoader(data_dev, batch_size=batch_size, shuffle=False, last_batch='rollover')
    test_dataloader = mx.gluon.data.DataLoader(data_dev, batch_size=batch_size, shuffle=False, last_batch='rollover')

    model = bert.BERTClassifier(bert_base, num_classes=n_classes, dropout=0.1)
    # only need to initialize the classifier layer.
    model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.hybridize(static_alloc=True)

    # softmax cross entropy loss for classification
    loss_function = gluon.loss.SoftmaxCELoss()
    loss_function.hybridize(static_alloc=True)

    metric = mx.metric.Accuracy()

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'epsilon': 1e-9})

    # collect all differentiable parameters
    # grad_req == 'null' indicates no gradients are calculated (e.g. constant parameters)
    # the gradients for these params are clipped later
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']

    train_step = 0
    epoch_id = 0
    best_loss = None
    patience = 0
    while epoch_id<max_epochs:
        metric.reset()
        step_loss = 0
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            # load data to GPU
    #         print(batch_id)
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                # forward computation
                out = model(token_ids, segment_ids, valid_length.astype('float32'))
                ls = loss_function(out, label).mean()

            # backward computation
            ls.backward()

            # gradient clipping
            grads = [p.grad(c) for p in params for c in [ctx]]
            gluon.utils.clip_global_norm(grads, grad_clip)

            # parameter update
            trainer.step(1)
            step_loss += ls.asscalar()
            metric.update([label], [out])
            if (batch_id + 1) % (log_interval) == 0:
                print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                             .format(epoch_id, batch_id + 1, len(train_dataloader),
                                     step_loss / log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                step_loss = 0
            train_step +=1
        epoch_id+=1
        ########################
        #### RUN EVALUATION ####
        ########################
        dev_loss = []
        y_true = []
        y_pred = []
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dev_dataloader):
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # get logits and loss value
            out = model(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()
            dev_loss.append(ls.asscalar())
            probs = out.softmax()
            pred = nd.argmax(probs, axis=1).asnumpy()
            y_true.extend(list(np.reshape(label.asnumpy(), (-1))))
            y_pred.extend(pred)
        dev_loss = np.mean(dev_loss)
        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        print('EVALUATION ON DEV DATASET:')
        print('dev mean loss: {:.4f}, f1-score: {:.4f}, accuracy: {:0.4f}'.format(dev_loss, f1, acc))
        if best_loss is None or dev_loss < best_loss:
            model.save_parameters('{}_best'.format(filename, train_step))
            best_loss = dev_loss
            print('dev best loss updated: {:.4f}'.format(best_loss))
            patience=0
        else:
            if patience == max_patience:
                model.save_parameters('{}_{}'.format(filename, train_step))
                break
            new_lr = trainer.learning_rate/2
            trainer.set_learning_rate(new_lr)
            print('patience #{}: reducing the lr to {}'.format(patience, new_lr))
            patience+=1

    # RUN TEST
    import pdb; pdb.set_trace()
    # load the best pre-trained model for evaluation
    best_ckpt = glob('{}*best'.format(filename))[0]
    model = bert.BERTClassifier(bert_base, num_classes=n_classes, dropout=0.1)
    model.load_parameters(best_ckpt, ctx=ctx)

    y_true = []
    y_pred = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx)
        segment_ids = segment_ids.as_in_context(ctx)
        label = label.as_in_context(ctx)
        out = model(token_ids, segment_ids, valid_length.astype('float32')).softmax()
        pred = nd.argmax(out, axis=1).asnumpy()
        y_true.extend(list(np.reshape(label.asnumpy(), (-1))))
        y_pred.extend(pred)
    assert len(y_true)==len(y_pred)

    fig = print_results(np.reshape(y_true, (-1)), y_pred)
