from .graph import *
from .fields import *
from .utils import prepare_dataset
import os
import random

class ClassificationDataset(object):
    "Dataset class for classification task."
    def __init__(self, path, exts, train='train', valid='valid', test='test', vocab='vocab.txt'):
        vocab_path = os.path.join(path, vocab)
        self.text = {}
        self.label = {}
        with open(os.path.join(path, train + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.text['train'] = f.readlines() 
        with open(os.path.join(path, train + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.label['train'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.text['valid'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.label['valid'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.text['test'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.label['test'] = f.readlines()
        vocab = Vocab() 
        vocab.load(vocab_path)
        self.vocab = vocab
        self.src_field = Field(vocab) 

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, template, mode='train', batch_size=32, device='cpu'):
        text, label = self.text[mode], self.label[mode]        
        n = len(text)

        order = list(range(n))
        if mode == 'train':
            random.shuffle(order)

        text_buf, label_buf = [], [] 
        cnt = 0
        for idx in order:
            text_i = self.src_field(
                text[idx].strip().split())
            label_i = self.vocab[label[idx].strip()]
            text_buf.append(text_i)
            label_buf.append(label_i)
            cnt += len(text_i)
            if cnt >= batch_size:
                yield template(text_buf, label_buf, device=device)
                text_buf, label_buf = [], [] 
                cnt = 0

        if len(text_buf) != 0:
            yield template(text_buf, label_buf, device=device)

class TranslationDataset(object):
    '''
    Dataset class for translation task.
    By default, the source language shares the same vocabulary with the target language.
    '''
    INIT_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    MAX_LENGTH = 100
    def __init__(self, path, exts, train='train', valid='valid', test='test', vocab='vocab.txt', replace_oov=None):
        vocab_path = os.path.join(path, vocab)
        self.src = {}
        self.tgt = {}
        with open(os.path.join(path, train + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['train'] = f.readlines()
        with open(os.path.join(path, train + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['train'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['valid'] = f.readlines()
        with open(os.path.join(path, valid + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['valid'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[0]), 'r', encoding='utf-8') as f:
            self.src['test'] = f.readlines()
        with open(os.path.join(path, test + '.' + exts[1]), 'r', encoding='utf-8') as f:
            self.tgt['test'] = f.readlines()

        if not os.path.exists(vocab_path):
            self._make_vocab(vocab_path)

        vocab = Vocab(init_token=self.INIT_TOKEN,
                      eos_token=self.EOS_TOKEN,
                      pad_token=self.PAD_TOKEN,
                      unk_token=replace_oov)
        vocab.load(vocab_path)
        self.vocab = vocab
        strip_func = lambda x: x[:self.MAX_LENGTH]
        self.src_field = Field(vocab,
                               preprocessing=None,
                               postprocessing=strip_func)
        self.tgt_field = Field(vocab,
                               preprocessing=lambda seq: [self.INIT_TOKEN] + seq + [self.EOS_TOKEN],
                               postprocessing=strip_func)

    def get_seq_by_id(self, idx, mode='train', field='src'):
        "get raw sequence in dataset by specifying index, mode(train/valid/test), field(src/tgt)"
        if field == 'src':
            return self.src[mode][idx].strip().split()
        else:
            return [self.INIT_TOKEN] + self.tgt[mode][idx].strip().split() + [self.EOS_TOKEN]

    def _make_vocab(self, path, thres=2):
        word_dict = {}
        for mode in ['train', 'valid', 'test']:
            for line in self.src[mode] + self.tgt[mode]:
                for token in line.strip().split():
                    if token not in word_dict:
                        word_dict[token] = 0
                    else:
                        word_dict[token] += 1

        with open(path, 'w') as f:
            for k, v in word_dict.items():
                if v > 2:
                    print(k, file=f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def sos_id(self):
        return self.vocab[self.INIT_TOKEN]

    @property
    def eos_id(self):
        return self.vocab[self.EOS_TOKEN]

    def __call__(self, template, mode='train', batch_size=32, k=1,
                 device='cpu', dev_rank=0, ndev=1):
        '''
        Create a batched graph correspond to the mini-batch of the dataset.
        args:
            graph_pool: a GraphPool object for accelerating.
            mode: train/valid/test
            batch_size: batch size
            device: torch.device
            k: beam size(only required for test)
        '''
        src_data, tgt_data = self.src[mode], self.tgt[mode]
        n = len(src_data)
        # make sure all devices have the same number of batch
        n = n // ndev * ndev

        # XXX: is partition then shuffle equivalent to shuffle then partition?
        order = list(range(dev_rank, n, ndev))
        if mode == 'train':
            random.shuffle(order)

        src_buf, tgt_buf = [], []

        cnt = 0
        for idx in order:
            src_sample = self.src_field(
                src_data[idx].strip().split())
            tgt_sample = self.tgt_field(
                tgt_data[idx].strip().split())
            src_buf.append(src_sample)
            tgt_buf.append(tgt_sample)
            cnt += len(src_sample)
            if cnt >= batch_size:
                if mode == 'test':
                    yield template.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, device=device)
                else:
                    yield template(src_buf, tgt_buf, device=device)
                src_buf, tgt_buf = [], []
                cnt = 0

        if len(src_buf) != 0:
            if mode == 'test':
                yield template.beam(src_buf, self.sos_id, self.MAX_LENGTH, k, device=device)
            else:
                yield template(src_buf, tgt_buf, device=device)

    def get_sequence(self, batch):
        "return a list of sequence from a list of index arrays"
        ret = []
        filter_list = set([self.pad_id, self.sos_id, self.eos_id])
        for seq in batch:
            try:
                l = seq.index(self.eos_id)
            except:
                l = len(seq)
            ret.append(' '.join(self.vocab[token] for token in seq[:l] if not token in filter_list))
        return ret

def get_dataset(dataset):
    "we wrapped a set of datasets as example"
    prepare_dataset(dataset)
    if dataset == 'babi':
        raise NotImplementedError
    elif dataset == 'copy' or dataset == 'sort':
        return TranslationDataset(
            'data/{}'.format(dataset),
            ('in', 'out'),
            train='train',
            valid='valid',
            test='test',
        )
    elif dataset == 'multi30k':
        return TranslationDataset(
            'data/multi30k',
            ('en.atok', 'de.atok'),
            train='train',
            valid='val',
            test='test2016',
            replace_oov='<unk>'
        )
    elif dataset == 'wmt14':
        return TranslationDataset(
            'data/wmt14',
            ('en', 'de'),
            train='train.tok.clean.bpe.32000',
            valid='newstest2013.tok.bpe.32000',
            test='newstest2014.tok.bpe.32000',
            vocab='vocab.bpe.32000')
    elif dataset == 'long' or dataset == 'short':
        return ClassificationDataset(
            'data/' + dataset,
            ('in', 'out'),
            train='train',
            valid='val',
            test='test')
    else:
        raise KeyError()
