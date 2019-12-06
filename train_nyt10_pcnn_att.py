import sys, json
import torch
import os
import numpy as np
import opennre
from opennre import encoder, model, framework
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='nyt_pcnn_att', help='name of the model')
args = parser.parse_args()

# Some basic settings
root_path = '.'
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')

# Check data
opennre.download_nyt10(root_path=root_path)
opennre.download_glove(root_path=root_path)
rel2id = json.load(open(os.path.join(root_path, 'benchmark/nyt10/nyt10_rel2id.json')))
wordi2d = json.load(open(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'pretrain/glove/glove.6B.50d_mat.npy'))

# Define the sentence encoder

sentence_encoder = opennre.encoder.PCNNEncoder(
    token2id=wordi2d,
    max_length=120,
    word_size=50,
    position_size=5,
    hidden_size=230,
    blank_padding=True,
    kernel_size=3,
    padding_size=1,
    word2vec=word2vec,
    dropout=0.5
)

'''
sentence_encoder = opennre.encoder.RNNEncoder(
    token2id=wordi2d,
    max_length=120,
    hidden_size=150,
    word_size=50,
    position_size=5,
    blank_padding=True,
    word2vec=word2vec,
    kernel_size=3,
    padding_size=1,
    dropout=0.5,
)
'''
# Define the model
model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)

ckpt = 'ckpt/{}.pth.tar'.format(args.model_name)
if "adv" in args.model_name:
    adv_flag = True
else:
    adv_flag = False

# Define the whole training framework
framework = opennre.framework.BagRE(
    train_path='benchmark/nyt10/nyt10_train.txt',
    val_path='benchmark/nyt10/nyt10_val.txt',
    test_path='benchmark/nyt10/nyt10_test.txt',
    model=model,
    ckpt=ckpt,
    batch_size=160,
    max_epoch=60,
    weight_decay=0,
    lr=0.1,
    opt='sgd')

# Train the model
# framework.train_model(adv=adv_flag)
framework.train_adv_model()

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)

# Print the result
print('AUC on test set: {:.3f}'.format(result['auc']))
print('F1 on test set: {:.3f}'.format(result['f1']))
print('Mean-Prec on test set: {:.3f}'.format(result['mean_prec']))
print ("P@100: {:.3f}".format(result[99]))
print ("P@1000: {:.3f}".format(result[999]))

