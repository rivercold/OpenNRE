import torch
from torch import nn, optim
import json
from .data_loader import SentenceRELoader, BagRELoader
from .utils import AverageMeter
from tqdm import tqdm
import os
import torch.nn.functional as F

class BagRE(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd',
                 bag_size=None,
                 loss_weight=False):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.bag_size = bag_size
        # Load data
        if train_path != None:
            self.train_loader = BagRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                bag_size=bag_size,
                entpair_as_bag=False)

        if val_path != None:
            self.val_loader = BagRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=None,
                entpair_as_bag=True)
        
        if test_path != None:
            self.test_loader = BagRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                bag_size=None,
                entpair_as_bag=True
            )
        # Model
        self.model = nn.DataParallel(model)
        # Criterion
        if loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'bert_adam'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, adv=False, epsilon=0.0001):
        best_auc = 0
        if adv:
            print ("We use adversarial softmax for training with Epsilon {}".format(epsilon))
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                h, logits = self.model(label, scope, *args, bag_size=self.bag_size)
                # print ("logits", logits.size())
                # print ("logits[0]", logits[0], torch.sum(logits[0]))
                loss = self.criterion(logits, label)
                if adv:
                    norm_h = -epsilon * torch.norm(h.detach(), p=2, dim=1)  # m x 1
                    pertub = torch.zeros(logits.size()).float().cuda()  # m x k
                    for i in range(pertub.size(0)):
                        col = label[i]
                        pertub[i][col] = norm_h[i].data
                    adv_logits = pertub + logits
                    adv_loss = self.criterion(adv_logits, label)
                    loss  = adv_loss

                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)
                
                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Val 
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("auc: %.4f" % result['auc'])
            print("f1: %.4f" % (result['f1']))
            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_auc = result['auc']
        print("Best auc on val set: %f" % (best_auc))

    def train_adv_model(self, epsilon=0.01):
        best_auc = 0
        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits, w = self.model(label, scope, *args, bag_size=self.bag_size)
                # print ("logits", logits.size())
                # print ("logits[0]", logits[0], torch.sum(logits[0]))
                # print ("w matrix", w.size())
                loss = self.criterion(logits, label)
                with torch.enable_grad():
                    grad = torch.autograd.grad(loss, [w])[0]  # bs, l, d
                    grad = F.normalize(grad, dim=2)
                    # print (grad[0,0,:].size(), grad[0,0,:], torch.norm(grad[0,0,:]))
                    pertub_x = epsilon * grad  # m x 1
                w_adv = w + pertub_x.detach()
                adv_logits, _ = self.model(label, scope, w_adv, *data[4:], bag_size=self.bag_size, input_feat=True)
                adv_loss = self.criterion(adv_logits, label)
                loss = adv_loss

                score, pred = logits.max(-1)  # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_pos_acc.update(pos_acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg)

                # Optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Val
            print("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader)
            print("auc: %.4f" % result['auc'])
            print("f1: %.4f" % (result['f1']))
            if result['auc'] > best_auc:
                print("Best ckpt and saved.")
                torch.save({'state_dict': self.model.module.state_dict()}, self.ckpt)
                best_auc = result['auc']
        print("Best auc on val set: %f" % (best_auc))

    def eval_model(self, eval_loader):
        self.model.eval()
        with torch.no_grad():
            t = tqdm(eval_loader)
            pred_result = []
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                bag_name = data[1]
                scope = data[2]
                args = data[3:]
                logits = self.model(None, scope, *args, train=False) # results after softmax
                for i in range(logits.size(0)):
                    for relid in range(self.model.module.num_class):
                        if self.model.module.id2rel[relid] != 'NA':
                            pred_result.append({
                                'entpair': bag_name[i][:2],
                                'relation': self.model.module.id2rel[relid], 
                                'score': logits[i][relid].item()
                            })
            result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.module.load_state_dict(state_dict)
