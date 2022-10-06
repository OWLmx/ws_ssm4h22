
import pandas as pd
import numpy as np
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class SequenceEventsByTermDataset(Dataset):
    def __init__(self, df:pd.DataFrame, input_feats=['evvani1_logits_0', 'evvani1_logits_1', 'evvani1_logits_2'], instance_id=lambda x: '{}:{}'.format(x.file, x.id), label_field='event' ):#id_column='id'):
        # def __init__(self, text, labels)
        self.label_mapper= {'Disposition': 0, 'NoDisposition': 1, 'Undetermined': 2} # avoid confussion with padding value (0)
        logits = []
        labels = []
        ids = []
        for kgrp, grp in df.groupby(by=['file', 'term']):
            if len(grp) > 1: # only consider sequences with more than one term mention (single sequences consist should stay with the transformer's prediction)
                grp = grp.sort_values(by=['ith_pos'], ascending=True)
                # logits_x = grp[[f'{logits_to_use}_logits_0', f'{logits_to_use}_logits_1', f'{logits_to_use}_logits_2']].values
                logits_x = grp[input_feats].values
                logits.append(logits_x)
                if label_field and label_field in grp.columns:                    
                    labels_x = grp[[label_field]].values
                    labels.append( [ self.label_mapper[l[0]] for l in labels_x] )
                else: # only inference
                    labels.append( [-1]*len(grp) )
                # ids.append(grp[[id_column]].values)                
                ids.append(grp.apply(instance_id, axis=1).values)

        self.logits = logits
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.logits[idx]
        id = self.ids[idx]
        sample = {"logits": data, "label": label, "id": id}
        return sample


def collate_batch(batch, device='cpu'):
    # print("********************************")
    # print(batch)
    logits = [torch.tensor(b['logits']) for b in batch]
    labels = [torch.tensor(b['label']) for b in batch]
    padded_logits = pad_sequence(logits, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    # print(padded_labels.shape)
    # ids = [ sum(b['id'].tolist(), []) for b in batch] # ids (str) as nested list not padded 
    ids = [ (b['id'].tolist()) for b in batch] # ids (str) as nested list not padded 
    
    
    # mask = (padded_labels != 0).to(device)
    mask = (padded_labels >= 0).to(device)
    # padded_labels[padded_labels==-1]=0
    
    #   return text_list.to(device),label_list.to(device)    
    return padded_logits, padded_labels, mask, ids


class Reclassifier(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.crf = CRF(num_labels)

    def forward(self, logits, labels=None, mask=None, return_dict=None):

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels, mask), self.crf.viterbi_decode(logits, mask=mask)
            # loss = 0 - log_likelihood.sum() # sum over batch losses
            loss = 0 - log_likelihood.mean() # sum over batch losses
            # print(f"LogLikelihood: {log_likelihood.sum()}, loss: {loss}")
        else:
            tags = self.crf.viterbi_decode(logits)
        # print(tags)
        # print(type(tags))
        # print(len(tags))
        # tags = torch.tensor(tags)

        # if not return_dict:
        #     output = (tags,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return loss, tags

    def predict(self, logits, mask):
        with torch.no_grad():
            yhat = self.crf.viterbi_decode(logits, mask) # torch.ones(y.shape).type(torch.bool) )# mask) # because predicting one by one mask all true
        return yhat


def train_loop(dataloader, model, loss_fn, optimizer, loss_every=100, print_loss=False):
    size = len(dataloader.dataset)
    losses = []
    for batch, (X, y, mask, ids) in enumerate(dataloader):
        # inputs, labels, mask = data

        # forward + backward + optimize
        loss, tags = model(X, y, mask)
                
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss_every and batch % loss_every == 0:
            loss, current = loss.item(), batch * len(X)
            losses.append(loss)
            if print_loss:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def test_loop(dataloader, model, loss_fn):
    losses, accuracies = [],[]
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    num_instances = 0
    with torch.no_grad():
        for X, y, mask, ids in dataloader:
            test_lossx, pred = model(X, y, mask)
            matchs = torch.eq(torch.tensor(sum(pred, [])).reshape(-1), torch.masked_select(y, mask)) # flatten yhat and ytrue flattenized by masked_select
            correct += matchs.sum().item()
            num_instances += mask.sum().item()
            # print(f"\t {matchs.sum().item()}  from {mask.sum().item()}")
            test_loss += test_lossx.item()
            losses.append(test_lossx.item())
            accuracies.append(correct)

    # print(test_loss)
    # print(test_losses)
    test_loss /= num_batches
    # correct /= size
    correct /= num_instances
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct, losses, accuracies



def train(dl_train, dl_valid, out_dir=""):
    # %matplotlib qt


    reclassifier = Reclassifier(num_labels=3)
    # optimizer = optim.SGD(reclassifier.parameters(), lr=0.000001, momentum=0.9)
    optimizer = optim.AdamW(reclassifier.parameters(), lr=1e-4, weight_decay=1e-3)
    patience = 20


    epochs = 350
    best_epoch = -1
    best_loss = 1000.0
    best_acc = 0.0
    train_losses_all, test_losses_all, test_accuracies_all, best_test_losses, curr_train_loss = [], [], [], [], []
    train_losses_avg, test_losses_avg = [], []
    num_cycles_notbest = 0
    # for t in tqdm(range(epochs), total=epochs, unit='epoch'):
    with tqdm(range(epochs), total=epochs, unit='epoch') as pbar:
        for t in pbar:
            # print(f"Epoch {t+1}\n-------------------------------")
            train_losses = train_loop(dl_train, reclassifier, loss_fn=None, optimizer=optimizer, loss_every=100, print_loss=False)
            test_loss, correct, test_losses, test_accuracies = test_loop(dl_valid, reclassifier, loss_fn=None)
            train_losses_all.extend(train_losses)
            test_losses_all.extend(test_losses)
            test_accuracies_all.extend(test_accuracies)
            train_losses_avg.append(np.mean(train_losses))
            test_losses_avg.append(np.mean(test_losses))
            if test_loss < best_loss:
                # print(f"***********  Best loss [Epoch: {t}] [{test_loss} < {best_loss}] ... dumping")
                torch.save(reclassifier.state_dict(), path.join(out_dir, "seq_reclassifier.pt") )
                best_loss = test_loss
                best_acc = correct
                best_epoch = t
                pbar.set_postfix(best_loss=test_loss, best_accuracy=100.*correct, best_epoch=t)

                best_test_losses.append(best_loss)
                curr_train_loss.append(np.min(train_losses))
                # print('-------- Current params---------')
                # print(list(reclassifier.parameters()))
                # print('-----------------')

                num_cycles_notbest = 0
            else:
                num_cycles_notbest += 1
                if num_cycles_notbest >= patience:
                    break

            # plt.plot(train_losses_all, label='Train loss')
            # plt.plot(test_losses_all, label='Test loss')
            plt.plot(best_test_losses, 'og', label='Best test loss')
            # plt.plot(curr_train_loss, label='Avg train loss')
            plt.plot(train_losses_avg, label='Avg train loss')
            plt.plot(test_losses_avg, label='Avg test loss')
            plt.legend()
            plt.draw()
            plt.pause(0.1)
            plt.cla()    


    # plt.plot(train_losses_all, label='Train loss')
    # plt.plot(test_losses_all, label='Test loss')
    plt.plot(best_test_losses, 'og', label='Best test loss')
    # plt.plot(curr_train_loss, label='Avg train loss')
    plt.plot(train_losses_avg, label='Avg train loss')
    plt.plot(test_losses_avg, label='Avg test loss')
    plt.legend()
    plt.show()
    print("Done!")


    print(f"Best validation: \n Epoch: {best_epoch} \n Accuracy: {(100*best_acc):>0.1f}%, Avg loss: {best_loss:>8f} \n")

    # print('-------- Final params---------')
    # print(list(reclassifier.parameters()))
    # print('-----------------')



def predict(model, df, input_feats:list):
    ds_test = SequenceEventsByTermDataset(df, input_feats=input_feats) 
    print(f"Test : {df.shape} -> {len(ds_test)} -> Labels: { len(sum(ds_test.labels, []))} ")
    dl_test = DataLoader(ds_test, batch_size=1, collate_fn=collate_batch, shuffle=False)
    ytrues, yhats, ids = [], [], []
    for X, y, mask, idsx in dl_test:
        # yhat = model.predict(X, y, mask)
        with torch.no_grad():
            yhat = model.crf.viterbi_decode(X, torch.ones(y.shape).type(torch.bool) )# mask) # because predicting one by one mask all true
            yhats.extend( sum(yhat, []) )
            ytrues.extend( y.reshape(-1).tolist() ) 
            ids.extend(idsx)            
        # print(f"{ytrues} --> {yhats}")
        # matchs = torch.eq(torch.tensor(sum(pred, [])).reshape(-1), torch.masked_select(y, mask)) # flatten yhat and ytrue flattenized by masked_select
        # correct += matchs.sum().item()
    return ytrues, yhats, sum(ids, [])


def reclasify_event_sequences(df, current_yhat_column, new_yhat_column, reclassifier, input_feats:list, collect_reclses=True, debug=False ):
    # predict reclassification (only applied to sequences > 2)
    _, yhats, ids = predict(reclassifier, df, input_feats=input_feats)

    # default classification = current_yhat
    df[new_yhat_column] = df[current_yhat_column]

    label_mapper = list({'Disposition': 0, 'NoDisposition': 1, 'Undetermined': 2}.keys())
    reclassifications = {}
    # change reclassified (using ids)
    for id, yhati in zip(ids, yhats):
        file, idx = id.split(':')
        yhat = label_mapper[yhati]
        if collect_reclses:
            prev_value = df.loc[(df['file']==file) & (df['id']==idx), current_yhat_column ].values[0]
            if prev_value != yhat:
                rcls_type = f"{prev_value}:{yhat}"
                reclassifications[rcls_type] = reclassifications[rcls_type] + 1 if rcls_type in reclassifications else 1
                if debug:
                    print(f"... reclassifiying [{id}] from [{prev_value}] to [{yhat}] ")

        df.loc[(df['file']==file) & (df['id']==idx), new_yhat_column ] = yhat
        

    return df, reclassifications