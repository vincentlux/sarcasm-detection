#!/usr/bin/env python
# coding: utf-8

# test building own dataset using torchtext
import os, argparse
import numpy as np
import pandas as pd
import torch
import spacy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext import datasets


# LABEL = data.Field(sequential=False)

###########TODO: ADD submit=True args so that for submission, use all data to train?


def get_data(train, test):
    LABEL = data.LabelField(dtype=torch.float)
    COMMENT = data.Field()
    # PARENT = data.Field()
    fields = [('l', LABEL), ('c', COMMENT), (None, None)]

    print(train, test)

    train_data, test_data = data.TabularDataset.splits(
                                            path = '',
                                            train = train,
                                            test = test,
                                            format = 'tsv',
                                            fields = fields,
                                            skip_header = True
    )

    ## build own data set finish

    train_data, valid_data = train_data.split(split_ratio=0.9, random_state=random.seed(42))

    # print(vars(train_data[1]))

    COMMENT.build_vocab(train_data, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    return train_data, valid_data, test_data, COMMENT




def get_dataloader(device, train_data, valid_data, test_data, bS):
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        sort_key=lambda x: len(x.c),
        sort_within_batch=False,
        batch_size=bS,
        device=device)
    return train_iterator, valid_iterator, test_iterator

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #x = [sent len, batch size]

        embedded = self.dropout(self.embedding(x))

        #embedded = [sent len, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded)

        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        #hidden = [batch size, hid dim * num directions]
        return self.fc(hidden.squeeze(0))



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pred = model(batch.c).squeeze(1)
        loss= criterion(pred, batch.l)
        acc = binary_accuracy(pred, batch.l)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.c).squeeze(1)
            loss = criterion(predictions, batch.l)
            acc = binary_accuracy(predictions, batch.l)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict_sentiment(model, COMMENT, sentence, s_tkr):
    tokenized = [tok.text for tok in s_tkr.tokenizer(str(sentence))]
    indexed = [COMMENT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', default=20, type=int)
    parser.add_argument("--bS", default=32, type=int, help='Batch size')
    parser.add_argument('--hS', default=256, type=int, help='hidden size')
    parser.add_argument('--num_layer', default=2, type=int, help='number of layer')
    parser.add_argument('--eS', default=100, type=int, help='embedding size')
    parser.add_argument('--dr', default=0.5, type=float, help='drop out rate')
    parser.add_argument("--opt_test", action='store_true', help='if exist, use new opt')
    parser.add_argument("--lr", default=0.0001, type=float, help='learning rate')

    parser.add_argument('--bidirect', action='store_true', help='if present, use bidirectional lstm')
    parser.add_argument("--path", default='./data', type=str, help='data dir')
    parser.add_argument("--out_path", default='./result', type=str, help='out dir')
    parser.add_argument("--train_file", default='train_resplit.tsv', type=str, help='training data')
    parser.add_argument("--test_file", default='val_resplit.tsv', type=str, help='validation data')
    parser.add_argument("--sub_name", default='submit', type=str, help='name for submission tsv, .tsv will be added automatically')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('Using gpu')

    if args.bidirect:
        print('Using bidirectional LSTM')

    # load dataset
    data_path = args.path
    print(os.path.join(data_path, args.train_file))
    train_f = os.path.join(data_path, args.train_file)
    test_f = os.path.join(data_path, args.test_file)
    train_data, val_data, test_data, COMMENT = get_data(train_f, test_f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build dataloader (torchtext)
    train_iterator, valid_iterator, test_iterator = get_dataloader(device, train_data, val_data, test_data, args.bS)

    print(len(COMMENT.vocab))
    # len(COMMENT.vocab.vectors)

    inp_size = len(COMMENT.vocab)
    out_size = 1
    assert args.eS == len(COMMENT.vocab.vectors[1])
    model = RNN(inp_size, args.eS, args.hS, out_size, args.num_layer, args.bidirect, args.dr)

    # Use GloVe
    pretrained_embeddings = COMMENT.vocab.vectors
    print(len(COMMENT.vocab.vectors[1]))
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # set opt and criterion
    if args.opt_test:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
    else:
        opt = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    
    best_valid_acc = -1
    # train
    for epoch in range(args.nepoch):
        print(epoch)
        train_loss, train_acc = train(model, train_iterator, opt, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        # save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            state = {'model': model.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join('.', 'model_best.pt'))

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    spacy_tkr = spacy.load('en')
    check_point = torch.load(os.path.join('.', 'model_best.pt'))
    print(predict_sentiment(model, COMMENT, "You are amazing", spacy_tkr))
    model.load_state_dict(check_point['model'])
    print(f"Best model from epoch {check_point['epoch']}")
    print(predict_sentiment(model, COMMENT, "You are amazing", spacy_tkr))
    # submit
    sub_f = os.path.join(data_path, 'test.tsv')
    sub_df = pd.read_csv(sub_f, sep='\t')
    # add a new col
    # sub_df['label'] = -1
    result=[]
    for i in range(len(sub_df)):
        sent = sub_df.iloc[i]['comment']
        score = predict_sentiment(model, COMMENT, sent, spacy_tkr)
        # print(score)
        if score > 0.5:
            # print("no_sarc")
            result.append(1)
        else:
            result.append(0)
        if i % 1000 == 0:
            print(i, " out of ", len(sub_df), " has been processed")
    print(len(result))
    sub_df['label'] = result

    dout = os.path.join(args.out_path, args.sub_name+'.csv')
    sub_df["label"].to_csv(dout, header=["label"], index_label="id")


    ########SAVE result
