import os
from collections import defaultdict
from torch import randperm
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch._utils import _accumulate
from dataset import TwitterFileArchiveDataset
from gru import GRUCell
from utils import init_weights, argmax, cuda, variable, get_sequence_from_indices
from dataset import Vocab
    
class NeuralLanguageModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, init_token, eos_token, teacher_forcing=0.7):
        super(NeuralLanguageModel,self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.init_token = init_token
        self.eos_token = eos_token
        self.vocab_size = vocab_size
        ##############################
        ### Insert your code below ###
        # create an embedding layer, a GRU cell, and the output projection layer
        ##############################
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.gru = torch.nn.GRU(embedding_size,hidden_size,1,batch_first=True)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)
        
        ###############################
        ### Insert your code above ####
        ###############################

    def cell_zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tensor of zeros of the shape of (batch_size x hidden_size)
        """
        weight = next(self.parameters()).data
        hidden = Variable(weight.new(1,batch_size, self.hidden_size).zero_())
        return hidden

    def forward(self, inputs):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of tweets' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        expected: 7*20*27
        """
        batch_size, max_len = inputs.shape
       
        hidden = self.cell_zero_state(batch_size)
        
        x_i = variable(np.full((1,), self.init_token)).expand((batch_size,))
        #print x_i
        outputs = []
        embedding = self.embedding(inputs)
        for i in range(max_len):
            ##############################
            ### Insert your code below ###
            # `output` should be the output of the network at the current timestep
            ##############################       
            current_embed = embedding[:,i:i+1,:]
            #print(current_embed.size())
            #print(hidden.size())
            output,hn = self.gru(current_embed,hidden)
            """print "output of gru"
            print output.size()"""
            output = self.projection(output)
            """print "output of projection"
            print output.size()"""
            hidden = hn
            #if 5<self.teacher_forcing():
            #output = output.squeeze()

            ###############################
            ### Insert your code above ####
            ###############################
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        
        return outputs

    def produce(self, start_tokens=None, max_len=20):
        """
        Generate a tweet using the provided start tokens at the inputs on the initial timesteps
        :param start_tokens: A tensor of the shape (n,) where n is the number of start tokens
        :param max_len: Maximum length of the tweet
        :return: Indices of the tokens of the generated tweet
        """
        hidden = self.cell_zero_state(1)
        x_i = variable(np.full((1,), self.init_token))
        
        if start_tokens is not None:
            start_tokens = variable(start_tokens)
            
        
        outputs = []
        #print start_tokens
        for i in range(max_len):
            ##############################
            ### Insert your code below ###
            # `x_i` should be the output of the network at the current timestep
            ##############################
            if i==0:
                inputs = self.embedding(x_i)
                inputs = inputs.view(1,1,200)
            elif i < start_tokens.size(0):
                inputs = self.embedding(start_tokens[i])
               
                inputs = inputs.view(1,1,200)
            else:
                
                inputs = self.embedding(x_i)
                inputs =inputs.view(1,1,200)
           
            
                
            output,hn = self.gru(inputs,hidden)
            hidden = hn
            
           
           
            #softmax_out = F.softmax(output,2)
            
            output = self.projection(output)
            output = F.softmax(output,dim=2)
            
            #print output.size()
            x_i = torch.multinomial(output.squeeze(),1)
            
            ###############################
            ### Insert your code above ####
            ###############################
            outputs.append(x_i)
    
            

        outputs = torch.cat(outputs)
        
        return outputs
    


def main():
    train_on = 'obama'  # 'trump' or 'obama'
    val_size = 0.2
    max_len = 20
    embedding_size = 200
    hidden_size = 300
    batch_size = 64
    nb_epochs = 1
    max_grad_norm = 5
    teacher_forcing = 0.7

    # load data and create datasets
    # note that they use the same Vocab object so they will share the vocabulary
    # (in particular, for a given token both of them will return the same id)
    trump_tweets_filename = 'data/trump_tweets.txt'
    obama_tweets_filename = 'data/obama_white_house_tweets.txt'
    dataset_trump = TwitterFileArchiveDataset(trump_tweets_filename, max_len=max_len)
    dataset_obama = TwitterFileArchiveDataset(obama_tweets_filename, max_len=max_len, vocab=dataset_trump.vocab)

    dataset_trump.vocab.prune_vocab(min_count=3)

    if train_on == 'trump':
        dataset_train = dataset_trump
        dataset_val_ext = dataset_obama
    elif train_on == 'obama':
        dataset_train = dataset_obama
        dataset_val_ext = dataset_trump
    else:
        raise ValueError('`train_on` cannot be {} - use `trump` or `obama`'.format(train_on))

    val_len = int(len(dataset_train) * val_size)
    train_len = len(dataset_train) - val_len
    dataset_train, dataset_val = torch.utils.data.dataset.random_split(dataset_train, [train_len, val_len])

    # note that the the training and validation sets come from the same person,
    # whereas the val_ext set come from a different person

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    data_loader_val_ext = torch.utils.data.DataLoader(dataset_val_ext, batch_size=batch_size, shuffle=False)
    print('Training on: {}'.format(train_on))
    print('Train {}, val: {}, val ext: {}'.format(len(dataset_train), len(dataset_val), len(dataset_val_ext)))

    vocab_size = len(dataset_trump.vocab)
    model = NeuralLanguageModel(
        embedding_size, hidden_size, vocab_size,
        dataset_trump.vocab[dataset_trump.INIT_TOKEN], dataset_trump.vocab[dataset_trump.EOS_TOKEN],
        teacher_forcing
    )
    model = cuda(model)
    init_weights(model)

    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset_trump.vocab[dataset_trump.PAD_TOKEN])

    phases = ['train', 'val', 'val_ext']
    data_loaders = [data_loader_train, data_loader_val, data_loader_val_ext]

    losses_history = defaultdict(list)
    for epoch in range(nb_epochs):
        for phase, data_loader in zip(phases, data_loaders):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = []
            for i, inputs in enumerate(data_loader):
                optimizer.zero_grad()

                inputs = variable(inputs)

                outputs = model(inputs)

                targets = inputs.view(-1)
                outputs = outputs.view(targets.size(0), -1)

                loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(parameters, max_grad_norm)
                    optimizer.step()

                epoch_loss.append(float(loss))

            epoch_loss = np.mean(epoch_loss)
            print('Epoch {} {}\t\tloss {:.2f}'.format(epoch, phase, epoch_loss))
            losses_history[phase].append(epoch_loss)

            # decode something in the validation phase
            if phase == 'val_ext':
                possible_start_tokens = [
                    ['I','for' ],
                ]
                start_tokens = possible_start_tokens[np.random.randint(len(possible_start_tokens))]
                start_tokens = np.array([dataset_trump.vocab[t] for t in start_tokens])
                outputs = model.produce(start_tokens, max_len=20)
                outputs = outputs.cpu().numpy()

                produced_sequence = get_sequence_from_indices(outputs, dataset_trump.vocab.id2token)
                print('{}'.format(produced_sequence))

    print('Losses:')
    print('\t'.join(phases))
    losses = [losses_history[phase] for phase in phases]
    losses = list(zip(*losses))
    for losses_vals in losses:
        print('\t'.join('{:.2f}'.format(lv) for lv in losses_vals))


if __name__ == '__main__':
    main()
