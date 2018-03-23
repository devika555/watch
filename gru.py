import torch
import torch.cuda
import torch.nn.functional as F
from torch.nn import Parameter


class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ##############################
        ### Insert your code below ###
        # create the weight matrices and biases. Use the `torch.nn.Parameter` class
        ##############################
        self.weight_input = Parameter(torch.zeros(input_size))
        self.weight_hidden = Parameter(torch.zeros(hidden_size))
        ###############################
        ### Insert your code above ####
        ###############################

    def forward(self, inputs, hidden):
        """
        Perform a single timestep of a GRU cell using the provided input and the hidden state
        :param inputs: Current input
        :param hidden: Hidden state from the previous timestep
        :return: New hidden state
        """
        ##############################
        ### Insert your code below ###
        # Perform the calculation according to the reference paper (see the README)
        # hidden_new is the new hidden state at the current timestep
        ##############################
        linear_i = F.linear(inputs,weight_input)
        linear_h = F.linear(hidden,weight_hidden)

        i_r,i_i,i_n = linear_i.chunk(3,1)
        h_r,h_i,h_n = linear_h.chunk(3,1)

        resetgate= F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i+h_i)
        newgate = F.tanh(i_n+resetgate*h_n)
        hidden_new = newgate + inputgate*(hidden - newgate)
        
        ###############################
        ### Insert your code above ####
        ###############################
        return hidden_new
