import torch
import torch.nn as nn
import torch.nn.functional as F
from FrEIA.framework import ReversibleGraphNet, InputNode, OutputNode, Node
from FrEIA.modules import AllInOneBlock
import config_and_data as c

# Define CondNet: Feed-forward network for conditional input (momenta)
class CondNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CondNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on the last layer
        return x

# Function to add conditional block to CINN model
def add_conditioning_block(model, cond_net):
    # Wrapper model to apply conditioning from CondNet
    class ConditionalCINN(nn.Module):
        def __init__(self, base_model, conditioning_net):
            super(ConditionalCINN, self).__init__()
            self.base_model = base_model
            self.cond_net = conditioning_net

        def forward(self, x, y_condition):
            # Pass the condition through CondNet
            cond_output = self.cond_net(y_condition)
            # Forward pass through reversible graph network with the conditional output
            return self.base_model(x, c=[cond_output])

    # Return the model with conditional network integrated
    return ConditionalCINN(model, cond_net)

# Subnet constructor for AllInOneBlock
def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, c.hidden_layer_sizes),
                         nn.ReLU(),
                         nn.Linear(c.hidden_layer_sizes, c_out))

# Example CINN Setup
def create_cinn_model():
    nodes = []

    # Input node
    nodes.append(InputNode(c.ndim_x, name='input'))

    # Use AllInOneBlock with subnet_constructor
    for i in range(c.N_blocks):
        nodes.append(Node(nodes[-1], 
                          AllInOneBlock, 
                          {'subnet_constructor': subnet_fc, 'permute_soft': True},
                          name=f'all_in_one_{i}'))

    # Output node
    nodes.append(OutputNode(nodes[-1], name='output'))

    # Initialize the conditional network (CondNet)
    cond_net = CondNet(input_dim=c.ndim_y, hidden_dim=c.hidden_layer_sizes, output_dim=c.ndim_x)

    # Initialize the base reversible network
    model = ReversibleGraphNet(nodes, verbose=c.verbose_construction)

    # Combine the base model with the conditional network
    model = add_conditioning_block(model, cond_net)

    return model