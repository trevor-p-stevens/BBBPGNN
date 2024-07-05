import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class BBBPGNN(torch.nn.Module):
    def __init__(self, feature_size, parameters):
        super(BBBPGNN, self).__init__()
        embedding_size = parameters['embedding_size']
        num_heads = parameters['num_heads']
        self.num_layers = parameters['num_layers']
        dropout_rate = parameters['dropout_rate']
        top_k_ratio = parameters['top_k_ratio']
        self.top_k_occurence = parameters['top_k_occurence']
        dense_neurons = parameters['dense_nuerons']
        edge_dim = parameters['edge_dim']

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.batch_norm_layers = ModuleList([])

        self.conv1 = TransformerConv(feature_size,
                                     embedding_size,
                                     heads=num_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.transf1 = Linear(embedding_size*num_heads, embedding_size)
        self.batch_norm1 = BatchNorm1d(embedding_size)

        for i in range(self.num_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=num_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))
            
            self.transf_layers.append(Linear(embedding_size*num_heads,embedding_size))
            self.batch_norm_layers.append(BatchNorm1d(embedding_size))
            if i%self.top_k_occurence == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.linear1 = Linear(embedding_size*2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))
        self.linear3 = Linear(int(dense_neurons/2), 1)

    def forward(self, x, edge_fts, edge_i, batch_i):
        x = self.conv1(x, edge_i, edge_fts)
        x = torch.relu(self.transf1(x))
        x = self.batch_norm1(x)
        g_rep = []

        for i in range(self.num_layers):
            x = self.conv_layers[i](x,edge_i,edge_fts)
            x = torch.relu(self.transf_layers[i](x))
            x = self.batch_norm_layers[i](x)

            if i % self.top_k_occurence == 0 or i == self.num_layers:
                x, edge_i, edge_fts, batch_i, _, _ = self.pooling_layers[int(i/self.top_k_occurence)](
                    x, edge_i, edge_fts, batch_i
                )
                g_rep.append(torch.cat([gmp(x, batch_i), gap(x, batch_i)], dim=1))

        x = sum(g_rep)

        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x
    

           
