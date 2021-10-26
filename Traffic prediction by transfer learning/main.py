import pickle
import json
import os
import scipy
import numpy as np
import scipy.stats as stats
from scipy import sparse
from dgl.utils import expand_as_pair
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

import random

patience = 15
batch_size = 32

lr = 0.005
patience = 10
batch_size = 32
start_epoch=0

rawadj = pickle.load(open('speed_data_processed/adj_sliced.pkl','rb'))
comm = pickle.load(open('speed_data_processed/comm_sliced.pkl','rb'))
rd_ids = pickle.load(open('speed_data_processed/all_road_id_slices.pkl','rb'))

print('===== loading start =====')
data = np.load('hk_speed_init.npy')                     #(T,N,6)
n_nodes = data.shape[1]


def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

td_p1 = []
td_p2 = []
td_p3 = []
td_p4 = []
td_p5 = []
didi_train = []
for c in comm:
    remainder = c['td_road']
    didis = c['didi_road']
    sp1,remainder = split(remainder,ratio=1/5.0)
    sp2,remainder = split(remainder,ratio=1/4.0)
    sp3,remainder = split(remainder,ratio=1/3.0)
    sp4,sp5 = split(remainder,ratio=1/2.0)
    didi_train+=didis
    td_p1+=sp1
    td_p2+=sp2
    td_p3+=sp3
    td_p4+=sp4
    td_p5+=sp5
    
td_train = td_p1+td_p2+td_p3
td_val = td_p4
td_test = td_p5
td_all = td_train+td_val+td_test

didi_all_mask = np.zeros((rawadj.shape[0]))
td_all_mask = np.zeros((rawadj.shape[0]))
td_train_mask = np.zeros((rawadj.shape[0]))
td_val_mask = np.zeros((rawadj.shape[0]))
td_test_mask = np.zeros((rawadj.shape[0]))
for i in didi_train:
    didi_all_mask[i] = 1
for i in td_all:
    td_all_mask[i] = 1
for i in td_train:
    td_train_mask[i] = 1
for i in td_val:
    td_val_mask[i] = 1
for i in td_test:
    td_test_mask[i] = 1
    
from datetime import datetime,timedelta
import numpy as np

template = "%Y%m%d%H%M%S"
start_time = datetime.strptime("20201001000000",template)
#start_time = datetime.strptime("20201201000000",template)
end_time = datetime.strptime("20210201000000",template)
total_ts = int((end_time-start_time).total_seconds()/(60*10))

start_ts = (24*6*9+1)
sample_range = list(map(lambda x:-x,[24*6*2+1,24*6*2,24*6*2-1,24*6*1+1,24*6*1,24*6*1-1,4,3,2,1]))

import dgl
import numpy as np

from dgl.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

device = 'cuda'

# Note: This is a simple extension of the GATConv in DGL
class BatchGATConv(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(BatchGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = self.leaky_relu

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, verbose = False):
        """Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        num_node = feat.shape[0]
        num_batch = feat.shape[1]
       
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(num_node, num_batch, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(num_node, num_batch, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                num_node, num_batch, self._num_heads, self._out_feats)
           
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # el: (1, num_batch, num_heads)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        if verbose:
            print(graph.edata['a'])
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(num_node, num_batch, -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class Net(nn.Module):
    def __init__(self, in_dim, out_dim,n_heads,n_align_layers,n_conv_layers):
        super(Net, self).__init__()
        self.n_heads = n_heads
        self.n_align_layers = n_align_layers
        self.n_conv_layers = n_conv_layers
        align_out_dim = [10 for _ in range(n_align_layers)]
        conv_out_dim = [10 for _ in range(n_conv_layers)]
        conv_out_dim[-1] = 1
        self.W_align_mu_r = nn.ModuleList([BatchGATConv(10, align_out_dim[i], num_heads=n_heads) for i in range(n_align_layers)])
        self.W_align_sigma_r = nn.ModuleList([BatchGATConv(10, align_out_dim[i], num_heads=n_heads) for i in range(n_align_layers)])
        self.W_align_mu_p = nn.ModuleList([BatchGATConv(10, align_out_dim[i], num_heads=n_heads) for i in range(n_align_layers)])
        self.W_align_sigma_p = nn.ModuleList([BatchGATConv(10, align_out_dim[i], num_heads=n_heads) for i in range(n_align_layers)])
        self.W_a_reg_mu = nn.ModuleList([nn.Linear(10,conv_out_dim[i]) for i in range(n_align_layers)])
        self.W_c_reg_mu = nn.ModuleList([nn.Linear(10,conv_out_dim[i]) for i in range(n_align_layers)])
        self.W_a_reg_sigma = nn.ModuleList([nn.Linear(10,conv_out_dim[i]) for i in range(n_align_layers)])
        self.W_c_reg_sigma = nn.ModuleList([nn.Linear(10,conv_out_dim[i]) for i in range(n_align_layers)])
        self.W_conv_mu_r = nn.ModuleList([BatchGATConv(10, conv_out_dim[i], num_heads=n_heads) for i in range(n_conv_layers)])
        self.W_conv_sigma_r = nn.ModuleList([BatchGATConv(10, conv_out_dim[i], num_heads=n_heads) for i in range(n_conv_layers)])
        self.W_conv_mu_p = nn.ModuleList([BatchGATConv(10, conv_out_dim[i], num_heads=n_heads) for i in range(n_conv_layers)])
        self.W_conv_sigma_p = nn.ModuleList([BatchGATConv(10, conv_out_dim[i], num_heads=n_heads) for i in range(n_conv_layers)])
        self.LN = LayerNorm((n_nodes,batch_size,align_out_dim[0]),elementwise_affine=False)
    def forward(self,x,mode = 'train',pseudo_adj=None,real_adj=None):
        # calculate alignment for current timestep  
        
        if(mode == 'align1'):
            h = x.permute(1,2,0,3)[:,:,:,:3]  #(T,N,B,F) -> (N,B,T,F)
            mu = h[:,:,:,0]
            sigma = h[:,:,:,1]
            mu_r = mu
            sigma_r = sigma
            mu_rl = mu
            mu_p = mu
            sigma_rl = sigma
            sigma_p = sigma
            
            #for i in range(self.n_align_layers-1):
            #    mu_r = self.LN(self.W_a_reg_mu[i](mu_r))
            #    sigma_r = self.LN(self.W_a_reg_sigma[i](sigma_r))                            
            for i in range(self.n_align_layers-1):
                mu_p = self.LN(self.W_align_mu_p[i](pseudo_adj,mu_p).mean(-2))
                sigma_p = self.LN(self.W_align_sigma_p[i](pseudo_adj,sigma_p).mean(-2))
                h_p = (mu_p+sigma_p)/2
                mu_p = h_p
                sigma_p = h_p            
            for i in range(self.n_align_layers-1):
                mu_rl = self.LN(self.W_align_mu_r[i](real_adj,mu_rl).mean(-2))
                sigma_rl = self.LN(self.W_align_sigma_r[i](real_adj,sigma_rl).mean(-2))
                h_rl = (mu_rl+sigma_rl)/2
                mu_rl = h_rl
                sigma_rl = h_rl
            
            mu_r = self.W_a_reg_mu[-1](mu_r)
            mu_p = self.W_align_mu_p[-1](pseudo_adj,h_p).mean(-2)
            mu_rl = self.W_align_mu_r[-1](real_adj,h_rl).mean(-2)
            sigma_r = self.W_a_reg_sigma[-1](sigma_r)
            sigma_p = self.W_align_sigma_p[-1](pseudo_adj,h_p).mean(-2)
            sigma_rl = self.W_align_sigma_r[-1](real_adj,h_rl).mean(-2)
            mu = (mu_r+mu_p+mu_rl)/3
            sigma = (sigma_r+sigma_p+sigma_rl)/3

            #mu = (mu_r+mu_p+0*mu_rl)/2
            #sigma = (sigma_r+sigma_p+0*sigma_rl)/2
            return mu,sigma
        
        if(mode == 'align'):
            h = x.permute(1,2,0,3)[:,:,:,:3]  #(T,N,B,F) -> (N,B,T,F)
            mu = h[:,:,:,0]
            sigma = h[:,:,:,1]
            
            return mu,sigma
                     
        # calculate convolution for current timestep
        if(mode == 'train'):
            h = x.permute(1,2,0,3)[:,:,:,-3:]
            mu = h[:,:,:,0]
            sigma = h[:,:,:,1]
            
            mu_r = mu
            mu_rl = mu
            mu_p = mu
            sigma_r = sigma
            sigma_rl = sigma
            sigma_p = sigma
            for i in range(self.n_align_layers-1):
                mu_r = self.LN(self.W_c_reg_mu[i](mu_r))
                sigma_r = self.LN(self.W_c_reg_sigma[i](sigma_r))               
            for i in range(self.n_align_layers-1):  
                mu_p = self.LN(self.W_conv_mu_p[i](pseudo_adj,mu_p).mean(-2))
                sigma_p = self.LN(self.W_conv_sigma_p[i](pseudo_adj,sigma_p).mean(-2))
                h_p = (mu_p+sigma_p)/2
                mu_p = h_p
                sigma_p = h_p
            for i in range(self.n_align_layers-1):  
                mu_rl = self.LN(self.W_conv_mu_r[i](real_adj,mu_rl).mean(-2))
                sigma_rl = self.LN(self.W_conv_sigma_r[i](real_adj,sigma_rl).mean(-2))
                h_rl = (mu_rl+sigma_rl)/2
                mu_rl = h_rl
                sigma_rl = h_rl
            
            mu_r = self.W_c_reg_mu[-1](mu_r)
            mu_p = self.W_align_mu_p[-1](pseudo_adj,h_p).mean(-2)
            mu_rl = self.W_align_mu_r[-1](real_adj,h_rl).mean(-2)
            sigma_r = self.W_c_reg_sigma[-1](sigma_r)
            sigma_p = self.W_align_sigma_p[-1](pseudo_adj,h_p).mean(-2)
            sigma_rl = self.W_align_sigma_r[-1](real_adj,h_rl).mean(-2)
            mu = (0*mu_r+mu_p+mu_rl)/2
            sigma = (0*sigma_r+sigma_p+sigma_rl)/2
            return mu,sigma
        
def bhattacharyya_distance(pm,ps,qm,qs):
    mask = (pm>-1)
    if mask.sum()==0:
        return None
    qm = qm[mask]
    pm = pm[mask]
    qs = qs[mask]
    ps = ps[mask]
    qs=torch.abs(qs)+0.01
    ps=torch.abs(ps)+0.01
    sqr = (qs*qs+ps*ps)
    return ((pm-qm)*(pm-qm)/4*sqr + torch.log((2+(ps*ps)/(qs*qs)+(qs*qs)/(ps*ps))/4)/4).mean()

def aleatoric_uncertainty(pm,qm,qs):
    mask = (pm>-1)
    if mask.sum()==0:
        return None
    pm = pm[mask]
    qm = qm[mask]
    qs = qs[mask]
    log_var = qs*qs
    l1 = (torch.exp(-log_var)*(pm-qm)*(pm-qm)).mean()
    l2 = log_var.mean()
    return (l1+l2)/2

def rmse_eval(pm,qm):
    mask = (pm>-1)#*(qm>-1)
    if mask.sum()==0:
        return None
    pm = pm[mask]
    qm = qm[mask]
    return torch.sqrt(((pm-qm)*(pm-qm)).mean())

def rmse(pm,qm):
    mask = (pm>-1)
    if mask.sum()==0:
        return None
    pm = pm[mask]
    qm = qm[mask]
    return torch.sqrt(((pm-qm)*(pm-qm)).mean())
        

def tsa(pm,qm,qs,m=3):
    print(pm.shape,qm.shape,qs.shape)
    qs = np.abs(qs)
    low = pm-m*qs
    high = pm+m*qs
    return ((qm>low)*(qm<high)).mean()
        
    
from scipy import sparse
max_epochs = 500
model = Net(10,10,3,2,4).to(device)
                     
rawadj += np.eye(rawadj.shape[0])
rawadj[rawadj>0]=1
rawadj = np.multiply(rawadj>0,rawadj<500).astype(int)
pseudo_rawadj = np.multiply(didi_all_mask,rawadj) #test
real_rawadj = np.multiply(td_train_mask,rawadj)     
print(real_rawadj.shape)
rawadj = sparse.csr_matrix(rawadj)                     
rawg = dgl.from_scipy(rawadj).to(device)
pseudo_rawadj = sparse.csr_matrix(rawadj)                     
pseudo_rawg = dgl.from_scipy(rawadj).to(device)
real_rawadj = sparse.csr_matrix(rawadj)                     
real_rawg = dgl.from_scipy(rawadj).to(device)
                     
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

didi_all_mask = np.array(didi_all_mask).astype(bool)
td_train_mask = np.array(td_train_mask).astype(bool)
td_val_mask = np.array(td_val_mask).astype(bool)
td_test_mask = np.array(td_test_mask).astype(bool)

val_data = data[:,td_val_mask,:]
tst_data = data[:,td_test_mask,:]
data[:,td_val_mask,3:]=-1
data[:,td_test_mask,3:]=-1
data[:,td_val_mask,4:]=-1
data[:,td_test_mask,4:]=-1


td_train_mask = torch.Tensor(td_train_mask).to(device)
td_val_mask = torch.Tensor(td_val_mask).to(device)
td_test_mask = torch.Tensor(td_test_mask).to(device)
td_tst_mask = td_test_mask

cnt = 0
tr_t_list = []
feat1_list = []
feat2_list = []
real_graph_list = []
pseudo_graph_list = []
label_list = []
val_label_list = []
tst_label_list = []
tst_score_list = []
batch_cnt = 0
val_loss_tracker = []

if(False):
    checkpoint = torch.load('model/saved_epoch_23.model')
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1


print('===== training start =====')

for i in range(start_epoch,max_epochs):
    print('===== epoch ',i,' start =====')
    val_loss = []
    tr_a_loss = []
    tr_b_loss = []
    tr_rmse_loss = []
    test_res_list = []
    test_true_list = []
    val_res_list = []
    val_true_list = []
    val_ss_list = []
    for t in range(start_ts+int(data.shape[0]/2),int(data.shape[0])):
        tr_t = list(map(lambda x:x+t,sample_range))
        feature = data[tr_t,:,:]
        is_valid = (feature[-1,:,3].mean()>-1)
        if not is_valid:
            continue
        batch_cnt+=1
        
        # Extract label and feature
        label_list.append(data[t,:,:])
        val_label_list.append(val_data[t,:,:])
        tst_label_list.append(tst_data[t,:,:])

        #label = torch.Tensor(data[t,:,:]).to(device)
        tr_t_list.append(tr_t)

        feat1_list.append(feature)
        # mask out nodes with no pseudo labels -- learning to align pseudo labels and real labels
        valid_p_arr = (feature[:,:,0].sum(0) != (-feature.shape[0])).reshape(-1,1)        
        mask = valid_p_arr*valid_p_arr.transpose()
        # remove edges where src or dst have no pseudo-labels - so default values (-1) won't interfere with the alignment
        pseudo_adj = pseudo_rawadj.multiply(mask)
        real_adj = real_rawadj.multiply(mask)
        pseudo_graph_list.append(pseudo_adj) #v1
        real_graph_list.append(real_adj) #v1
#        pseudo_graph_list.append(pseudo_adj.to(device))
#        real_graph_list.append(real_adj.to(device))

        optimizer.zero_grad()
        # get alignment result
        if batch_cnt%batch_size==0:
            batch_cnt = 0
            
            p_b_graph = sum(pseudo_graph_list)                                   #>0
            p_b_graph[p_b_graph>(batch_cnt/2)]=1
            p_g = dgl.add_self_loop(dgl.from_scipy(p_b_graph)).to(device)

            r_b_graph = sum(real_graph_list)                                   #>0
            r_b_graph[r_b_graph>(batch_cnt/2)]=1
            r_g = dgl.from_scipy(r_b_graph).to(device)
            
            b_label = np.stack(label_list,1)
            b_val_label = np.stack(val_label_list,1)
            b_tst_label = np.stack(tst_label_list,1)
            b_feat = np.stack(feat1_list,2)
            
            label = torch.Tensor(b_label).to(device)                    #(N,B,F)
            v_label = torch.Tensor(b_val_label).to(device)
            t_label = torch.Tensor(b_tst_label).to(device)
            
            feature = torch.Tensor(b_feat).to(device)                   #(T,N,B,F)

            #print(i,'align',(feature[:,:,:,0]==-1).cpu().numpy().mean())
            b_aligned_mu, b_aligned_sigma = model(feature,'align',pseudo_adj=p_g,real_adj=r_g) #aligned (N,B,T)
            #print('align',feature.sum(),b_aligned_mu.sum())
            #optimize on R points
            r_mask = (feature[:,:,:,-1]!=-1).permute(1,2,0).to(int)          #(N,B,T), all entries with R are selected; alternative - select RP
            R_mean_label = feature[:,:,:,3].permute(1,2,0)*r_mask   
            R_sigma_label = feature[:,:,:,4].permute(1,2,0)*r_mask

            R_mean_align = b_aligned_mu*r_mask
            R_sigma_align = b_aligned_sigma*r_mask
            # 1). loss  2). mask the loss with P nodes 3). BP and optimize
            bloss = bhattacharyya_distance(R_mean_label,R_sigma_label,R_mean_align,R_sigma_align)

            #bloss.backward()
            #optimizer.step()    
            tr_b_loss.append(bloss.cpu().detach().numpy())

            
            # save alignment result as real labels
            p_mask = (feature[:,:,:,2]!=-1).permute(1,2,0).to(int)                                 # select all P entries
            p_not_r = (p_mask - r_mask)                                                        # P entry but not R entry
            p_not_r[p_not_r<0]=0
            p_not_r = p_not_r.permute(2,1,0).cpu().detach().numpy().astype(bool)
        
            b_masked_aligned_mu = b_aligned_mu.permute(2,1,0).cpu().detach().numpy()            # values in P entry but not R entry
            b_masked_aligned_sigma = b_aligned_sigma.permute(2,1,0).cpu().detach().numpy()

            
            for tr_t_idx in range(len(tr_t_list)):
                masked_aligned_mu = b_masked_aligned_mu[:,tr_t_idx,:][p_not_r[:,tr_t_idx,:]]             # values in P entry but not R entry
                masked_aligned_sigma = b_masked_aligned_sigma[:,tr_t_idx,:][p_not_r[:,tr_t_idx,:]]
                d_tmp = data[tr_t_list[tr_t_idx],:,3]

                d_tmp[p_not_r[:,tr_t_idx,:]] = masked_aligned_mu                                   # Assign values to selected entries
                data[tr_t_list[tr_t_idx],:,3] = d_tmp

                d_tmp = data[tr_t_list[tr_t_idx],:,4]
                d_tmp[p_not_r[:,tr_t_idx,:]] = masked_aligned_sigma                                     # to corresponding R entries (mean,std)
                data[tr_t_list[tr_t_idx],:,4] = d_tmp

                # interpolate U nodes using neighbor aggregation (G-rawadj) weighted by variance 
                # alternative: move this to tensor, so that it has gradients
                U_mask = ((data[tr_t_list[tr_t_idx],:,2]+data[tr_t_list[tr_t_idx],:,5])==-2)                                      # entries without R or P labels
                RP_mean = data[tr_t_list[tr_t_idx],:,3]                                                            # all mean values in R entries
                RP_sigma = 1/(data[tr_t_list[tr_t_idx],:,4]+0.05)                                                  # all std values in R entries
                RP_mask = ((data[tr_t_list[tr_t_idx],:,2]+data[tr_t_list[tr_t_idx],:,5])!=-2)                                     # entries with either R or P labels
                RP = RP_mean*RP_sigma*RP_mask                                                       # weight mean by sigma, and mask
                sigsum = RP_sigma*rawadj                                                            # sum of weights
                U_mean = (RP*rawadj/sigsum)                                                  # aggregated neighbors # bug
                U_sigma = (1/np.sqrt(np.array(((1/RP_sigma)*(1/RP_sigma)*rawadj)/rawadj.sum(0))))    # sqrt(mean_neighbor(sig^2)) * u_mask

                d_tmp = data[tr_t_list[tr_t_idx],:,3]
                d_tmp[U_mask] = U_mean[U_mask]
                data[tr_t_list[tr_t_idx],:,3] = d_tmp

                d_tmp = data[tr_t_list[tr_t_idx],:,4]
                d_tmp[U_mask] = U_sigma[U_mask] 
                data[tr_t_list[tr_t_idx],:,4] = d_tmp
                # put into graph convolution
                                
                feat2_list.append(data[tr_t_list[tr_t_idx],:,:])
                
            b_feat = np.stack(feat2_list,2)
            feature = torch.Tensor(b_feat).to(device)
            optimizer.zero_grad()
            
           # print(i,'train',(feature[:,:,:,3]).cpu().numpy().mean())
            out_mu, out_sigma = model(feature,'train',pseudo_rawg,real_rawg) #aligned
            #print('out',feature.sum(),out_mu.sum())
       
            # 1). loss = ?? 2). mask the loss with train nodes 3). BP and optimize
            train_out_mu = out_mu[td_train_mask.to(bool),:,0]
            train_out_sigma = out_sigma[td_train_mask.to(bool),:,0]
            train_label_mu = label[td_train_mask.to(bool),:,3]
            aloss = aleatoric_uncertainty(train_label_mu,train_out_mu,train_out_sigma)
            tr_a_loss.append(aloss.cpu().detach().numpy())
            aloss.backward()
            optimizer.step()

            rmse_loss = rmse(train_label_mu,train_out_mu)
            #rmse_loss.backward()
            #optimizer.step()
            if rmse_loss is not None:
                tr_rmse_loss.append(rmse_loss.cpu().detach().numpy())
                
            val_out_mu = out_mu[td_val_mask.to(bool),:,0]
            val_out_sigma = out_sigma[td_val_mask.to(bool),:,0]
            val_label_mu = v_label[:,:,3]

            #rmse_loss = rmse(val_label_mu,val_out_mu)
            #if rmse_loss is not None:
            #    val_loss.append(rmse_loss.cpu().detach().numpy())
           
            val_res_list.append(val_out_mu)
            val_true_list.append(val_label_mu)
            val_ss_list.append(val_out_sigma)
            
            #print('val',val_label_mu.sum(),val_out_mu.sum())

            tst_out_mu = out_mu[td_tst_mask.to(bool),:,0]
            tst_out_sigma = out_sigma[td_tst_mask.to(bool),:,0]
            tst_label_mu = t_label[:,:,3]
            test_res_list.append(tst_out_mu)
            test_true_list.append(tst_label_mu)
#            print('tst',tst_label_mu.sum(),tst_out_mu.sum())

            label_list = []
            val_label_list = []
            tst_label_list = []
            real_graph_list = []
            pseudo_graph_list = []
            feat1_list = []
            tr_t_list = []
            feat2_list = []
    
    val_P = torch.cat(val_res_list,dim=0)
    val_T = torch.cat(val_true_list,dim=0)
    val_S = torch.cat(val_ss_list,dim=0)
    val_rmse = rmse(val_T,val_P).cpu().detach().numpy()
    
    PP = val_P.cpu().detach().numpy()
    TT = val_T.cpu().detach().numpy()
    SS = val_S.cpu().detach().numpy()
    
    #np.save('model/PP_'+str(i),PP)
    #np.save('model/TT_'+str(i),TT)
    #np.save('model/SS_'+str(i),SS)
    
    print('Epoch ',i,':','val-',val_rmse,'train-',np.mean(tr_b_loss),np.mean(tr_a_loss),np.mean(tr_rmse_loss))
    val_loss_tracker.append(val_rmse)
    sl = (SS*SS).mean()
    acc = tsa(TT,PP,SS,m=3)
    print(sl,acc)

    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i}
    torch.save(state, 'model/saved_epoch_'+str(i)+'.model')
    if(np.argmin(val_loss_tracker)+patience<len(val_loss_tracker)): # go test
        tst_P = torch.cat(test_res_list,dim=0)
        tst_T = torch.cat(test_true_list,dim=0)
        tst_rmse = rmse_eval(tst_T,tst_P).cpu().detach().numpy()
        print('test result rmse',tst_rmse)
        break
        
    