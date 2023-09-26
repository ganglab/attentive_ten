import torch
from torch.utils.data import Dataset
import numpy as np
import random
import networkx as nx
import itertools as its
import collections as cls
import torch.nn.functional as F
import scale_free_graph as sf
import os

HOME = os.environ['HOME']


class Custom_dataset(Dataset):

    def __init__(self, dynamic='HR', name=None,   stage='train', noise_std=0.2,
                 seed=12, num=10., past=3, future=1, dyn_dim=1,  purpose = 'ATEn' ):

        # print( stage )

        node_timeseries = np.load(
            HOME + f'/cause/data/real/result/{dynamic}/{name}.npy').astype( np.float32 )
        # HOME + '/cause/data/{0}/nematode/nodes_timeseries_coupling={1}_excited.npy'.format(dynamic, gc))
        # print(length)

        node_timeseries = node_timeseries[ :, :, :dyn_dim ]  # [n,length,channels]
        length = node_timeseries.shape[1]

        np.random.seed(seed)

        noise_std = np.mean(np.abs(node_timeseries), axis=(0, 1)) * noise_std  # [channels] # mean * density
        for c, std in enumerate(noise_std):
            noise = np.random.normal( loc=0, scale=std, size=node_timeseries[:, :, c].shape )
            node_timeseries[:, :, c] += noise

        # past, future = 3, 1
        node_timeseries_past = []
        node_timeseries_future = []
        for nt in node_timeseries:  # [lenght,channels]
            temp, temp2 = [], []
            for i in range(past, length - future + 1):  # i从3开始
                temp.append(nt[i - past:i, :].reshape(-1))
                temp2.append(nt[i:i + future, :].reshape(-1))  # i是future的首位index
            node_timeseries_past.append(temp)
            node_timeseries_future.append(temp2)
        self.node_timeseries_past = np.array(node_timeseries_past)  # [ nodes, length-past-future+1,past*ch ]
        self.node_timeseries_future = np.array(node_timeseries_future)  # [ nodes, length-past-future+1,future*ch]
        self.node_timeseries = node_timeseries[:, past:length - future + 1, :]  # [ nodes, length-past-future+1 ,ch]
        self.length = length - past - future + 1
        print(self.node_timeseries.shape, self.length, )

        test_num = int(1000 / 2)
        if name == 'cat':
            # fname = 'mixed.species_brain_1.graphml'
            fname = 'cat.graphml'
        if name == 'macaque':
            # fname = 'rhesus_brain_2.graphml'
            fname = 'macaque.graphml'
        if name == 'celegans':
            # fname = 'c.elegans_neural.male_1.graphml'
            fname = 'celegans.graphml'
        if name == 'rat':
            # fname = 'rattus.norvegicus_brain_1.graphml'
            fname = 'rat.graphml'
        if name == 'mouse':
            # fname = 'mouse_visual.cortex_2.graphml'
            fname = 'mouse.graphml'
            test_num = 100#int(180 / 2)
        # if name == 'fly':
        #     fname = 'drosophila_medulla_1.graphml'

        fname = HOME + '/cause/data/real/networks/' + fname

        g = nx.read_graphml(fname)
        mapping = dict(zip(g, range(0, len(g.nodes()))))
        g = nx.relabel_nodes(g, mapping)
        self.adj = nx.to_numpy_array(g)  # adj
        self.adj[self.adj > 1.0] = 1  # remove the weight of edges

        if dynamic == 'FHN':
            # indegree = adj.sum( axis=0 )
            # valid_node = np.where( indegree!=0 )[0]
            m = self.node_timeseries[:, :, 0].max(axis=1)  # 序列最大值，有些入度为1或0的节点的时间序列有问题
            valid_node = np.where(m > 0)[0]
            # valid_node = np.concatenate( [valid_node,valid_node2] )
            g = g.subgraph(valid_node)
        # print(len(g), len(g.edges()))
        edges = set(g.edges())
        no_edges = set(its.permutations(g.nodes(), 2)) - edges
        edges, no_edges = np.array(list(edges)), np.array(list(no_edges))
        np.random.shuffle( edges )
        np.random.shuffle( no_edges )
        # print(edges)

        # np.random.seed( seed )
        num = int(len(edges)*num) if type(num) == float else num

        edges_train = edges[ np.random.choice( len(edges) , num, replace=False) ] #train sampling from observed_area of network


        no_edges_train = no_edges[ np.random.choice( len(no_edges) , int(len(edges_train) ), replace=False) ]
        # no_edges = no_edges[np.random.choice(len(no_edges), 1, replace=False)]

        if (stage == 'train') or (stage == 'train_c') :
            self.pairs = np.concatenate([edges_train, no_edges_train], axis=0)
            self.edges = edges_train
            self.no_edges = no_edges_train
            # pairs = set(its.permutations(g.nodes(), 2)) - {tuple(i) for i in edges} - {tuple(i) for i in no_edges}
            # self.pairs2 = np.array(list(pairs))
            # np.random.seed(seed)
            # self.pairs2 = self.pairs2[np.random.choice(len(self.pairs2), 500, replace=False)][:500, :]  # [N,2]
            if (stage == 'train'):

                sample_length = int(length / 2)
                if dynamic == 'HR':
                    self.sample_length = sample_length
                elif dynamic == 'Rulkov':
                    self.sample_length = sample_length
                elif dynamic == 'Izh':
                    self.sample_length = sample_length
                # elif dynamic == 'Morris':
                #     self.sample_length = sample_length
                elif dynamic == 'FHN':
                    self.sample_length = sample_length

                # self.sample_num = self.edges.shape[0]
                self.sample_num = self.pairs.shape[0]
            else:
                self.sample_length = self.length
                # self.sample_length = sample_length
                self.sample_num = self.pairs.shape[0]

            print( f'train_smaples_shape:{self.pairs.shape},  noise_std:{noise_std}, edges:{edges_train}, '
                   f'no_edges:{no_edges_train}, sample_length:{self.sample_length},'
                   f' max(min)_of_timeseries:{self.node_timeseries.max()}({self.node_timeseries.min()}) ')



        else:  # test or valid

            # have_edges_valid = set(g.edges()) - {tuple(i) for i in edges}
            # havenot_edges_valid = set(its.permutations(g.nodes(), 2)) - set(g.edges()) - {tuple(i) for i in no_edges}
            have_edges_valid = {tuple(i) for i in edges} - {tuple(i) for i in edges_train} #valid sampling from observed_area of network
            havenot_edges_valid = {tuple(i) for i in no_edges} - {tuple(i) for i in no_edges_train}
            have_edges_valid, havenot_edges_valid = np.array(list(have_edges_valid)), np.array(list(havenot_edges_valid))
            have_edges_valid = have_edges_valid[np.random.choice(len(have_edges_valid), 50, replace=False)]
            # print( len(edges), len(no_edges), len(havenot_edges_valid) )
            havenot_edges_valid = havenot_edges_valid[np.random.choice(len(havenot_edges_valid), 50, replace=False)]
            # have_edges_valid = have_edges_valid[np.random.choice(len(have_edges_valid), 80, replace=False)]
            # havenot_edges_valid = havenot_edges_valid[np.random.choice(len(havenot_edges_valid), 80, replace=False)]



            if stage == 'test':

                have_edges_test = set(g.edges()) - {tuple(i) for i in edges_train} - {tuple(i) for i in have_edges_valid } #test sampling from whole network
                havenot_edges_test = set(its.permutations(g.nodes(), 2)) - set(g.edges()) - {tuple(i) for i in no_edges_train} - {tuple(i) for i in havenot_edges_valid}

                have_edges_test, havenot_edges_test = np.array(list(have_edges_test)), np.array(list(havenot_edges_test))

                have_edges_test = have_edges_test[ np.random.choice(len(have_edges_test), test_num, replace=False) ]
                havenot_edges_test = havenot_edges_test[np.random.choice(len(havenot_edges_test), test_num, replace=False)]
                self.pairs = np.concatenate( [have_edges_test, havenot_edges_test], axis=0 )

            else: #valid
                self.pairs = np.concatenate( [have_edges_valid, havenot_edges_valid], axis=0 )


            # self.pairs = np.concatenate([self.pairs, edges, no_edges], axis=0)
            self.sample_length = self.length
            # self.node_timeseries = torch.Tensor( self.node_timeseries )
            self.sample_num = len(self.pairs)
            print( f'{stage}_smaples_shape:{self.pairs.shape}, noise_std:{noise_std}'  )

        # self.past, self.future = 3, 1
        self.stage = stage
        self.purpose = purpose
        np.random.seed()


    def __getitem__( self, index ):

        if self.stage == 'train':
            # index = np.random.choice(len(self.edges))
            # n1, n2 = self.edges[index]
            index = np.random.choice(len(self.pairs))
            n1, n2 = self.pairs[index]
        #
        else:  # test or train_c or valid
            n1, n2 = self.pairs[index]
        # index = np.random.choice( len(self.pairs) )
        # n1, n2 = self.pairs[index]

        start = np.random.choice( self.length - self.sample_length + 1 )
        # print(start)
        sample = np.transpose( self.node_timeseries[[n2, n1], start: start + self.sample_length, :],
                               (2, 0, 1) )  # [2,sample_length,ch] -- [ch,2,sample_len]
        # print( type(sample), sample.shape )
        # if self.stage == 'train2':
        #     if np.random.choice(2) == 1:
        #         sample = sample[:, :, ::-1]
        #         sample = sample.copy()

        # sample = np.expand_dims( self.node_timeseries[[n2, n1], :],  axis=0 )  # [2,lenght]
        # target = self.node_timeseries[ n1, self.past:-self.future ].copy().reshape(-1,1) #,[length-10,1]

        if self.purpose == 'ATEn':

            target_future = self.node_timeseries_future[n2,
                            start: start + self.sample_length].copy()  # ,[length-past-future,future]
            target_past = self.node_timeseries_past[n2, start: start + self.sample_length]
            source_past = self.node_timeseries_past[n1, start: start + self.sample_length]

            joint = np.concatenate([source_past, target_past, target_future], axis=1)
            # [length-10,10],[length-10,10],[length-10,1] -- [length-10,21]
            joint2 = np.concatenate([target_past, target_future], axis=1)

            np.random.shuffle(target_future)  # [length-past-future+1,future]
            # indep = np.concatenate([self.node_timeseries_past[n2], target], axis=1)
            indep = np.concatenate([source_past, target_past, target_future], axis=1)
            # [length-10,10],[length-10,10],[length-10,1]
            indep2 = np.concatenate([target_past, target_future], axis=1)

            return sample, \
                   torch.Tensor(joint), torch.Tensor(indep), \
                   torch.Tensor(joint2), torch.Tensor(indep2),  self.adj[n1, n2]
        #
        else:

            return sample, self.adj[n1, n2]



    def __len__(self):

        if self.stage == 'train':
            return self.sample_num * 10
        else: # ( self.stage == 'test' ) or (self.stage == 'train2') or 'valid'
            return self.sample_num




