import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.parametrizations import spectral_norm


class estimator( nn.Module ):
    def __init__( self , in_dim, out_dim=1 ):
        super( estimator, self ).__init__()

        self.dense1 = nn.Sequential( nn.Linear(in_features=in_dim, out_features=256, bias=True)  ,
                                    nn.ReLU())
        self.dense2 = nn.Sequential( nn.Linear(in_features=256, out_features=64, bias=True) ,
                                    nn.ReLU() )
        self.dense3 = nn.Sequential(  nn.Linear(in_features=64, out_features=64, bias=True) ,
                                    nn.ReLU() )
        self.dense4 = nn.Sequential(  nn.Linear(in_features=64, out_features=256, bias=True) ,
                                    nn.ReLU())

        self.dense_last =  nn.Linear( in_features=256, out_features= out_dim, bias=True )



    def forward( self, x ):

        x =  self.dense1( x )
        x = self.dense2( x )
        x = self.dense3( x )
        x = self.dense4( x )

        logit = self.dense_last( x ) #[2bs,l,1]
        logit = torch.squeeze( logit, dim=-1 ) #[2bs,l]

        return logit





