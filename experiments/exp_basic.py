import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer,\
    MKAT_iTransformer, \
    MKAT_Transformer, MKAT_Transformer_OnlyEncoder, \
    MKAT_Informer,MKAT_Flashformer,MKAT_Flowformer, MKAT_Reformer,MKAT_iTransformer_only_modular_kan,MKAT_iTransformer_only_group_kan,\
    PatchTST,FEDformer,Autoformer,TimesNet,DLinear,\
    TimeMixer,MKAT_TimeMixer,KAN_iTransformer,TimeKAN,\
    MKAT_iTransformer_not_shared, MKAT_Transformer_OnlyEncoder_not_shared


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'MKAT_iTransformer': MKAT_iTransformer,
            'MKAT_Transformer':MKAT_Transformer,
            'MKAT_Transformer_OnlyEncoder':MKAT_Transformer_OnlyEncoder,
            'MKAT_Informer':MKAT_Informer,
            'MKAT_Flashformer':MKAT_Flashformer,
            'MKAT_Flowformer':MKAT_Flowformer,
            'MKAT_Reformer':MKAT_Reformer,
            'MKAT_iTransformer_only_modular_kan' :MKAT_iTransformer_only_modular_kan,
            'MKAT_iTransformer_only_group_kan':MKAT_iTransformer_only_group_kan,
            'TimeMixer':TimeMixer,
            'MKAT_TimeMixer':MKAT_TimeMixer,
            'KAN_iTransformer': KAN_iTransformer,
            'TimeKAN': TimeKAN,
            'MKAT_iTransformer_not_shared' :MKAT_iTransformer_not_shared,
            'MKAT_Transformer_OnlyEncoder_not_shared':MKAT_Transformer_OnlyEncoder_not_shared,
            'PatchTST': PatchTST,
            'FEDformer':FEDformer,
            'Autoformer':Autoformer,
            'TimesNet':TimesNet,
            'DLinear':DLinear,    
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
