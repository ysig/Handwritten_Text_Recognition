import os
import argparse
import time

# Alphabet for OCR
# alphabet = [' ', '!', '"', '&', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#               ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
#               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
#               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '~', '°',
#               'é', '§', '$', '+', '%', "'", '©', '|', '\\', '#', '@', '£', '€', '®']

# '_' is the blank character for CTC

PP = os.path.dirname(__file__)
class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.time = time.strftime("%m-%d_%H:%M:%S", time.localtime())

    def initialize(self, parser):
        # DATA AND PREPROCESSING PARAMETERS
        parser.add_argument('--dataset', type=str, default='IAM',
                            help="Which dataset to train and test on. Supported values are 'IAM' and 'ICFHR2014'.")
        parser.add_argument('--tr_data_path', type=str,
                            default='/home/ysig/data/',
                            help="Path to folder containing training datasets")
        parser.add_argument('--data_path', type=str,
                            default='/home/ysig/data/',
                            help="Path to folder containing datasets for prediction")
        parser.add_argument('--model_path', type=str,
                            default=os.path.join(PP, 'trained_networks', 'ICFHR2014_model_imgH64.pth'),
                            help="Path to file of saved trained model")
        parser.add_argument('--imgH', type=int, default=64)
        parser.add_argument('--imgW', type=int, default=800)
        parser.add_argument('--alphabet', type=str,
                          default="""_!?#&|\()[]<>*+,-.'"€$£$§=/⊥0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzéèêâàù """)
        # Alphabet for IAM :
        # """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
        # Alphabet for ICFHR2014:
        # """_!?#&|\()[]<>*+,-.'"€$£$§=/⊥0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzéèêâàù """
        # Alphabet for synlines:
        # """_%~`@!?#&|\()[]<>*+,-.'"€$£$§=/⊥0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzéèêâàù """
        parser.add_argument('--data_aug', type=bool, default=True)
        parser.add_argument('--keep_ratio', type=bool, default=True)
        parser.add_argument('--enhance_contrast', type=bool, default=False,
                            help='Enhance contrast of input images or not. Recommended for ICFHR2014')
        # PARAMETERS FOR LOADING/SAVING NETWORKS AND OPTIMIZER STATES
        parser.add_argument('--train', type=bool, default=True, help='Train a network or not')
        parser.add_argument('--weights_init', type=bool, default=True)
        parser.add_argument('--pretrained', type=str, default='')
        parser.add_argument('--save', type=bool, default=False, help='Whether to save the trained network')
        parser.add_argument('--save_model_path', type=str, default='./',
                            help="Where to save the network after training")
        # PARAMETERS FOR PLOT
        parser.add_argument('--previous_epochs', type=int, default=0)
        # TRAINING PARAMETERS
        parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA or not')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=400, help='Training epoch number')
        parser.add_argument('--milestones', type=list, default=[100, 250], help='Milestones(epochs) to change lr')
        # OPTIMIZER PARAMETERS
        parser.add_argument('--optimizer', type=str, default='rmsprop',
                            help="Which optimizer to use. Supported values are 'rmsprop', 'adam', 'adadelta', and 'sgd'.")
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam.')
        parser.add_argument('--rho', type=float, default=0.9, help='rho for ADADELTA')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 penalty) ')

        # FEATURE EXTRACTOR PARAMETERS
        parser.add_argument('--feat_extractor', type=str, default='custom_resnet',
                            help="Which feature extractor to use. Supported values are 'resnet18', 'custom_resnet' and 'conv'.")
        # Parameters for 'conv' structure
        parser.add_argument('--N_CONV_LAYERS', type=int, default=7)  # 7
        parser.add_argument('--NC', type=int, default=1, help='Number of channels given as an input of CRNN')
        parser.add_argument('--N_CONV_OUT', type=list,
                            default=[64, 128, 256, 256, 512, 512, 512])  # [16, 32, 64, 128] #
                            # default=[16, 32, 48, 64, 80]                   )
        parser.add_argument('--CONV', type=dict, default={
            'kernel': [3, 3, 3, 3, 3, 3, 1],  # [3,3,3,3], #
            #'kernel': [3, 3, 3, 3, 3, 3, 3],
            'stride': [(2, 1), (2, 1), (2, 1), 1, 1, 1, 1],  # [1,1,1,1], #
            'padding': [1, 1, 1, 1, 1, 1, 0]  # [1,1,1,1] #
        })
        # Batch normalization
        parser.add_argument('--BATCH_NORM', type=list,
                            #default=[False, False, True, False, True, False, True]  # [True, True, True, True] #
                            default=[True, True, True, True, True])
        # Maxpooling
        parser.add_argument('--MAX_POOL', type=dict, default={
            'kernel': [2, 2, 0, (2, 2), 0, (2, 2), 0],  # [2,2,2,4], #
            #'kernel': [2, 2, 2, 0, 0],
            'stride': [2, 2, 0, (2, 1), 0, (2, 1), 0],  # [2,2,2,4], #
            #'stride': [2, 2, 0, (2, 1), 0],
            'padding': [0, 0, 0, (0, 1), 0, (0, 1), 0]  # [0,0,0,0] #
        })
        # RECURRENT NETWORK PARAMETERS
        parser.add_argument('--N_REC_LAYERS', type=int, default=1, help='Number of recurrent layers in the network.')
        parser.add_argument('--N_REC_INPUT', type=int, default=512, help='Number of channels of the input given to the recurrent network. Must be equal to the number of channels outputed by the feature extractor.')
        parser.add_argument('--N_HIDDEN', type=int, default=256, help='Number of hidden layers in the recurrent cells')
        parser.add_argument('--BIDIRECTIONAL', type=bool, default=True, help='Use bidirectional LSTM or not')
        parser.add_argument('--DROPOUT', type=float, default=0.0, help='Dropout parameter within [0,1] in BLSTM')

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '---------------------Options------------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------------End---------------------\n'
        print(message)

        opt_file = os.path.join(opt.__getattribute__('log_dir'), 'params.txt')
        with open(opt_file, 'w') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parser(self, verbose=False):
        # Get attributes
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        # Add dictionnaries to switch between predictions and labels
        cdict = {c: i for i, c in enumerate(opt.__getattribute__('alphabet'))}  # character -> int
        icdict = {i: c for i, c in enumerate(opt.__getattribute__('alphabet'))}  # int -> character
        opt.__setattr__('cdict', cdict)
        opt.__setattr__('icdict', icdict)
        # Add location to save trained network
        log_dir = opt.__getattribute__('save_model_path') + self.time
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        opt.__setattr__('log_dir', log_dir)
        # print('opt', opt)
        # Update and print
        self.parser = parser
        self.opt = opt
        # if verbose:
        #     self.print_options(self.opt)

        return self.opt
