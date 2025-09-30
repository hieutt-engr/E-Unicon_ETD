import torch
import os
import time
import argparse

def prepare_fin(config):
    fin = open(config.result_file, 'a')
    fin.write('-------------------------------------\n')
    fin.write(config.model_name + '\n')
    fin.write('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '\n')
    fin.write('Data root dir: ' + config.root_dir + '\n')
    fin.write('d_model: ' + str(config.d_model) + '\t pad_size: ' + str(config.pad_size) + '\t nhead: ' + str(config.nhead) + '\t num_layers: ' + str(config.num_layers) + '\n')
    fin.write('batch_size: ' + str(config.batch_size) + '\t learning rate: ' + str(config.lr) + '\t smooth factor: ' + '\n\n')
    fin.close()
    
def parser_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="data/ksm_transformer_best_result")
    # parser.add_argument('--indir', type=str, default="data/ksm/")
    parser.add_argument('--model', type=str, default="efficientnet")
    parser.add_argument('--log_mode', type=str, default="train")
    parser.add_argument('--window_size', type=int, default=37)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default='cae')
    parser.add_argument('--model_name', type=str, default='ConResnet50_MPNCOV_ETD')
    parser.add_argument('--model_path', type=str, default='save/')
    
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--resume', type=str, default=None, 
                        help='path to the checkpoint to resume from')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # optimization classifier
    parser.add_argument('--epoch_start_classifier', type=int, default=90)
    parser.add_argument('--learning_rate_classifier', type=float, default=0.01,
                        help='learning rate classifier')
    parser.add_argument('--lr_decay_epochs_classifier', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_classifier', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay_classifier', type=float, default=0,
                        help='weight decay_classifier')
    parser.add_argument('--momentum_classifier', type=float, default=0.9,
                        help='momentum_classifier')
    # mixup
    parser.add_argument('--lamda', type=float, default=0.5, 
                        help='universum lambda')
    parser.add_argument('--mix', type=str, default='mixup', 
                        choices=['mixup', 'cutmix'], 
                        help='use mixup or cutmix')
    parser.add_argument('--size', type=int, default=32, 
                        help='parameter for RandomResizedCrop')

    # model dataset
    parser.add_argument('--mean', type=str, 
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, 
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, 
                        help='path to custom dataset')
    parser.add_argument('--n_classes', type=int, default=2, 
                        help='number of class')

    # method
    parser.add_argument('--method', type=str, default='UniCon', 
                        choices=['UniCon', 'SupCon', 'SimCLR'],
                        help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07, 
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    # parser.add_argument('--model_path', type=str, default='model/')
    args,_ = parser.parse_known_args()
    return args

class Config:
    def __init__(self, args):
        self.model_name = args.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch, 'has_mps', False) else 'cpu')

        if args.mode == 'cae':
            self.dout_mess = 20 # 4 weeks
            self.d_model = self.dout_mess
            self.nhead = 20  # ori: 5
            print("The mode using CAE component")
        elif args.mode == 'dnn':
            self.dout_mess = 20 # 4 weeks
            self.d_model = self.dout_mess
            self.nhead = 10  # ori: 5
            print("The mode using DNN component")
        else:
            self.dout_mess = 28
            self.d_model = 28
            self.nhead = 28  # ori: 5
            print("Not using")

        self.model_path = args.model_path
        self.pad_size = args.window_size  
        self.window_size = args.window_size  
        self.max_time_position = 10000
        self.num_layers = 7
        self.log_e = 2
        self.model = args.model
        self.log_mode = args.log_mode
        print("Log mode: ", self.log_mode)
        
        self.mode = args.mode
        self.use_embedding = False
        self.embedding_size = 64
        self.add_norm = False

        self.classes_num = 2
        self.n_classes = args.n_classes
        self.method = args.method
        self.epoch_start_classifier = args.epoch_start_classifier
        self.lamda = args.lamda
        self.mix = args.mix
        self.size = args.size
        self.temp = args.temp
        

        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.lr = args.lr  # 0.0001 learning rate
        self.root_dir = args.indir

        self.model_save_path = args.model_path + self.model_name + '/'
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        self.result_file = 'results/' + args.indir.split('/')[1] + '_' + args.mode + '.txt'

        self.isload_model = False
        # self.model_path = 'model/' + self.model_name + '/' + self.model_name + '_model_' + str(self.start_epoch) + '.pth'