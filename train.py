import find_mxnet
import mxnet as mx
import numpy as np
import argparse
import os, sys
import train_model

SIZE = 12

def get_cnn():

    data = mx.symbol.Variable('data')
     
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(2,2), num_filter=8, stride=(2,1), name='conv1')
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu", name='relu')
         
    # second conv
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(1,1), num_filter=4, name='conv2')
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu", name='relu2')
    ''' 
    # third conv
    conv3 = mx.symbol.Convolution(data=relu1, kernel=(1,1), num_filter=8, name='conv3')
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu", name='relu3')
    '''
    # first fullc
    flatten = mx.symbol.Flatten(data=relu2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=16, name='fc1')
    relu4 = mx.symbol.Activation(data=fc1, act_type="relu", name='relu4')
    '''
    fc2 = mx.symbol.FullyConnected(data=bn1, num_hidden=16, name='fc2')
    relu5 = mx.symbol.Activation(data=fc2, act_type="relu", name='relu5') 
    
    bn2 = mx.symbol.BatchNorm(data=relu2)   
    fc3 = mx.symbol.FullyConnected(data=bn2, num_hidden=32, name='fc3')
    relu3 = mx.symbol.Activation(data=fc3, act_type="relu", name='relu3') 
    
    bn3 = mx.symbol.BatchNorm(data=relu3)
    fc4 = mx.symbol.FullyConnected(data=bn3, num_hidden=32, name='fc4')
    relu4 = mx.symbol.Activation(data=fc4, act_type="relu", name='relu4') 
    
    bn4 = mx.symbol.BatchNorm(data=relu4)
    fc5 = mx.symbol.FullyConnected(data=bn4, num_hidden=64, name='fc5')
    relu5 = mx.symbol.Activation(data=fc5, act_type="relu", name='relu5') 
    '''
    # second fullc
    q = mx.symbol.FullyConnected(data=relu4, num_hidden=5, name='q')
    
    # loss
    cnn = mx.symbol.SoftmaxOutput(data=q, name='softmax')
    return cnn

def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir + 'experience.npz'
        obs = np.load(open(data_dir, 'rb'))['obs'] 
        act = np.load(open(data_dir, 'rb'))['act']
        training_size = obs.shape[0] * 5 / 6
        print training_size
        train = mx.io.NDArrayIter(
            data        = obs[:training_size],
            label       = act[:training_size],
            batch_size  = args.batch_size,
            shuffle     = True)

        val = mx.io.NDArrayIter(
            data        = obs[training_size:],
            label       = act[training_size:],
            batch_size  = args.batch_size,
            shuffle     = True)
        
        return (train, val)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_shape = (1, 8, 2)
    net = get_cnn()

    # train
    train_model.fit(args, net, get_iterator(data_shape))
