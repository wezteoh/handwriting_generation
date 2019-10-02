import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

# import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

# import model and utilities
from model import LSTMRandWriter, LSTMSynthesis
from utilz import decay_learning_rate, save_checkpoint

# find gpu
cuda = torch.cuda.is_available()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='rand_write',
                        help='"rand_write" or "synthesis"')
    parser.add_argument('--cell_size', type=int, default=400,
                        help='size of LSTM hidden state')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--timesteps', type=int, default=800,
                        help='LSTM sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--learning_rate', type=float, default=8E-4,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='lr decay rate for adam optimizer per epoch')
    parser.add_argument('--num_clusters', type=int, default=20,
                        help='number of gaussian mixture clusters for stroke prediction')
    parser.add_argument('--K', type=int, default=10,
                        help='number of attention clusters on text input')
    args = parser.parse_args()
    
    # prepare training data
    train_data = [np.load('data/train_strokes_800.npy'), np.load('data/train_masks_800.npy'), np.load('data/train_onehot_800.npy'),
                np.load('data/train_text_lens.npy')]
    for _ in range(len(train_data)):
        train_data[_] =torch.from_numpy(train_data[_]).type(torch.FloatTensor)
        if cuda:
            train_data[_] = train_data[_].cuda()
    train_data = [(train_data[0][i], train_data[1][i], 
                train_data[2][i], train_data[3][i]) for i in range(len(train_data[0]))] 
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
    # prepare validation data
    validation_data = [np.load('data/validation_strokes_800.npy'), np.load('data/validation_masks_800.npy'), 
                    np.load('data/validation_onehot_800.npy'), np.load('data/validation_text_lens.npy')]
    for _ in range(len(validation_data)):
        validation_data[_] = torch.from_numpy(validation_data[_]).type(torch.FloatTensor)
        if cuda:
            validation_data[_] = validation_data[_].cuda()
    validation_data = [(validation_data[0][i], validation_data[1][i], validation_data[2][i], validation_data[3][i]) 
                    for i in range(len(validation_data[0]))] 
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    
    # training
    if args.task == 'rand_write':
        rand_write_train(args, train_loader, validation_loader)
        
    if args.task == 'synthesis':
        synthesis_train(args, train_loader, validation_loader)
        

def rand_write_train(args, train_loader, validation_loader):
    # define model and optimizer
    model = LSTMRandWriter(args.cell_size, args.num_clusters)
    if cuda:
        model = model.cuda()
    
    optimizer = optim.Adam([{'params':model.parameters()},], lr=args.learning_rate)
    
    # initialize null hidden states and memory states
    init_states = [torch.zeros((1,args.batch_size,args.cell_size))]*4
    if cuda:
        init_states  = [state.cuda() for state in init_states]
    init_states  = [Variable(state, requires_grad = False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init = init_states

    t_loss = []
    v_loss = []
    best_validation_loss = 1E10

    # update training time
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        train_loss = 0
        for batch_idx, (data, masks, onehots, text_lens) in enumerate(train_loader):
            
            # gather training batch
            step_back = data.narrow(1,0,args.timesteps)
            x = Variable(step_back, requires_grad=False)
            masks = Variable(masks, requires_grad=False)
            masks = masks.narrow(1,0,args.timesteps)
            
            optimizer.zero_grad()
            # feed forward
            outputs = model(x, (h1_init, c1_init), (h2_init, c2_init))
            end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho , prev, prev2 = outputs
            
            # supervision
            data = data.narrow(1,1,args.timesteps)
            y = Variable(data, requires_grad=False)
            loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)/torch.sum(masks)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
        
        # update training performance
        print('====> Epoch: {} Average train loss: {:.4f}'.format(\
            epoch+1, train_loss/(len(train_loader.dataset)//args.batch_size)))
        t_loss.append(train_loss/(len(train_loader.dataset)//args.batch_size))
        
        # validation
        # prepare validation sample data
        (validation_samples, masks, onehots, text_lens) = list(enumerate(validation_loader))[0][1]
        step_back2 = validation_samples.narrow(1,0,args.timesteps)
        masks = Variable(masks, requires_grad=False)
        masks = masks.narrow(1,0,args.timesteps)
        
        x = Variable(step_back2, requires_grad=False)
        
        validation_samples = validation_samples.narrow(1,1,args.timesteps)
        y = Variable(validation_samples, requires_grad = False)
        
        outputs = model(y, (h1_init, c1_init), (h2_init, c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho , prev, prev2 = outputs 
        loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)/torch.sum(masks)
        validation_loss = loss.item()
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(\
            epoch+1, validation_loss))
        v_loss.append(validation_loss)
    
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(epoch, model, validation_loss, optimizer, args.model_dir, args.task + '_best.pt')
        
        # # learning rate annealing
        # if (epoch+1)%10 == 0:
        #     optimizer = decay_learning_rate(optimizer)
        
        # checkpoint model and training
        filename = args.task + '_epoch_{}.pt'.format(epoch+1)
        save_checkpoint(epoch, model, validation_loss, optimizer, args.model_dir, filename)
        
        print('wall time: {}s'.format(time.time()-start_time))
        
    f1 = plt.figure(1)
    plt.plot(range(1, args.num_epochs+1), t_loss, color='blue', linestyle='solid')
    plt.plot(range(1, args.num_epochs+1), v_loss, color='red', linestyle='solid')
    f1.savefig(args.task +"_loss_curves", bbox_inches='tight')
    
    
def synthesis_train(args, train_loader, validation_loader):
    # infer padded text len and vocab len
    padded_text_len, vocab_len = train_loader.dataset[0][2].size()
    
    # define model and optimizer
    model = LSTMSynthesis(padded_text_len, vocab_len, args.cell_size, args.num_clusters, args.K)
    if cuda:
        model = model.cuda()
    
    optimizer = optim.Adam([{'params':model.parameters()},], lr=args.learning_rate)
    
    # initialize null hidden, memory states and cluster centers
    h1_init = c1_init = torch.zeros((args.batch_size, args.cell_size))
    h2_init = c2_init = torch.zeros((1, args.batch_size, args.cell_size))
    kappa_old = torch.zeros(args.batch_size, args.K)
    
    if cuda:
        h1_init, c1_init = h1_init.cuda(), c1_init.cuda()
        h2_init, c2_init = h2_init.cuda(), c2_init.cuda()
        kappa_old = kappa_old.cuda()
        
    h1_init, c1_init = Variable(h1_init, requires_grad=False), Variable(c1_init, requires_grad=False)
    h2_init, c2_init = Variable(h2_init, requires_grad=False), Variable(c2_init, requires_grad=False)
    kappa_old = Variable(kappa_old, requires_grad=False)
    
    t_loss = []
    v_loss = []
    best_validation_loss = 1E10
    
    # training
    start_time = time.time()
    for epoch in range(args.num_epochs):
        train_loss =0
        for batch_idx, (data, masks, onehots, text_lens) in enumerate(train_loader):
            
            # gather training batch
            step_back = data.narrow(1,0,args.timesteps)
            x = Variable(step_back, requires_grad=False)
            onehots = Variable(onehots, requires_grad = False)
            masks = Variable(masks, requires_grad=False)
            masks = masks.narrow(1,0,args.timesteps)
            text_lens = Variable(text_lens, requires_grad=False)
            
            # focus window weight on first text char
            w_old = onehots.narrow(1,0,1).squeeze()
            
            optimizer.zero_grad()
            
            # feed forward
            outputs = model(x,onehots, text_lens, w_old, kappa_old, (h1_init, c1_init), (h2_init, c2_init))
            end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, prev, prev2, old_phi = outputs
            data = data.narrow(1,1,args.timesteps)
            y = Variable(data, requires_grad=False)
            loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)/torch.sum(masks)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
    
        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch+1, train_loss/(len(train_loader.dataset)//args.batch_size)))
        t_loss.append(train_loss/(len(train_loader.dataset)//args.batch_size))
    
        # validation
        # prepare validation data
        (validation_samples, masks, onehots, text_lens) = list(enumerate(validation_loader))[0][1]
        step_back = validation_samples.narrow(1,0,args.timesteps)
        masks = Variable(masks, requires_grad=False)
        masks = masks.narrow(1,0,args.timesteps)
        onehots = Variable(onehots, requires_grad=False)
        text_lens = Variable(text_lens, requires_grad=False)
        
        w_old = onehots.narrow(1,0,1).squeeze()
        x = Variable(step_back, requires_grad=False)
        
        validation_samples = validation_samples.narrow(1,1,args.timesteps)
        y = Variable(validation_samples, requires_grad = False)
    
        outputs = model(x, onehots, text_lens, w_old, kappa_old, (h1_init, c1_init), (h2_init, c2_init))
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w, kappa, prev, prev2, old_phi = outputs
        loss = -log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, y, masks)/torch.sum(masks)
        validation_loss = loss.item()
        print('====> Epoch: {} Average validation loss: {:.4f}'.format(\
            epoch+1, validation_loss))
        v_loss.append(validation_loss)
    
        
        # # learning rate annealing
        # if (epoch+1)%10 == 0:
        #     optimizer = decay_learning_rate(optimizer)
        
        # checkpoint model and training
        filename = args.task + '_epoch_{}.pt'.format(epoch+1)
        save_checkpoint(epoch, model, validation_loss, optimizer, args.model_dir, filename)
        
        print('wall time: {}s'.format(time.time()-start_time))
        
    f1 = plt.figure(1)
    plt.plot(range(1, args.num_epochs+1), t_loss, color='blue', linestyle='solid')
    plt.plot(range(1, args.num_epochs+1), v_loss, color='red', linestyle='solid')
    f1.savefig(args.task +"_loss_curves", bbox_inches='tight')
    
# training objective
def log_likelihood(end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, \
                    y, masks):
    # targets
    y_0 = y.narrow(-1,0,1)
    y_1 = y.narrow(-1,1,1)
    y_2 = y.narrow(-1,2,1)
    
    # end of stroke prediction
    end_loglik = (y_0*end + (1-y_0)*(1-end)).log().squeeze()
    
    # new stroke point prediction
    const = 1E-20 # to prevent numerical error
    pi_term = torch.Tensor([2*np.pi])
    if cuda:
        pi_term = pi_term.cuda()
    pi_term = -Variable(pi_term, requires_grad = False).log()
    
    z = (y_1 - mu_1)**2/(log_sigma_1.exp()**2)\
        + ((y_2 - mu_2)**2/(log_sigma_2.exp()**2)) \
        - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 + log_sigma_2).exp())
    mog_lik1 =  pi_term -log_sigma_1 - log_sigma_2 - 0.5*((1-rho**2).log())
    mog_lik2 = z/(2*(1-rho**2))
    mog_loglik = ((weights.log() + (mog_lik1 - mog_lik2)).exp().sum(dim=-1)+const).log()
    
    return (end_loglik*masks).sum() + ((mog_loglik)*masks).sum()

if __name__ == '__main__':
    main()
