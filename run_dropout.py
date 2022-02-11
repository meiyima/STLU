import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import scipy.stats
import torch._VF as _VF
import math
import ustlmonitor as ustl
import confidencelevel

import argparse
from backdoor import lshhash, chash

parser = argparse.ArgumentParser(description='MI')
parser.add_argument('-method', type=str, default='qual')
args = parser.parse_args()

# load data
# airdataset = torch.rand(24001) # 24 hours * 10 days 

# a = torch.load("local1.dat")
a = torch.load("beijing_pm25.dat")
# print(a.shape)
airdataset = a[0]
# amount of data used, part 7552
# airdataset = a[2][0 : 7552]
# airdataset = a[0 : 7552]
# all data
# airdataset = a[2][0 :]
rnnmodel = "LSTM"

# print(torch.size(airdataset))

airdata = torch.cat((airdataset[0 :-1],airdataset[1 :])).view(-1, 2)

# 1 - B dropconnect; 2 - B dropout; 3 - G  dropconnect; 4 - G dropout


## hourly, 1 hour -> 3 hours 

BATCH_SIZE = 128
EPOCH = 70 #30
TIMEUNITES = 12 #20 
N_MC = 100 # dropout, iteration times 
LR = 0.01
DROPOUT_RATE = 0.7 # larger p, less uncertainty;
DROPOUT_TYPE = 2
SEED = 32
HIDDEN_SIZE = 32
torch.manual_seed(SEED)


# past data points
PASTUNITES = 7  # less than TIMEUNITES 12
FUTUREUNITES = TIMEUNITES - PASTUNITES 

# devided by timeunite
# airdata_byunite = torch.cat((airdataset[0 :-1],airdataset[1 :])).view(-1, TIMEUNITES, 2)


# devided by timeunite, repeated per unit
airdata_byunite = torch.zeros(airdata.size(0) - TIMEUNITES +1,TIMEUNITES,2)
# print(airdata_byunite.size())


for i in range(airdata_byunite.size(0)):
    airdata_byunite[i] = airdata[i:i+TIMEUNITES]
    

# print(airdata_byunite)

# Shuffle data
airdata_byunite = airdata_byunite[torch.randperm(airdata_byunite.size(0))]


train_split = 0.9
valid_split = 0.95
train_data = torch.utils.data.TensorDataset(airdata_byunite[:int(train_split * airdata_byunite.size(0))])
valid_data = torch.utils.data.TensorDataset(airdata_byunite[int(train_split * airdata_byunite.size(0)):int(valid_split * airdata_byunite.size(0))])
test_data = torch.utils.data.TensorDataset(airdata_byunite[int(valid_split * airdata_byunite.size(0)):])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

    
class LSTMCellWithMask(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCellWithMask, self).__init__(input_size, hidden_size, bias=True)
        
    def forward_with_mask(self, input, mask, hx=None):
        (mask_ih, mask_hh) = mask
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return _VF.lstm_cell(
            input, hx,
            self.weight_ih * mask_ih, self.weight_hh * mask_hh,
            self.bias_ih, self.bias_hh,
        )
        
    

class LSTMModel(nn.Module):
    # initial function
    def __init__(self, train_dropout_type=1):
        super(LSTMModel, self). __init__()
        self.lstm = LSTMCellWithMask(1, HIDDEN_SIZE)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)
        # self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.train_dropout_type = train_dropout_type


    # forward function
    def forward(self, input):
        outputs = []
        h_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.float, device=device)
        
        # mask = (torch.ones(4*100, 1, dtype=torch.float), torch.ones(4*100, 100, dtype=torch.float))
        if self.train_dropout_type == 1:
            mask1 = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*DROPOUT_RATE)/DROPOUT_RATE  
            mask2 = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float)*DROPOUT_RATE)/DROPOUT_RATE 
        elif self.train_dropout_type == 2:
            para = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*DROPOUT_RATE)/DROPOUT_RATE 
            mask1 = para
            mask2 = para.expand(-1, HIDDEN_SIZE)
        elif self.train_dropout_type == 3:
            p = math.sqrt((1-DROPOUT_RATE)/DROPOUT_RATE)
            mask1 = torch.normal(1, torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float)*p)
        elif self.train_dropout_type == 4:
            p = math.sqrt((1-DROPOUT_RATE)/DROPOUT_RATE)
            para = torch.normal(1, torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*p)
            mask1 = para
            mask2 = para.expand(-1, HIDDEN_SIZE)    
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask  = (mask1.to(device),mask2.to(device))
        # print(mask)
        
        for i in range(input.size(1)):
            h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
            # h_t = F.dropout(h_t, p=DROPOUT_RATE)
            output = self.linear(h_t)
            # output = self.linear(F.relu(self.linear2(h_t)))
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

    def forward_test(self, input, dropout_rate=DROPOUT_RATE, dropout_type=DROPOUT_TYPE):
        outputs = []
        h_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.float, device=device)
        
        input_local = input[:, 0, :]
        
        # mask = (torch.ones(4*100, 1, dtype=torch.float), torch.ones(4*100, 100, dtype=torch.float))
        if dropout_type == 1:
            mask1 = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*dropout_rate)/dropout_rate  
            mask2 = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float)*dropout_rate)/dropout_rate 
        elif dropout_type == 2:
            para = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*dropout_rate)/dropout_rate 
            mask1 = para
            mask2 = para.expand(-1, HIDDEN_SIZE)
        elif dropout_type == 3:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            mask1 = torch.normal(1, torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float)*p)
        elif dropout_type == 4:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            para = torch.normal(1, torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*p)
            mask1 = para
            mask2 = para.expand(-1, HIDDEN_SIZE)    
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask  = (mask1.to(device),mask2.to(device))
        
        
        
        
        for i in range(input.size(1)):
            h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
            
            # h_t, c_t = self.lstm(input_local, (h_t, c_t))
            # h_t = F.dropout(h_t, p=dropout_rate)
            output = self.linear(h_t)
            # output = self.linear(F.relu(self.linear2(h_t)))
            input_local = output
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
        
    
    def forward_test_with_past(self, input, dropout_rate=DROPOUT_RATE, dropout_type=DROPOUT_TYPE):
        outputs = []
        h_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.float, device=device)
        
        input_local = input[:, 0, :] 
        
         # mask = (torch.ones(4*100, 1, dtype=torch.float), torch.ones(4*100, 100, dtype=torch.float))
        if dropout_type == 1:
            mask1 = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*dropout_rate)/dropout_rate  
            mask2 = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float)*dropout_rate)/dropout_rate 
        elif dropout_type == 2:
            para = torch.bernoulli(torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*dropout_rate)/dropout_rate 
            mask1 = para
            mask2 = para.expand(-1, HIDDEN_SIZE)
        elif dropout_type == 3:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            mask1 = torch.normal(1, torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float)*p)
        elif dropout_type == 4:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            para = torch.normal(1, torch.ones(4*HIDDEN_SIZE, 1, dtype=torch.float)*p)
            mask1 = para
            mask2 = para.expand(-1, HIDDEN_SIZE)    
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask  = (mask1.to(device),mask2.to(device))
        
        
        
        
        for i in range(PASTUNITES+1):
            h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
            # h_t, c_t = self.lstm(input_local, (h_t, c_t))
            # h_t, c_t = self.lstm(input[:, i, :], (h_t, c_t))
            # h_t = F.dropout(h_t, p=dropout_rate)
            output = self.linear(h_t)
            # output = self.linear(F.relu(self.linear2(h_t)))
            input_local = output
            # outputs += [output]
        
        outputs.append(input_local)   
        for i in range(FUTUREUNITES-1):
            h_t, c_t = self.lstm.forward_with_mask(input_local, mask, (h_t, c_t))
            # h_t, c_t = self.lstm(input_local, (h_t, c_t))
            # h_t = F.dropout(h_t, p=dropout_rate)
            output = self.linear(h_t)
            # output = self.linear(F.relu(self.linear2(h_t)))
            input_local = output
            outputs.append(output)
            # print(outputs)
        
        # print(outputs)
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


# input [batch_size per batch (128, 1024), time_units, input_dimension]
# bigger batch size, smaller learning rate 



# training
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, datasample in tqdm(enumerate(train_loader)):
        # print(datasample)
        data, target = datasample[0][:, :, 0:1], datasample[0][:, :, 1]
        data, target = data.to(device), target.to(device)
        # print(target)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        # print(batch_idx, loss.item())
        
    train_loss /= len(train_loader)
    print("train loss: %.5f" %train_loss)



# testing 
def test(model, device, criterion, test_loader, epoch):
    # model.eval()
    test_loss = 0
    test_loss_mean = 0
    traceset = []
    with torch.no_grad():
        for datasample in tqdm(test_loader):
            if datasample[0].size(0) != BATCH_SIZE: 
                current_batch_size = datasample[0].size(0)
            else:
                current_batch_size = BATCH_SIZE
            
            data, target = datasample[0][:, :, 0:1], datasample[0][:, :, 1]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # output = torch.zeros(N_MC, current_batch_size, TIMEUNITES)
            output = torch.zeros(N_MC, current_batch_size, FUTUREUNITES)
            
            
            
            for i in range(N_MC):
                # print(i, data)
                # output[i] = model.forward_test(data)
                output[i] = model.forward_test_with_past(data).cpu()
                
                # print(i, output[i])
                # print(i, target[:, PASTUNITES: TIMEUNITES])
                test_loss += criterion(output[i], target[:, PASTUNITES: TIMEUNITES].cpu()).item()
            
            # trace = torch.cat((target[:, PASTUNITES: TIMEUNITES], output.mean(dim = 0), output.std(dim = 0)), dim = 1)
            # if epoch == EPOCH:
            #     trace = torch.stack((output.mean(dim = 0), output.std(dim = 0), target[:, PASTUNITES: TIMEUNITES].cpu(), torch.zeros(output.mean(dim = 0).size())), dim = -1)
            #     traceset.append(trace)
            # print(trace.size())
            # print(target[:, PASTUNITES: TIMEUNITES], output.mean(dim = 0), output.std(dim = 0))
            test_loss_mean += criterion(output.mean(dim = 0), target[:, PASTUNITES: TIMEUNITES].cpu()).item()
        # print(traceset)
        
        # if epoch == EPOCH:    
        #     traceset = torch.cat(traceset, dim = 0)  
        #     print(traceset.numpy().shape)
            
            
        #     # save to .dat file, comment it when test the code
        #     np.save("afterworktest_airsignal_"+str(FUTUREUNITES)+"_"+rnnmodel+"_dropouttype_"+str(DROPOUT_TYPE)+"_dropoutpara_"+str(DROPOUT_RATE)+".dat", traceset.numpy())
            
    
        test_loss /= (len(test_loader)*N_MC)
        test_loss_mean /= (len(test_loader))
        print("test loss: %.5f" %test_loss)
        print("test loss_mean: %.5f" %test_loss_mean)

def getdist(tr_pred, tr_orig, conf=0.95):
    ppf = ustl.get_ppf(conf)
    lower = tr_pred[:, : , 0] - ppf * tr_pred[:, : , 1]
    upper = tr_pred[:, : , 0] + ppf * tr_pred[:, : , 1]
    dist = torch.max(torch.max(lower - tr_orig, tr_orig - upper), torch.zeros(lower.size()))
    # print(lower[0], upper[0], tr_orig[0], dist[0])
    return dist
    
def getconfloss(tr_pred, requirement, flag=True):
    # print(requirement(tr_pred, func='eq'))
    strong = confidencelevel.calculatecf_strong(requirement(tr_pred, func='eq'), 0)
    weak = confidencelevel.calculatecf_weak(requirement(tr_pred, func='eq'), 0)
    if flag:
        if strong[1] > 0:
            return 1 - strong[1]
        else:
            return weak[0] + 1
    else:
        strong_ = [1-weak[1], 1-weak[0]]
        weak_ = [1-strong[1], 1-strong[0]]
        if strong_[1] > 0:
            return 1 - strong_[1]
        else:
            return weak_[0] + 1     

def getconfdistloss(tr_pred, tr_orig):
    cdf = scipy.stats.norm.cdf((torch.abs(tr_orig - tr_pred[:, :, 0]) / tr_pred[:, :, 1]))
    return cdf * 2 - 1
    
def evaluate_model(model, dropout_type, dropout_rate, device, criterion, test_loader, requirement, beta=0.6, method='quan', N_MC=N_MC):
    # model.eval()
    # test_loss = 0
    # test_loss_mean = 0
    eva_loss = 0
    rho_rate = 0
    dist_loss = 0
    dist_rate = 0
    traceset = []
    with torch.no_grad():
        for datasample in test_loader:
            if datasample[0].size(0) != BATCH_SIZE: 
                current_batch_size = datasample[0].size(0)
            else:
                current_batch_size = BATCH_SIZE
            
            data, target = datasample[0][:, :, 0:1], datasample[0][:, :, 1]
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # output = torch.zeros(N_MC, current_batch_size, TIMEUNITES)
            output = torch.zeros(N_MC, current_batch_size, FUTUREUNITES)
            
            
            
            for i in range(N_MC):
                # print(i, data)
                # output[i] = model.forward_test(data)
                output[i] = model.forward_test_with_past(data, dropout_rate=dropout_rate, dropout_type=dropout_type).cpu()
                
                # print(i, output[i])
                # print(i, target[:, PASTUNITES: TIMEUNITES])
                # test_loss += criterion(output[i], target[:, PASTUNITES: TIMEUNITES].cpu()).item()
            
            # trace = torch.cat((target[:, PASTUNITES: TIMEUNITES], output.mean(dim = 0), output.std(dim = 0)), dim = 1)
            # if epoch == EPOCH:
            
            trace = torch.stack((output.mean(dim = 0), output.std(dim = 0), target[:, PASTUNITES: TIMEUNITES].cpu(), torch.zeros(output.mean(dim = 0).size())), dim = -1)
            if method=='conf':
                # dist = getdist(trace[:, :, :-2], trace[:, :, -2]).mean(dim=1)
                # dist_rate += (dist == 0).sum().item()
                dist_rate += (getdist(trace[:, :, :-2], trace[:, :, -2]) == 0).float().mean(dim=1).sum().item()
                rho = 0
                for j in range(trace.size(0)):
                    rho_set = requirement(trace[j, :, :-2])
                    rho_orig = requirement(trace[j, :, -2:])
                    if rho_orig[0] < 0:
                        flag = False
                    else:
                        flag = True
                    if (rho_set[0] >= 0 and rho_orig[0] >= 0) or (rho_set[1] <= 0 and rho_orig[0] <= 0):
                        rho_rate += 1
                    rho += getconfloss(trace[j, :, :-2], requirement, flag)
                dist = torch.tensor(getconfdistloss(trace[:, :, :-2], trace[:, :, -2])).mean(dim=1).sum().item()
                eva_loss += (1 - beta) * dist + beta * rho
                dist_loss += dist
            else:
                dist = getdist(trace[:, :, :-2], trace[:, :, -2]).mean(dim=1)
                # dist_rate += (dist == 0).sum().item()
                dist_rate += (getdist(trace[:, :, :-2], trace[:, :, -2]) == 0).float().mean(dim=1).sum().item()
                rho = 0
                for j in range(trace.size(0)):
                    rho_set = requirement(trace[j, :, :-2])
                    rho_orig = requirement(trace[j, :, -2:])
                    # print(rho_orig)
                    # print(rho_set, rho_orig)
                    # print(rho_set)
                    # print(rho_set * ((rho_orig > 0)-0.5))
                    if rho_orig[0] > 0:
                        rho += -min(rho_set[0], 0)
                    else:
                        rho += -min(-rho_set[1], 0)
                    if (rho_set[0] >= 0 and rho_orig[0] >= 0) or (rho_set[1] <= 0 and rho_orig[0] <= 0):
                        rho_rate += 1
                        
                    # rho += -min((rho_set * ((rho_orig > 0) - 0.5) * 2)[0], 0)
                    
                if method=='quan':
                    eva_loss += ((1- beta) * dist).sum().item() + beta * rho
                    dist_loss += dist.sum().item()
            # inner = requirement(trace)
            # traceset.append(trace)
            # print(trace.size())
            # print(target[:, PASTUNITES: TIMEUNITES], output.mean(dim = 0), output.std(dim = 0))
            # test_loss_mean += criterion(output.mean(dim = 0), target[:, PASTUNITES: TIMEUNITES].cpu()).item()
        # print(traceset)
        
        # if epoch == EPOCH:    
            # traceset = torch.cat(traceset, dim = 0)  
            # print(traceset.numpy().shape)
                
            # save to .dat file, comment it when test the code
            # np.save("afterworktest_airsignal_"+str(FUTUREUNITES)+"_"+rnnmodel+"_dropouttype_"+str(DROPOUT_TYPE)+"_dropoutpara_"+str(DROPOUT_RATE)+".dat", traceset.numpy())
        if method=='qual':
            eva_loss = - ((1 - beta) * dist_rate + beta * rho_rate)
            dist_loss = - dist_rate
        
        eva_loss /= len(test_loader.dataset)
        dist_loss /= len(test_loader.dataset)
        rho_rate /=  len(test_loader.dataset)
        dist_rate /= len(test_loader.dataset)
        # print("eva loss: %.5f" % eva_loss)
        # print("test loss: %.5f" % test_loss)
        # print("test loss_mean: %.5f" % test_loss_mean)
    return eva_loss, dist_loss, rho_rate, dist_rate

model = LSTMModel(DROPOUT_TYPE)
criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)



# run 
for epoch in range(1, EPOCH+1):
    
    print("Epoch: %.0f" %epoch)
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, criterion, test_loader, epoch)
    

DR_RATE_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# DR_RATE_LIST = [0.3, 0.5, 0.7]

def requirement_func_until(signal, conf=0.95, func='monitor'):
    if func=='monitor':
        return ustl.umonitor((("until",(0,4)), ((("neg", 0), (("mu", signal), [60, conf])), (("neg", 0), (("mu", signal), [50, conf])))), 0)
    else:
        return (("until",(0,4)), ((("neg", 0), (("mu", signal), [60, conf])), (("neg", 0), (("mu", signal), 50))))
def requirement_func_always(signal, conf=0.95, func='monitor'):
    if func=='monitor':
        return ustl.umonitor((("always",(0,4)), (("neg", 0), (("mu", signal), [50, conf]))), 0)
    else:
        return (("always",(0,4)), (("neg", 0), (("mu", signal), 50)))

# current
# Select current with valid_data
current_best = 2e9
inside_best = 2e9
# method = 'qual'
method = args.method
requirement_func = requirement_func_always
for dropout_type in range(1, 4 + 1):
    for dropout_rate in DR_RATE_LIST:
        eva_loss, inside_loss, rho_rate, dist_rate = evaluate_model(model, dropout_type, dropout_rate, device, criterion, valid_loader, requirement_func, method=method)
        print(dropout_type, dropout_rate, eva_loss, inside_loss, rho_rate, dist_rate)
        if eva_loss < current_best:
            best_para = (dropout_type, dropout_rate, eva_loss, inside_loss)
            current_best = eva_loss
        if inside_loss < inside_best:
            best_para_inside = (dropout_type, dropout_rate, eva_loss, inside_loss)
            inside_best = inside_loss
            
print('Best set:', best_para)
print('Evaluate:', evaluate_model(model, best_para[0], best_para[1], device, criterion, test_loader, requirement_func, method=method))
print('Best inside:', best_para_inside)
print('Evaluate:', evaluate_model(model, best_para_inside[0], best_para_inside[1], device, criterion, test_loader, requirement_func, method=method))

