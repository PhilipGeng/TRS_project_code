import numpy as np
import argparse
from model import *
import torch
import torch.nn as nn
import torch.optim as optimizer


parser = argparse.ArgumentParser()

parser.add_argument('-p','--path_prefix',default='../citynet-phase1-data/')
parser.add_argument('-n','--city_name',default='cd')
parser.add_argument('-y','--output_length',default=6)
parser.add_argument('-s','--service',default='speed')
parser.add_argument('-x','--input_lag_mode',default=1)
parser.add_argument('-m','--model',default='GCN')
parser.add_argument('-b','--batch_size',default=16)
parser.add_argument('-l','--learning_rate',default=0.001)
parser.add_argument('-u','--num_layers',default=3)

#load all args
args = parser.parse_args()

pref = args.path_prefix # prefix of file paths ~/citynet-phase1-data
cname = args.city_name  # bj or sh or ...
ylength = args.output_length # temporal length of prediction output
service = args.service #service name -- demand/inflow/... or all
lag_mode = args.input_lag_mode #see below definition of lag
model_name = args.model.upper() # 'HA', 'LR', 'ARIMA', 'CNN', 'CONVLSTM', 'GCN', 'GAT'
batch = int(args.batch_size)
lr = float(args.learning_rate)
nl = int(args.num_layers)

patience = 15

print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def min_max_normalize(data,cut_off_percentile=0.99):
    sl = sorted(data.flatten())
    max_val = sl[int(len(sl)*cut_off_percentile)]
    min_val = sl[int(len(sl)*(1-cut_off_percentile))]
    data[data>max_val]=max_val
    data[data<min_val]=min_val
    data-=min_val
    data/=(max_val-min_val)
    return data,max_val,min_val

def split(data,lags,temporal_mask,portion=[0.7,0.15,0.15]):
    assert(sum(portion)==1.0)
    x = []
    y = []
    cnt=-lags[0]
    for i in range(-lags[0],len(temporal_mask)-ylength):
        x_idx = list(map(lambda x:x+cnt,lags))
        y_idx = [cnt+o for o in range(0,ylength)]
        
        x_idxs = list(map(lambda x:x+i,lags))
        y_idxs = [i+o for o in range(0,ylength)]

        selected = temporal_mask[(x_idxs+y_idxs)]
        if(temporal_mask[i]==1):
            cnt+=1
        if(selected.sum()<len(selected)):
            continue
        if (temporal_mask[x_idx+y_idx]==0).sum() == 0:
            x.append(data[x_idx])
            y.append(data[y_idx])

    x = np.stack(x,0)
    y = np.stack(y,0)
    trainx = np.array(x[:int(portion[0]*len(x))])
    trainy = np.array(y[:int(portion[0]*len(y))])
    valx = np.array(x[int(portion[0]*len(x)):int((portion[0]+portion[1])*len(x))])
    valy = np.array(y[int(portion[0]*len(y)):int((portion[0]+portion[1])*len(y))])
    testx = np.array(x[int((portion[0]+portion[1])*len(x)):])
    testy = np.array(y[int((portion[0]+portion[1])*len(x)):])
    return trainx,trainy,valx,valy,testx,testy

# loss functions
def masked_mae(T,P,mask,preserve_f_dim=False):
    mask = np.expand_dims(mask,(0,-1))
    mask = np.expand_dims(mask,(0))
    mask = np.repeat(np.repeat(np.repeat(mask,T.shape[0],0),T.shape[-1],-1),T.shape[1],1)
    if(preserve_f_dim):
        return (abs(T-P)*mask).sum((0,1,2,3))/(mask.sum()/T.shape[-1])
    else:
        return (abs(T-P)*mask).sum()/mask.sum()

def mae(T,P,preserve_f_dim=False):
    if(preserve_f_dim):
        return (abs(T-P).sum((0,1,2)))/(T.shape[0]*T.shape[1]*T.shape[2])
    else:
        return abs(T-P).mean()

def masked_rmse(T,P,mask,preserve_f_dim=False):
    mask = np.expand_dims(mask,(0,-1))
    mask = np.expand_dims(mask,(0))
    mask = np.repeat(np.repeat(np.repeat(mask,T.shape[0],0),T.shape[-1],-1),T.shape[1],1)
    print(T.shape,P.shape,mask.shape)
    if(preserve_f_dim):
        return np.sqrt((((T-P)**2)*mask).sum((0,1,2,3))/(mask.sum()/T.shape[-1]))
    else:
        return np.sqrt((((T-P)**2)*mask).sum()/mask.sum())

def rmse(T,P,preserve_f_dim=False):
    if(preserve_f_dim):
        return np.sqrt(((T-P)**2).sum((0,1,2))/(T.shape[0]*T.shape[1]*T.shape[2]))
    else:
        return np.sqrt(((T-P)**2).mean())

def mae_pytorch(T,P):
    return torch.abs(T-P).mean()
def rmse_pytorch(T,P):
    return torch.sqrt(((T-P)**2).mean())


print('==========loading data files============')

# all loaded data in (Sample,x/y_length(temporal),lng,lat,feature) format -- by split() function
# for single tasks, feature = 1
# for all tasks, feature = 4 (stacked)
if(service == 'speed'):
    speed_10min = np.load(pref+'traffic_speed/'+cname+'/'+cname+'_speed.npy') #10min
    A = np.load(pref+'traffic_speed/'+cname+'/'+cname+'_adj.npy')
    data = np.expand_dims(speed_10min,-1)
    data[np.isnan(data)]= 0    
    data,data_max,data_min = min_max_normalize(data)
A += np.eye(A.shape[0])
A= torch.Tensor(A).to(device)
A_euc = A
A_poi = A
A_road = A

print('masked data loaded:',service,'-',data.shape,'max',data_max,'min',data_min)
print('adjacency loaded:',service,'-',A.shape)


#history lags, with 30min temporal sampling interval
no_lag = [-1]
hour_lag = list(-i for i in range(12,0,-1))
one_day_lag = [-48,-4,-3,-2,-1]

if(int(lag_mode)==0):
    lag=no_lag
if(int(lag_mode)==1):
    lag=hour_lag
if(int(lag_mode)==2):
    lag=one_day_lag

temporal_mask = np.ones(data.shape[0])
#split train-val-test dataset
train_x,train_y,val_x,val_y,test_x,test_y = split(data,lag,temporal_mask)
print('split data to: \n train_x-%s, train_y-%s, \n val_x-%s, val_y-%s, \n test_x-%s, test_y-%s'%(train_x.shape,train_y.shape,val_x.shape,val_y.shape,test_x.shape,test_y.shape))


if(model_name in ['HA','LR','ARIMA']): 
    if(model_name in ['HA']): #no training phase
        model = HA(out_dim=ylength,mode=1)
        test_pred = model.predict(test_x)
        
    if(model_name in ['LR','ARIMA']): # one-shot training phase
        if(model_name == 'LR'):
            model = LR()
        if(model_name == 'ARIMA'):
            model = ARIMA()        
        model.train(train_x,train_y)
        test_pred = model.predict(test_x)
        
elif(model_name in ['CNN','CONVLSTM','GCN','GAT','GLR']): #iterative training
    train_x = torch.Tensor(train_x).to(device)
    train_y = torch.Tensor(train_y).to(device)
    val_x = torch.Tensor(val_x).to(device)
    val_y = torch.Tensor(val_y).to(device)
    test_x = torch.Tensor(test_x).to(device)
    test_y = torch.Tensor(test_y).to(device)

    #to do: fill this when pytorch env available
    if(model_name=='CNN'):
        pass
    if(model_name=='CONVLSTM'):
        pass
    if(model_name[0]=='G'):
        in_dim = 12
        out_dim = ylength
            
        if(model_name=='GCN'):
            model = GCN(in_dim,out_dim,A_euc,A_poi,A_road,ylength,n_layers=nl).to(device)
        if(model_name=='GAT'):
            model = GAT(in_dim,out_dim,A_euc,A_poi,A_road,ylength,n_layers=nl,device=device).to(device)
        if(model_name=='GLR'):
            model = GLR(in_dim,out_dim,ylength).to(device)
        optimizer = optimizer.Adam(model.parameters(), lr=lr, weight_decay=1e-6)        
            
        val_err = []
        tst_err = []
        preds = []
        epoch = 0
        idx = np.array([i for i in range(0,train_x.size(0))])
        while True:
            epoch += 1
            np.random.shuffle(idx)
            train_x = train_x[idx,:,:,:]
            train_y = train_y[idx,:,:,:]
            model.train()
            for i in range(0,train_x.size(0),int(batch)):
                input_x = train_x[i:i+batch,:,:,:]
                input_y = train_y[i:i+batch,:,:,:]
                optimizer.zero_grad()
                output = model(input_x)
                loss = rmse_pytorch(input_y,output)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()

                val_out_list = []
                for i in range(0,val_x.size(0),int(batch)):
                    end_pos = min(val_x.size(0),i+batch)    
                    optimizer.zero_grad()            
                    val_out_list.append(model(val_x[i:end_pos,:,:,:]))
                val_out = torch.cat(val_out_list,dim=0)

                loss = rmse_pytorch(val_y,val_out)
                val_err.append(loss.item())
                if(epoch%1==0):
                    print('epoch',epoch,' -- val loss:',loss.item())


                test_out_list = []
                for i in range(0,test_x.size(0),int(batch)):
                    end_pos = min(test_x.size(0),i+batch)    
                    optimizer.zero_grad()            
                    test_out_list.append(model(test_x[i:end_pos,:,:,:]))
                test_out = torch.cat(test_out_list,dim=0)

                tst_out = test_out.cpu().detach().numpy()
                preds.append(tst_out)
                if(len(preds)>patience):
                    preds = preds[1:]

                #early stopping criterion
                if(np.argmin(val_err)==len(val_err)-patience):
                    test_pred = preds[0]
                    test_y = test_y.cpu().detach().numpy()
                    break

mae = mae(test_y,test_pred)
rmse = rmse(test_y,test_pred)
print('test score: rmse = %.4f, mae = %.4f'%(rmse*(data_max-data_min),mae*(data_max-data_min)))
