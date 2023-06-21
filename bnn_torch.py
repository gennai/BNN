import numpy as np
import torch as T
import torchbnn as bnn
import pandas as pd
import os
import sys ; sys.path.append(os.getcwd())
from FCSetup_v41 import FEATURES, SETUP

from matplotlib import pyplot as plt



device = T.device("cpu")
#device = T.device("cuda:0")

class hhDataSet(T.utils.data.Dataset):
  def __init__(self, num_rows=None):
    # like 5.0, 3.5, 1.3, 0.3, 0
    file_hh = "/gwpool/users/gennai/mtt-fit/output_SKIM2017_ggF_BSM_m800_trainOdd_v41.h5"
    df_hh = pd.read_hdf(file_hh)[:num_rows]
    df_hh["sample_class1"] = 1.
    file_DY = "/gwpool/users/gennai/mtt-fit/output_SKIM2017_DY_amc_incl_trainOdd_v41.h5"
    #df_DY = pd.read_hdf(file_DY)[:num_rows]
    #df_DY["sample_class3"] = 1.
    file_TTsem = "/gwpool/users/gennai/mtt-fit/output_SKIM2017_TTsem_trainOdd_v41.h5"
    df_TTsem = pd.read_hdf(file_TTsem)[:num_rows]
    df_TTsem["sample_class2"] = 1.  
    #df=pd.concat([df_hh,df_DY,df_TTsem], ignore_index=True)
    df=pd.concat([df_hh,df_TTsem], ignore_index=True)
    df = df.sample(frac=1, random_state=2023)

    target     = SETUP['target'     ] if 'target'     in SETUP.keys() else 'target'
    assert target not in FEATURES , "The target variable is in the feature list"
    x_train = df[FEATURES]    
    #x_test  = df.loc[df['is_test' ]==1, FEATURES]  
    y_train = df[target]    
    #y_test  = df.loc[df['is_test' ]==1, target  ]  
    
    self.x_data = T.tensor(x_train.values, dtype=T.float32)
    self.y_data = T.tensor(y_train.values, dtype=T.float32)

    
  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if T.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x_data[idx]
    spcs = self.y_data[idx] 
    sample = { 'features' : preds, 'classes' : spcs }
    return sample


class BayesianNet(T.nn.Module):
  def __init__(self):            # 4-100-3
    super(BayesianNet, self).__init__()
    self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=66, out_features=100)
    self.hid2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=100, out_features=100)
    self.hid3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=100, out_features=100)
    self.hid4 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=100, out_features=100)
    self.hid5 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=100, out_features=100)
    self.hid6 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=100, out_features=100)
    self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
      in_features=100, out_features=2)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = T.relu(self.hid3(z))
    z = T.relu(self.hid4(z))
    z = T.relu(self.hid5(z))
    z = T.relu(self.hid6(z))
    z = self.oupt(z)  # no softmax: CrossEntropyLoss() 
    return z
  
def accuracy_quick(model, dataset):
  n = len(dataset)
  X = dataset[0:n]['features']  # all X 
  Y = T.argmax(dataset[0:n]['classes'],dim=1)  # 1-D

  with T.no_grad():
    oupt = model(X)
  arg_maxs = T.argmax(oupt, dim=1)  # collapse cols  
  num_correct = T.sum(Y==arg_maxs)
  acc = (num_correct * 1.0 / len(dataset))
  return acc.item()

def main():
  print("\nBegin Bayesian neural network HH demo ")
  # 0. prepare
  np.random.seed(1)
  T.manual_seed(1)
  np.set_printoptions(precision=7, suppress=True, sign=" ")
  np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
    
  # 1. load training data
  print("\nCreating HH train Dataset and DataLoader ")
  nMaxEvents = 20000
  ds = hhDataSet(nMaxEvents)
  train_size = int(0.8*len(ds))
  print (len(ds))
  test_size = int(0.2*len(ds))
  train_ds, test_ds = T.utils.data.random_split(ds, [train_size, test_size])
  bat_size = 500
  train_ldr = T.utils.data.DataLoader(train_ds, batch_size=bat_size, shuffle=True)

  # 2. create network
  net = BayesianNet().to(device)

  # 3. train model (could put this into a train() function)
  max_epochs = 200 
  #max_epochs = 2
  ep_log_interval = 10
  ce_loss = T.nn.CrossEntropyLoss()   # applies softmax()
  kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
  optimizer = T.optim.Adam(net.parameters(), lr=0.01)

  print("\nbat_size = %3d " % bat_size)
  print("loss = highly customized ")
  print("optimizer = Adam 0.01")
  print("max_epochs = %3d " % max_epochs)

  print("\nStarting training")
  net.train()
  for epoch in range(0, max_epochs):
    epoch_loss = 0  # for one full epoch
    num_lines_read = 0
    ce_tot = 0

    for (batch_idx, batch) in enumerate(train_ldr):
      num_lines_read += 1
      X = batch['features']  # [10,4]
      Y = batch['classes']  # alreay flattened
      optimizer.zero_grad()
      oupt = net(X)

      cel = ce_loss(oupt, Y)
      kll = kl_loss(net)
      beta = 1E-8
      tot_loss = cel + (beta * kll)

      ce_tot += cel.item()
      epoch_loss += tot_loss.item()  # accumulate
      tot_loss.backward()  # update wt distribs
      optimizer.step()

    epoch_loss = epoch_loss/num_lines_read    
    ce_tot = ce_tot/num_lines_read
    if epoch % ep_log_interval == 0:
      print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))
      #print("epoch = %4d   ce_loss = %0.4f" % (epoch, ce_tot))
  
  print("Training done ")

  # 4. evaluate model accuracy
  print("\nComputing Bayesian network model accuracy")
  net.eval()
  acc = accuracy_quick(net, train_ds)  # item-by-item
  print("Accuracy on train data = %0.4f" % acc)

  # 5. make a prediction
  print("\nPredicting species ")


  x = test_ds[:]["features"]
  #for i in test_ds[:]["classes"]:
  #  print(i)
  myAccs = []
  for i in range(500):
    with T.no_grad():
      logits = net(x).to(device)  # values do not sum to 1.0      
    probs = T.softmax(logits, dim=1).to(device)
    #print(probs.numpy())
    acc_test = accuracy_quick(net, test_ds)
    myAccs.append(acc_test)
    print("Accuracy on test data = %0.4f" % acc_test)

  fig = plt.figure(figsize=(10, 7), dpi=100)
  plt.hist(myAccs,bins=50,range=(0.85,1.01))
  plt.show()

if __name__ == "__main__":
  main()