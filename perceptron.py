import numpy as np                          
from tqdm.notebook import tqdm              ## Progress bars
import pickle                               
import matplotlib.pyplot as plt    
import pandas as pd
import sys

import torch                                ## Neural Network framework 
from torch import nn                        ## Neural network classes
from torch.nn.functional import softmax     ## Softmax function
from torch.utils.data import DataLoader     ## Dataloader


import torch.utils.data                     ## Utils
from torchvision import datasets            ## Datasets (required for download FashionMNIST)
import torchvision.transforms               ## Transforms 
from torchvision.transforms import ToTensor ## Convierte una imagen en un tensor de PyTorch, normalizado entre [0,1]
from sklearn.model_selection import KFold   ## Sklearn kfold 



import utils                                ## Compute feature map sizes

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


## Download training dataset
training_data = datasets.FashionMNIST(
    root="data",          ## Carpeta de descarga
    train=True,           ## Elegimos el conjunto de entrenamiento
    download=True,        ## Pedimos que lo descargue
    transform=ToTensor()  ## Lo transformamos en un "tensor" normalizado entre 0 y 1
)

## Download test dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


## Labels
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}




torch.manual_seed(40)                       #Fix torch seed for reproducibility 





### Multilayer Perceptron class
class MultiLayerPerceptron(nn.Module):
    def __init__(self,n_hidden_1,dropout_prob):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, n_hidden_1),
            nn.Dropout(self.dropout_prob),
            nn.ReLU(),
            
            nn.Linear(n_hidden_1,10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x






loss_fun = nn.CrossEntropyLoss()                                                               
dataset = torch.utils.data.ConcatDataset([training_data, test_data]) #Merge both datasets



num_epochs=50
batch_size=500
k_fold=5

splits=KFold(n_splits=k_fold,shuffle=True,random_state=40)          
                                                                   


def train_loop(dataloader,model,loss_fun,optimizer,verbose_each=32):
  size= len(dataloader.dataset)      #samples number 
  num_batches = len(dataloader)      #number of batches per epoch
  sum_train_loss = 0

  model.train()  
  model=model.to(device)

  for batch, (X,y) in enumerate(dataloader):
    X=X.to(device)
    y=y.to(device)
    pred=model(X)
    loss=loss_fun(pred,y)

    #backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_batch= loss.item()          
    sum_train_loss=sum_train_loss+loss_batch

    #Print progress every 100 batchs
    if batch % verbose_each*len(X) == 0:
        loss   = loss.item()  
        sample = batch*len(X) 
        print(f"batch={batch} loss={loss:>7f}  muestras-procesadas:[{sample:>5d}/{size:>5d}]")   
  avg_train_loss = sum_train_loss/num_batches
  return avg_train_loss

def test_loop(dataloader,model,loss_fun):

    num_samples  = 0
    num_batches  = 0
    avrg_loss    = 0
    frac_correct = 0

    size=len(dataloader.dataset)            #samples number 
    num_batches = len(dataloader)           #number of batches per epoch
    sum_test_loss = 0

    model.eval()                           
    model=model.to(device)                 
   


    with torch.inference_mode():            #similar to torch.no_grad() 
      for X,y in dataloader:      
        X=X.to(device)
        y=y.to(device)

        pred=model(X)
        loss=loss_fun(pred,y)

        num_batches += 1
        avrg_loss += loss_fun(pred,y).item()     #average loss value of the batch

        num_samples += y.size(0)
        frac_correct += (pred.argmax(1)==y).type(torch.float).sum().item() 

        #save test errors
        loss_batch = loss_fun(pred,y).item()
        sum_test_loss += loss_batch 

   

    avrg_loss    /= num_batches              
    frac_correct /= num_samples             

    print(f"Test Error: Avg loss: {avrg_loss:>8f} \n")
    return avrg_loss,frac_correct






data=[]
data_cross=[]
n_hidden_vec=[2048,1024,512,256]
dropout_prob_vec=[0.2,0.4]
alpha=0.01
sum_val=0,0
fold_val_losses=[]
model_path="model_perceptron"

#Main loop
for n in n_hidden_vec:
    for dropout_prob in dropout_prob_vec:
        
        sum_val=0
        fold_val_losses=[]

        for fold, (train_idx,val_idx) in enumerate(splits.split(dataset) ):
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
             
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

           
            model = MultiLayerPerceptron(n,dropout_prob)
            model.to(device)

            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001,eps=1e-08,weight_decay=alpha,amsgrad=False)

            ##### BEGIN OF LOOP OVER EPOCHS #####
            for epoch in range(num_epochs):
                #Print epoch
                if epoch%5==0:
                    print(f'{n} {dropout_prob} {fold+1}. Epoca: ', epoch)

                #Append errors of each epoch
                train_loop(train_loader,model,loss_fun,optimizer)
                train_loss,train_fraction_correctas=test_loop(train_loader,model,loss_fun)
                validation_loss, validation_fraccion_correctas=test_loop(test_loader,model,loss_fun)
                #Append info to data array for the dataframe
                data.append(np.round([n,dropout_prob,fold+1,epoch,train_loss,train_fraction_correctas,validation_loss,validation_fraccion_correctas],6))
            ##### END OF LOOP OVER EPOCHS #####

            #Save model parameters
            torch.save(model.state_dict(), f'./save_models/{model_path}_{fold}_{n}_{dropout_prob}.pth')

            fold_val_losses.append(validation_loss)

            ##### END OF LOOP OVER EPOCHS #####

        sum_val=sum(fold_val_losses)

        fold_val_loss=sum_val/k_fold
        data_cross.append([n,dropout_prob,fold_val_loss])


    


columnns_names=['n','dropout_prob','k','epoch','train loss','train accuracy','validation loss', 'validation accuracy']



perceptron_df_path="perceptron_df"
df=pd.DataFrame(data,columns=columnns_names)
df.to_pickle(f'./save_dataframes/{perceptron_df_path}.pkl')


perceptron_cv_errors_path="perceptron_cv_dataframe"
df_cross_erros=pd.DataFrame(data_cross,columns=['n','dropout_prob','Final model validation loss'])
df_cross_erros.to_pickle(f'./save_dataframes/{perceptron_cv_errors_path}.pkl')







def plot_img(data, idx):
    """
    Plot an example
    """
    figure = plt.figure(figsize=(4, 4))
    img, label = data[idx]
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def show_prediction(example):
    """
    Show prediction for an example
    """
    model.eval()
    x, y = test_data[example][0], test_data[example][1]
    with torch.no_grad():
        pred = model(x.unsqueeze(1).to(device))
    sorted = pred.sort()
    values = softmax(sorted.values[0], dim=-1)
    indices = sorted.indices[0]
    print(
        f'Correct label: {labels_map[y]}', 
        end='\n----------------\n'
    )
    print('Label     Probability')
    for v, idx in list(zip(values,indices))[::-1]:
        label_pred = labels_map[idx.item()]
        print(f'{label_pred:13}{v.item():.5f}')