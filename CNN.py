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
from torchvision.transforms import ToTensor
from sklearn.model_selection import KFold   ## Sklearn kfold 



import utils                                ## Compute feature map sizes



# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

## Download training dataset
training_data = datasets.FashionMNIST(
    root="data",         
    train=True,          
    download=True,       
    transform=ToTensor()  
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


torch.manual_seed(40)           #Fix torch seed for reproducibility





## CNN network
class CNN(nn.Module):
    def __init__(self,conv_kernel_size_1,conv_kernel_size_2,pool_kernel_size):
        super().__init__()

        ## Hyperparameters 
        out_channels_1,conv_kernel_size_1,padding_1,conv_stride_1=32,conv_kernel_size_1,1,1
        out_channels_2,conv_kernel_size_2,padding_2,conv_stride_2=64,conv_kernel_size_2,1,1
        pool_kernel_size_1,pool_stride_1=pool_kernel_size,2
        pool_kernel_size_2,pool_stride_2=pool_kernel_size,2
        n_hidden_1,n_hidden_2,dropout_prob=256,128,0.25
        

        #Feature size after first CNN+Pooling layer
        feature_size_after_conv_1=utils.conv_size(28,padding_1,conv_kernel_size_1,conv_stride_1)
        feature_size_after_pooling_1=utils.pooling_size(feature_size_after_conv_1,pool_kernel_size_1,pool_stride_1)

        #Feature size after second CNN+Pooling layer
        feature_size_after_conv_2=utils.conv_size(feature_size_after_pooling_1,padding_2,conv_kernel_size_2,conv_stride_2)
        feature_size_after_pooling_2=utils.pooling_size(feature_size_after_conv_2,pool_kernel_size_2,pool_stride_2)
        final_feature_map_size=feature_size_after_pooling_2



        ## First convolutional layer:
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels= 1,              
                out_channels= out_channels_1,            
                kernel_size= conv_kernel_size_1,                                 
                padding= padding_1,                  
            ),
            nn.BatchNorm2d(out_channels_1),               
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=pool_kernel_size_1, stride=pool_stride_1),    
        )

        ## Second convolutional layer:
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=out_channels_1,              
                out_channels=out_channels_2,            
                kernel_size=conv_kernel_size_2,                               
                padding=padding_2
            ),    
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=pool_kernel_size_1, stride=pool_stride_2),                
        )

        #Flatten of the output
        self.flatten = nn.Flatten()

        #Fully-connected perceptron
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(int(final_feature_map_size*final_feature_map_size*out_channels_2), int(n_hidden_1))    ,
            nn.Dropout(dropout_prob), 
            nn.ReLU(),
            
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.Dropout(dropout_prob), 
            nn.ReLU(),

            nn.Linear(n_hidden_2, 10),
            nn.ReLU() 
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)     
        x = self.linear_relu_stack(x)
        return x




loss_fun = nn.CrossEntropyLoss()                                                            


dataset = torch.utils.data.ConcatDataset([training_data, test_data])        #Merge datasets



num_epochs=75
batch_size=500
k_fold=5

splits=KFold(n_splits=k_fold,shuffle=True,random_state=40)



def train_loop(dataloader,model,loss_fun,optimizer,verbose_each=32):
  size= len(dataloader.dataset)     #samples number
  num_batches = len(dataloader)     #number of batches per epoch
  sum_train_loss = 0

  model.train()  
  model=model.to(device) 

  for batch, (X,y) in enumerate(dataloader):
    X=X.to(device)
    y=y.to(device)
    pred=model(X)
    loss=loss_fun(pred,y)

    #Backpropagation 
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

    size=len(dataloader.dataset)           #samples number
    num_batches = len(dataloader)          #number of batches per epoch
    sum_test_loss = 0

    model.eval()                           
    model=model.to(device)                  
   


    with torch.inference_mode():                #similar to torch.no_grad() 
      for X,y in dataloader:       
        X=X.to(device)
        y=y.to(device)

        pred=model(X)
        loss=loss_fun(pred,y)

        num_batches += 1
        avrg_loss += loss_fun(pred,y).item()    #average loss value of the batch

        num_samples += y.size(0)
        frac_correct += (pred.argmax(1)==y).type(torch.float).sum().item() 

        #save test errors
        loss_batch = loss_fun(pred,y).item()
        sum_test_loss += loss_batch 

   

    avrg_loss    /= num_batches              
    frac_correct /= num_samples             

    print(f"Test Error: Avg loss: {avrg_loss:>8f} \n")
    return avrg_loss,frac_correct






conv_kernel_size_1_vec=[3,]
conv_kernel_size_2_vec=[3,]
pool_kernel_size_vec=[2,]


data=[]
data_cross=[]
sum_val=0,0
fold_val_losses=[]
model_path="CNN_model_params"

#Main loop
for kernel_size_1 in conv_kernel_size_1_vec:
    for kernel_size_2 in conv_kernel_size_2_vec:
        for pooling_kernel_size in pool_kernel_size_vec:
         
            sum_val=0
            fold_val_losses=[]

            for fold, (train_idx,val_idx) in enumerate(splits.split(dataset) ):
        

                train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
                test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
                #Make DataLoaders 
                train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
                test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

                
                model = CNN(kernel_size_1,kernel_size_2,pooling_kernel_size)
                model.to(device)


                optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,eps=1e-08,weight_decay=0.01,amsgrad=False)

                ##### BEGIN OF LOOP OVER EPOCHS #####
                for epoch in range(num_epochs):
                    if epoch%10==0:
                        print(f'{fold+1}. Epoca: ', epoch)

                    train_loop(train_loader,model,loss_fun,optimizer)
                    train_loss,train_fraction_correctas=test_loop(train_loader,model,loss_fun)
                    validation_loss, validation_fraccion_correctas=test_loop(test_loader,model,loss_fun)
                    data.append(np.round([fold+1,epoch,train_loss,train_fraction_correctas,validation_loss,validation_fraccion_correctas],6))
                ##### END OF LOOP OVER EPOCHS #####

                #Save model parameters
                torch.save(model.state_dict(), f'./save_models/{model_path}_{fold}_{kernel_size_1}_{kernel_size_2}_{pooling_kernel_size}.pth')

                fold_val_losses.append(np.round(validation_loss,6))

           

            sum_val=sum(fold_val_losses)

            fold_val_loss=sum_val/k_fold
            data_cross.append([fold_val_loss])


        


columnns_names=['k','epoch','train loss','train accuracy','validation loss', 'validation accuracy']

CNN_df_path="CNN_df"
df=pd.DataFrame(data,columns=columnns_names)
df.to_pickle(f'./save_dataframes/{CNN_df_path}.pkl')


CNN_df_path_cv_errors="CNN_df"
df_cross_erros=pd.DataFrame(data_cross,columns=['Final model validation loss'])
df_cross_erros.to_pickle(f'./save_dataframes/{CNN_df_path_cv_errors}.pkl')









def plot_img(data, idx):
    """
    Plot an example
    """
    figure = plt.figure(figsize=(4, 4))
    img, label = data[idx]
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def show_prediction(model,example):
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
