

from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
def confusion_matrix_pyplot(y_true,y_pred,num_classes,name=""):
       if isinstance(y_true,torch.Tensor):
              y_true=y_true.clone().detach().cpu().numpy()
       if isinstance(y_pred,torch.Tensor):
              y_pred=y_pred.clone().detach().cpu().numpy()
       if y_true.shape[-1]==num_classes:
              y_true=np.argmax(y_true,1)
       if y_pred.shape[-1]==num_classes:
              y_pred=np.argmax(y_pred,1)
       print(y_true.shape,y_pred.shape)
       array=confusion_matrix(y_true,y_pred)
       sns.set(font_scale=2.0)
       df=pd.DataFrame(array,index=range(num_classes),columns=range(num_classes))
       fig=plt.figure(figsize=(10,10))
       ax=fig.add_subplot(111)
       ax=sns.heatmap(df,square=True,annot=True,ax=ax,cmap="YlGnBu")
       plt.title("---")
       fig=ax.get_figure()
       if name!="":
              fig.savefig(name)
       else:
              fig.savefig("test.png")


class Kernel_VIS(nn.Module):
       def __init__(self):
              super(Kernel_VIS, self).__init__()
              self.flatten=nn.Flatten()
              self.adaptivepool2d=nn.AdaptiveAvgPool2d((1,1))
       @torch.no_grad()
       def tensor_matrix(self,tensor):
              print(tensor.shape)
              if tensor.ndim == 4:
                     sm  = self.flatten(self.adaptivepool2d(tensor))
              else:
                     sm = tensor

              norm = sm.pow(2).sum(1, keepdim=True).sqrt()
              map = sm / norm
              s_k = map@map.T
              return s_k
       def forward(self,tensor_list):
              tensor_list=[self.tensor_matrix(tensor) for tensor in tensor_list]
              tensor_list=[tensor.cpu().numpy() for tensor in tensor_list]
              return tensor_list


