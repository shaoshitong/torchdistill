import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time,datetime
import plotly as py
import chart_studio
import plotly.io as pio
mesh_size = .02
margin = 0.25
pio.templates.default = "simple_white"
pyplt = py.offline.plot
p = pio.renderers['png']
p.width = 600
p.height = 600
# Load and split data

def Matrix_VIS(matrix):
    rang=matrix.shape[0]
    min=np.min(matrix)
    max=np.max(matrix)
    color=(matrix-min)/(max-min+1e-12)
    xrange=np.arange(0,rang,1)
    yrange=np.arange(0,rang,1)
    # xx,yy=np.meshgrid(xrange,yrange)
    # Z=matrix
    Z=(matrix-matrix.min())/(matrix.max()-matrix.min()+1e-6)
    # df=np.concatenate([xx.reshape(-1)[...,None],yy.reshape(-1)[...,None],color.reshape(-1)[...,None]*10],axis=1)
    fig=go.Figure()
    fig.add_trace(
        go.Contour(x=xrange,y=yrange,z=Z,
                      showscale=False,
                      colorscale='RdBu',
                      opacity=1,
                      name='Score',
                     connectgaps=True
                   )
    )
    time1_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    fig.write_image(time1_str+".png")