#
# GSML.py
#
# Getting Started in Machine Learning
#
# This file contains functions used by notebooks for the book
#
# Getting Started in Machine Learning: Easy Recipes for Python 3, 
# Scikit-Learn, Jupyter (2019) by Bella Romeo, Sherwood Forest Books, 
# Los Angeles, CA, USA, ISBN-13: 978-0-9966860-6-8
# (c) Copyright 2019.
#
# This software is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import colorsys
def get_colors(n):
    """Returns list of n evenly spaced colors Hue"""
    return [[colorsys.hsv_to_rgb(x/(n+1),1,1)] for x in range(n)]

import numpy as np
def ShowCategories(ax,X,Y,classifier,
                   nsteps=100,alpha=.1,
                   levels="Auto",fit=False,markers="Auto",colors="Auto"):
    """ShowCategories(ax,X,Y,classifier,nsteps=100,alpha=.1,levels="Auto")
    ax = plot axis to draw plot on
    X = two dimensional (two column) array containing features
    Y = class labels (same length as X)
    classifier = a fit classifier from sklearn
    fit =  True, the model has already been fit. If False, needs to be fit.
    nsteps = number of steps in each direction to calculate fit at. The plot 
    performs an exhaustive evaluation, nsteps x nsteps, so the default will calculate
    10,000 classifications. A higher value of nsteps will plot a smoother class
    boundaries at the expense of more classification runs.
    alpha = alpha value to use for background colors
    levels to use for contours = should use "Auto" if class labels are 0, 1, 2, ....
    otherwise this should be a list that distinguishes between the classes.
    
    Return value is the modified plot axis
    """
    
    # define preferred order of markers 

    if markers=="Auto":    
    	marker_styles=["o","s","v","d","+","x","*","<",">","p","h","1","3","2","4"]
    else:
    	marker_styles=markers	

    nmarker_styles=len(marker_styles)
    def get_marker(n):
        style=int(n) if n < nmarker_styles else 0
        return marker_styles[style]
    
    
    if not(fit):
	    classifier.fit(X,Y)
		
    # plot data points
    
    x=X.T[0]
    y=X.T[1]
    classvals=np.unique(Y)
    nclasses=len(classvals)
    
    # draws different marker for each class, up to a point
    
    if nclasses>nmarker_styles:
        print("Warning: maximum of",nmarker_styles,"marker styles will be used")
        
    # contour levels
    # default assumes that class labels are 0, 1, 2, ...
    # draws contours at 0.5, 1.5, 2.5, ....
    
    if levels=="Auto":
        levels=np.arange(0.5,nclasses,1)
    
    if colors=="Auto":
        clrs=get_colors(nclasses)
    else:
        clrs=colors
    for classval in classvals:
        xpts,ypts=(X[Y==classval]).T
        ax.scatter(xpts,ypts,c=clrs[int(classval)],marker=get_marker(classval))
    
    # generate mesh of xy points that nsteps x nsteps in size
    
    xmesh, ymesh = np.meshgrid(
        np.linspace(x.min(), x.max(), nsteps),
        np.linspace(y.min(), y.max(), nsteps))
    
    # unwind the array into a single long one-dimensional array

    unwound_mesh = np.c_[
        xmesh.ravel(), 
        ymesh.ravel()]
    
    # apply the classifier to every element in the unwound mesh
    # each elemen of the unwound mesh is the coordinates of a point in the plane
    
    unwound_prediction=classifier.predict(unwound_mesh)
    
    # unwound_predictions is a sequence of classifications for each point in the plane
    # reshape this to have the same shape as the original mesh
    
    prediction=unwound_prediction.reshape(xmesh.shape)
    
    # generate a contour plot (color filled)
    
    ax.contourf(xmesh,ymesh,prediction,
               alpha=alpha)
    
    # draw thick black lines between the contour levels
    
    ax.contour(xmesh,ymesh,prediction,alpha=1,levels=levels,linewidths=2,colors="k")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    
    return(ax)
#
# Visualize a neural network
#
from graphviz import Digraph
# see https://graphviz.readthedocs.io
from os.path import isfile
def uniqueFileName(filename, type=""):    
    nameparts = filename.rsplit(".",1)
    if len(nameparts)<2:
        nameparts.append(type)
    left, right = nameparts
    i = 1
    fname = left+"."+right
    while isfile(fname):
        fname = left+str(i)+"."+right
        i += 1
    return(fname)

def VisualizeNN(hidden, inputs, outputs, file="NNVisualization"):
    """VisualizeNN(hidden, inputs, outputs, filename)
	
	hiden = tuple of hidden layers (sequence of integers, e.g., (3,4,3)
	
	inputs = list like labels for input layer
	
	outputs = list like labels for output layer
	
	file = optional graphviz file name
    """
    layers=[len(inputs),*hidden,len(outputs)]
    dot=Digraph()
    i=1
    def nodename(i,j):
        return "L["+str(i)+"]["+str(j)+"]"
    for layer in layers:  
        for j in range(1,layer+1):
            if i==1:
                thelabel=inputs[j-1]
                theshape="ellipse"
            elif i==len(layers):
                thelabel=outputs[j-1]
                theshape="ellipse"
            else:
                thelabel=" "
                theshape="circle"
            dot.node(nodename(i,j),label=thelabel,shape=theshape)
            if i>1:
                thisnode=nodename(i,j)
                ilowerlayer=i-1
                ilowerlayers=layers[ilowerlayer-1]
                for k in range(1,ilowerlayers+1):
                    dot.edge(nodename(ilowerlayer,k),thisnode)#,         
        i+=1
    outputfile=uniqueFileName(file,type="gv")
    dot.render(outputfile, view=True)
