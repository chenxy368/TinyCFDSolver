# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 02:39:05 2022

@author: HP
"""
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_one_streamlines(u, v, dx, dy, title):
    assert u.shape == v.shape
    
    x= np.linspace(0, dx, u.shape[1])
    y = np.linspace(0, dy, u.shape[0])
    xx, yy = np.meshgrid(x,y)

    plt.streamplot(xx, yy, u, v, color=np.sqrt(u ** 2 + v ** 2), density = 1.5, linewidth = 1.5, cmap = 'jet')
    plt.colorbar(label = 'velocity[m/s]')
    plt.xlabel('x/m', fontsize = 14)
    plt.ylabel('y/m', fontsize = 14)

    plt.title(title, fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.1])
    plt.xlim([0,0.1])
    plt.show()

def plot_one_contourf(array, dx, dy, title, colorbar_label, vmin, vmax):
    X = np.linspace(0, dx, array.shape[1])
    Y = np.linspace(0, dy, array.shape[0])
    norm = colors.Normalize(vmin = vmin, vmax = vmax)
    im = cm.ScalarMappable(norm=norm, cmap='jet')
    plt.contourf(X, Y, array, 20, cmap = 'jet', norm = norm)
    plt.colorbar(im, label = colorbar_label)
    plt.xlabel('x/m', fontsize = 14)
    plt.ylabel('y/m', fontsize = 14)
    plt.title(title)
    plt.show()

def animate(arrays, dx, dy, title, colorbar_label, vmin, vmax):
    norm = colors.Normalize(vmin = vmin, vmax = vmax)
    im = cm.ScalarMappable(norm=norm, cmap='jet')
    X = np.linspace(0, dx, arrays[0].shape[1])
    Y = np.linspace(0, dy, arrays[0].shape[0])
    fig, ax = plt.subplots()
    ax.contourf(X, Y, arrays[0], 20, cmap = 'jet',  norm = norm)
    fig.colorbar(im, ax = ax, label = colorbar_label)
    def frame(i):
        norm = colors.Normalize(vmin = vmin, vmax = vmax)
        ax.cla()
        ax.contourf(X, Y, arrays[i], 20, cmap = 'jet', norm = norm)
        ax.set_xlabel('x/m',fontsize = 14)
        ax.set_ylabel('y/m',fontsize = 14)
        ax.set_title(title)
    
    ani = animation.FuncAnimation(fig, frame, frames = len(arrays), interval = 1)
    ani.save(title + '.gif', fps=10)
    plt.show()