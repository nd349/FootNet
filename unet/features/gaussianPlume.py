# -*- coding: utf-8 -*-
# @Author: Alex Turner and Nikhil Dadheech
# @Date:   2022-06-20 19:47:43
# @Last Modified by:   nikhildadheech
# @Last Modified time: 2022-06-27 16:28:13


import os
import numpy as np
import pandas as pd
import imageio
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


singleLocPlotLocation = "../data/plots/plumeSingleLocation/"


class GaussianPlume():
    def __init__(self):
        self.nos_df = self.getUSCoastLine()
        pass

    def get_files(self, directory, extension=""):
        if extension:
            files = [f for f in listdir(directory) if f[-len(extension):] == extension]
        else:
            files = [f for f in listdir(directoy)]
        files = [directory+file for file in files]
        return files

    def getUSCoastLine(self, fileLocation="/home/disk/p/nd349/nikhil.dadheech/pointSources/data/NOS80k.csv"):
        ''' Return US coastline data '''
        # https://www.dropbox.com/s/2ko30nyprpd8dmt/NOS80k.csv
        nos_df = pd.read_csv(fileLocation, names=["lon", "lat"])
        for idx in range(nos_df.shape[0]):
            if nos_df['lon'][idx] <=-9999:
                nos_df['lon'][idx] = np.nan
                nos_df['lat'][idx] = np.nan
        return nos_df

    def GaussianPlume(self, lon,lat,fLon,fLat,uu,vv, aA=104, aB=213, wA=6, wB=2):
        # Grid info
        nX,nY = len(lon),len(lat)
        c   = np.zeros([nY,nX],dtype=float)
        # Windspeed and direction
        wspd = np.sqrt(uu**2.+vv**2.)
        wdir = np.arctan2(vv,uu)
        # Parameters and stability class
        x0   = 1e3 
        # wA = 6; wB = 2
        # aA, wA = 10., 6. #104
        # aB, wB = 21., 2. #213
        a   = (wspd - wA)/(wB - wA)*(aB - aA) + aA
        if a < aA: 
            a = aA
        if a > aB: 
            a = aB
        # Flatten the matrices
        lon,lat = np.meshgrid(lon,lat)
        out_dim = c.shape
        c, lon, lat = c.flatten('F'), lon.flatten('F'), lat.flatten('F')
        # Determine which values we should be calculating
        calc = [(vv > 0 and fLat <= lat[i]) \
        or (vv < 0 and fLat >= lat[i]) \
        or (uu > 0 and fLon <= lon[i]) \
        or (uu < 0 and fLon >= lon[i]) for i in range(len(c))]
        # Gaussian plume
        xx   = (lon[calc] - fLon)/120.*1e3
        yy   = (lat[calc] - fLat)/120.*1e3
        r    = np.sqrt(xx**2.+yy**2.)
        phi   = np.arctan2(yy,xx)-wdir
        lx   = r*np.cos(phi)
        ly   = r*np.sin(phi)
        sig   = a*(lx/x0)**0.894
        c[calc] = 1./(sig*wspd) * np.exp(-0.5 * (ly/sig)**2. )
        # Check for NaNs
        c[np.isnan(c)] = 0.
        # Reshape the plume to a matrix
        return np.reshape(c,out_dim,order='F')


    def plot_singleLoc(self, lats, lons, data, saveLocation=singleLocPlotLocation, title="", save=False, show=False):
        ''' Plot gaussianPlume '''
        h = plt.pcolor(lons, lats, data, vmin=0, vmax=np.max(data)/10000)
        plt.colorbar(h)
        h2 = plt.plot(self.nos_df['lon'], self.nos_df['lat'], 'k',)
        # plt.xlim(-125, -120)
        # plt.ylim(36, 40)
        plt.xlim(-123, -122)
        plt.ylim(37.5, 38.5)
        if title:
            plt.title(title)
        if save:
            plt.savefig(saveLocation+title.replace(" ", "_")+".png")
        if show:
            plt.show()
        plt.close()
        return

    def getGIFSingleLoc(self, plotLocation, gifname, extension="", drop=False):
        if extension:
            ext = extension
        else:
            ext = ".png"

        filenames = self.get_files(plotLocation, extension=ext)
        with imageio.get_writer(gifname, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                for idx in range(5):
                    writer.append_data(image)
        print(f'GIF: {gifname} has been saved!')
        if drop:
            # Remove files
            for filename in set(filenames):
                os.remove(filename)
        return


if __name__ == '__main__':
    startLat = 36
    endLat = 40
    startLon = -125
    endLon = -120
    nLon = 601
    nLat = 481
    lons = np.linspace(startLon, endLon, nLon)
    lats = np.linspace(startLat, endLat, nLat)
    receptorLon = -122.237
    receptorLat = 37.804
    u = 13.56; v=-10.11
    plume = GaussianPlume()
    nos_df = plume.getUSCoastLine()
    data = plume.GaussianPlume(lons, lats, receptorLon, receptorLat, u,v)
    plume.plot_singleLoc(lats, lons, data, title=f"Gaussian Plume for (u:{u}, v:{v}) at {receptorLat}, {receptorLon}", save=True)
    plume.getGIFSingleLoc(singleLocPlotLocation, "../data/plots/plume.gif")

