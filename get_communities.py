#!/usr/bin/python3
from os import walk, path, makedirs

from statistics import mean

import metrics, utils
import sys

import multiprocessing 

try:
   multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
   pass

absolute_path = path.dirname(path.abspath(__file__))
import datetime,time

dataset = sys.argv[1]
folder = sys.argv[2]

# function, weighted graph, radii and threshold
algorithms_properties = { 
        #"LP": (utils.LP,True,[5],[0.98]), 
        #"louvain": (utils.louvain,True,[10],[0.98]), 
        "infomap": (utils.infomap,True,[5],[0.98]), 
        }

class Segment(object):
    def __init__(self,dirpath,algorithm,dataset,folder,stamp,radius,threshold):
        self.dirpath            = dirpath
        self.algorithm          = algorithm
        self.function           = algorithms_properties[algorithm][0]
        self.weighted           = algorithms_properties[algorithm][1]
        self.radius             = radius
        self.threshold          = threshold
        self.dataset            = dataset
        self.folder             = folder
        self.stamp              = stamp

    def __call__(self, filename):
        utils.get_communities(self.function,self.name,self.dirpath,filename,self.dataset,write=True,weighted=self.weighted,stamp=self.stamp,radius=self.radius,threshold=self.threshold)
        return None

if __name__ == "__main__":
    dirpath,_,images = list(walk(absolute_path+"/dataset/"+dataset+"/images/"+folder))[0]
    try:
        # --- DEV --- open image here in order to compute RGB and Lab
        jobs = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(10) # remove number to use all
        for algorithm in algorithms_properties:
            radius = algorithms_properties[algorithm][2]
            threshold = algorithms_properties[algorithm][3]
            for r in radius:
                for t in threshold:
                    posix_now = time.time()
                    d = datetime.datetime.fromtimestamp(posix_now)
                    stamp = "".join(str(d).split(".")[:-1])
                    makedirs(absolute_path+"/"+name+"/"+dataset+"/"+str(r)+"-"+str(t),exist_ok=True)

                    segment = Segment(dirpath,algorithm,dataset,folder,stamp,r,t)
                    results = pool.map_async(segment, sorted(images), chunksize=len(images)//jobs)

    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
