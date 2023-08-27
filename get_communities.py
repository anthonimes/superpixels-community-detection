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
        "LP": (utils.LP,[5],[0.98]), 
        "louvain": (utils.louvain,[10],[0.98]), 
        "infomap": (utils.infomap,[5],[0.98]), 
        }

class Segment(object):
    def __init__(self,dirpath,algorithm,dataset,stamp,radius,threshold):
        self.dirpath            = dirpath
        self.algorithm          = algorithm
        self.function           = algorithms_properties[algorithm][0]
        self.radius             = radius
        self.threshold          = threshold
        self.dataset            = dataset
        self.stamp              = stamp

    def __call__(self, filename):
        utils.get_communities(self.function,self.algorithm,self.dirpath,filename,self.dataset,stamp=self.stamp,radius=self.radius,threshold=self.threshold)

if __name__ == "__main__":
    dirpath,_,images = list(walk(absolute_path+"/dataset/"+dataset+"/images/"+folder))[0]
    try:
        # --- DEV --- open image here in order to compute RGB and Lab
        jobs = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(jobs) # remove number to use all
        for algorithm in algorithms_properties:
            radius = algorithms_properties[algorithm][1]
            threshold = algorithms_properties[algorithm][2]
            for r in radius:
                for t in threshold:
                    posix_now = time.time()
                    d = datetime.datetime.fromtimestamp(posix_now)
                    stamp = "".join(str(d).split(".")[:-1])
                    makedirs(absolute_path+"/communities/"+algorithm+"/"+dataset+"/"+str(r)+"-"+str(t),exist_ok=True)

                    segment = Segment(dirpath,algorithm,dataset,stamp,r,t)
                    cs=1
                    if len(images) >= jobs:
                        cs=len(images)//jobs
                    pool.map_async(segment, sorted(images), chunksize=cs)
    except Exception as e:
        print(e)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
