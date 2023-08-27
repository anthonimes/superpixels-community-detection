#!/usr/bin/python3
from os import walk

from statistics import mean

import metrics, utils
import sys

import multiprocessing 

try:
   multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
   pass

from os import makedirs, path

absolute_path = path.dirname(path.abspath(__file__))

# weighted?, radii, thresholds
properties= (True,[5],[0.98])

dataset = sys.argv[1]
folder = sys.argv[2]

class Segment(object):
    def __init__(self,dirpath,dataset,folder,radius,threshold):
        self.dirpath            = dirpath
        self.weighted           = properties[0]
        self.radius             = radius
        self.threshold          = threshold
        self.dataset            = dataset
        self.folder             = folder

    def __call__(self, filename):
        utils.get_graphs(self.dirpath,filename,self.dataset,write=True,weighted=self.weighted,radius=self.radius,threshold=self.threshold)

if __name__ == "__main__":
    dirpath,_,images = list(walk(absolute_path+"/dataset/"+dataset+"/images/"+folder))[0]
    try:
        jobs = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(jobs) # remove number to use all
        radius = properties[1]
        threshold = properties[2]
        for r in radius:
            for t in threshold:
                makedirs(absolute_path+"/graphs/"+dataset+"/"+str(r)+"-"+str(t),exist_ok=True)

                segment = Segment(dirpath,dataset,folder,r,t)
                cs=1
                if len(images) >= jobs:
                    cs=len(images)//jobs
                pool.map_async(segment, sorted(images), chunksize=cs)
    except Exception as e:
        print(e)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
