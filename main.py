#!/usr/bin/python3
from os import walk, path

from statistics import mean

import metrics, utils
import sys

import multiprocessing 

try:
   multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
   pass

from os import makedirs

absolute_path = path.dirname(path.abspath(__file__))
datasets = [
        ("example","test"),
        #("BSDS","test"),
        #("SBD","test"), 
        #("NYUV2","test"),  
        #("SUNRGBD","test") 
]

algorithms = ["LP","louvain","infomap"]
radii = {"LP": [5], "louvain": [5], "infomap": [5]}
graph_weight_thresholds = {"LP": [0.98], "louvain": [0.98], "infomap": [0.98]} 
numbers_of_regions = [5000,2500,2000,1500,1000,800,600,400,200]

class Segment(object):
    def __init__(self,dirpath,algorithm,dataset,folder,number_of_regions,radius,threshold):
        self.dirpath                     = dirpath
        self.algorithm                   = algorithm
        self.number_of_regions           = number_of_regions
        self.radius                      = radius
        self.threshold                   = threshold
        self.dataset                     = dataset
        self.folder                      = folder

    def __call__(self, filename):
        utils.study(self.algorithm,self.dirpath,filename,self.dataset,number_of_regions=self.number_of_regions,radius=self.radius,threshold=self.threshold)

if __name__ == "__main__":
    try:
        jobs = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(jobs) # remove number to use all
        for algorithm in algorithms:
            for dataset,folder in datasets:
                dirpath,_,images = list(walk(absolute_path+"/dataset/"+dataset+"/images/"+folder))[0]
                for radius in radii[algorithm]:
                    for graph_weight_threshold in graph_weight_thresholds[algorithm]:
                        for number_of_regions in numbers_of_regions:
                            makedirs(absolute_path+"/output/"+algorithm+"/"+dataset+"/"+str(radius)+"-"+str(graph_weight_threshold)+"/"+str(number_of_regions),exist_ok=True)
                            makedirs(absolute_path+"/csv/"+algorithm+"/"+dataset+"/"+str(radius)+"-"+str(graph_weight_threshold)+"/"+str(number_of_regions),exist_ok=True)

                            segment = Segment(dirpath,algorithm,dataset,folder,number_of_regions,radius,graph_weight_threshold)
                            cs=1
                            if len(images) >= jobs:
                                cs=len(images)//jobs
                            pool.map_async(segment, sorted(images),chunksize=cs)
    except Exception as e:
        print(e)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
