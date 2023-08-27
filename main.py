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
import datetime,time

absolute_path = path.dirname(path.abspath(__file__))
datasets = [ 
        ("BSDS","test"),
        #("SBD","test"), 
        #("NYUV2","test"),  
        #("SUNRGBD","test") 
]
radii = {"LP": [5], "louvain": [5], "infomap": [5]}
graph_weight_thresholds = {"LP": [0.98], "louvain": [0.98], "infomap": [0.98]} 
numbers_of_regions = [5000,2500,2000,1500,1000,800,600,400,200]

superpixels = { 
        "LP": (utils.LP,radii["LP"],graph_weight_thresholds["LP"],numbers_of_regions,merge_methods,merge_thresholds), 
        "louvain": (utils.louvain,radii["louvain"],graph_weight_thresholds["louvain"],numbers_of_regions,merge_methods,merge_thresholds), 
        "infomap": (utils.infomap,radii["infomap"],graph_weight_thresholds["infomap"],numbers_of_regions,merge_methods,merge_thresholds), 
        }

class Segment(object):
    def __init__(self,dirpath,method,dataset,folder,number_of_regions,stamp,radius,threshold):
        self.dirpath                     = dirpath
        self.name                        = method
        self.function                    = superpixels[method][0]
        self.number_of_regions           = number_of_regions
        self.radius                      = radius
        self.threshold                   = threshold
        self.dataset                     = dataset
        self.folder                      = folder
        self.stamp                       = stamp

    def __call__(self, filename):
        utils.study(self.function,self.name,self.dirpath,filename,self.dataset,number_of_regions=self.number_of_regions,write=True,weighted=self.weighted,stamp=self.stamp,radius=self.radius,threshold=self.threshold,merge_similarity_threshold=self.merge_similarity_threshold,merge_method=self.merge_method)
        return None

if __name__ == "__main__":
    try:
        jobs = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(jobs) # remove number to use all
        for name in superpixels:
            for dataset,folder in datasets:
                dirpath,_,images = list(walk(absolute_path+"/dataset/"+dataset+"/images/"+folder))[0]
                for radius in radii[name]:
                    for graph_weight_threshold in graph_weight_thresholds[name]:
                        for number_of_regions in numbers_of_regions:
                            posix_now = time.time()
                            d = datetime.datetime.fromtimestamp(posix_now)
                            stamp = "".join(str(d).split(".")[:-1])

                            makedirs(absolute_path+"/output/"+name+"/"+dataset+"/"+str(radius)+"/"+str(number_of_regions)+"/"+str(graph_weight_threshold)+"-"+stamp,exist_ok=True)
                            makedirs(absolute_path+"/csv/"+name+"/"+dataset+"/"+str(radius)+"/"+str(number_of_regions)+"/"+str(graph_weight_threshold)+"-"+stamp,exist_ok=True)

                            segment = Segment(dirpath,name,dataset,folder,number_of_regions,stamp,radius,graph_weight_threshold)
                            pool.map_async(segment, sorted(images),chunksize=len(images)//jobs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
