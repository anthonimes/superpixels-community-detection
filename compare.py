#!/usr/bin/python3
from os import walk, path, makedirs

from statistics import mean

import metrics, utils
import sys
import numpy
import csv

import datetime,time
import multiprocessing 

dataset                 = sys.argv[1]
folder                  = sys.argv[2]
path_to_segmentations   = sys.argv[3]
name                    = sys.argv[4]

try:
   multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
   pass

def read_segmentation(pathfolder,method,dataset,filename):
        segmentation = list()
        filepath=pathfolder+"/"+filename[:-4]+".csv"
        with open(filepath, "r", newline='') as csvfile:
            segreader=csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in segreader:
                segmentation.append(list(map(int,row)))
        segmentation=numpy.asarray(segmentation)
        return segmentation

absolute_path = path.dirname(path.abspath(__file__))

# groundtruth method, groundtruth extension, several groundtruth?
compare_datasets = {
        "BSDS": (metrics._get_groundtruth_BSDS,".mat",True), \
        "NYUV2": (metrics._get_groundtruth_csv,".csv",False), \
        "SUNRGBD": (metrics._get_groundtruth_csv,".csv",False), \
        "SBD": (metrics._get_groundtruth_csv,".csv",False) \
        }

class Segment(object):
    def __init__(self,images_path,method,get_groundtruth,dataset,folder,csv_folder):
        self.images_path        = images_path
        self.name               = method
        self.dataset            = dataset
        self.folder             = folder
        self.csv_folder         = csv_folder
        self.get_groundtruth    = get_groundtruth

    def __call__(self, filename):
        # groundtruth comparison 
        segmentation = read_segmentation(self.csv_folder,self.name,self.dataset,filename)
        uenp,asa,br,bp,ev,ev_rgb,compactness=metrics.compare(filename,segmentation,self.get_groundtruth[0],self.get_groundtruth[1],self.images_path,self.dataset,self.folder,self.get_groundtruth[2])
        return (filename,len(numpy.unique(segmentation)),mean(uenp),max(uenp),min(uenp),mean(asa),min(asa),max(asa),mean(br),min(br),max(br),mean(bp),min(bp),max(bp),ev,ev_rgb,compactness)

if __name__ == "__main__":
    images_path,_,images = list(walk(absolute_path+"/dataset/"+dataset+"/images/"+folder))[0]
    (csv_path,radii_folders,_) = list(walk(path_to_segmentations))[0]
    try:
        jobs = multiprocessing.cpu_count()-1
        pool = multiprocessing.Pool(jobs) # remove number to use all

        for radius_folder in radii_folders:
            csv_folders = list(walk(csv_path+"/"+radius_folder))[0][1]
            for csv_folder in csv_folders:
                print("dealing with folder {}".format(csv_path+"/"+radius_folder+"/"+csv_folder))
                merge_csv_folders = list(walk(csv_path+"/"+radius_folder+"/"+csv_folder))[0][1]
                for merge_csv_folder in merge_csv_folders:
                    pathfolder=absolute_path+"/results/"+name+"/"+dataset+"/"
                    pathfolder+="/".join((path_to_segmentations.split("/"))[7:])+"/"+radius_folder+"/"+csv_folder
                    makedirs(pathfolder,exist_ok=True)

                    path_csv="/"
                    path_csv+="/".join((path_to_segmentations.split("/"))[1:-1])+"/"+"/"+radius_folder+"/"+csv_folder+"/"+merge_csv_folder
                    filepath = pathfolder+"/"+merge_csv_folder+".csv"

                    segment = Segment(images_path,name,compare_datasets[dataset],dataset,folder,path_csv)
                    cs=1
                    if(len(images) >= jobs):
                        cs=len(images)//jobs
                    results = pool.map(segment, images,chunksize=cs)
                    
                    with open(filepath, "w", newline='') as csvfile:
                        segwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

                        if(dataset in ["BSDS"]):
                            segwriter.writerow(("filename","segments","ue","max_ue","min_ue","asa","min_asa","max_asa","br","min_br","max_br","bp","min_bp","max_bp","ev","ev_rgb","compactness_value"))
                            for r in results:
                                segwriter.writerow(r)
                        else:
                            segwriter.writerow(("filename","segments","ue","asa","br","bp","ev","ev_rgb","compactness_value"))
                            for r in results:
                                segwriter.writerow([r[0],r[1],r[2],r[5],r[8],r[11],r[14],r[15],r[16]])

    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
