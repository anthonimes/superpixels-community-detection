from skimage.segmentation import mark_boundaries, find_boundaries, slic

from statistics import mean
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure,img_as_ubyte,data,img_as_uint,filters
from skimage import graph
from skimage import filters,io,color
from skimage.util import img_as_float

import numpy, sys, pickle

import matplotlib.pyplot as plt

from os import path
absolute_path = path.dirname(path.abspath(__file__))

def baseline_radius_graph(image,radius=10,sigma=125,threshold=.9):
    import math
    hoffset = image.shape[1]

    vertices = [ u*hoffset+v for u in range(image.shape[0]) for v in range(image.shape[1]) ]
    connectivity = 1

    arcs = list()

    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            pixel = u*hoffset+v

            row_begin,col_begin = (max(0,u-radius),max(0,v-radius))
            row_end,col_end = (min(image.shape[0]-1,u+radius),min(image.shape[1]-1,v+radius))

            indices=[(r,v) for r in range(row_begin,row_end+1) if r!=u]
            indices.extend([(u,c) for c in range(col_begin,col_end+1) if c!=v])

            tmp=list()
            for index,(r_index,c_index) in enumerate(indices):
                diff = image[(r_index,c_index)] - image[(u,v)]
                diff = numpy.linalg.norm(diff)
                sim = math.e ** (-(diff ** 2) / sigma)
                if(sim>=threshold or threshold==0):
                    tmp.append((pixel,r_index*hoffset+c_index,sim))
            arcs.extend(tmp)

    return vertices,arcs

def study(function,name,dirpath,filename,dataset,number_of_regions=1000,write=True,stamp=None,radius=1,threshold=0.98):
    image_file = io.imread(dirpath+"/"+filename)
    image = img_as_float(image_file)
    image = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
    
    with open(absolute_path+"/communities/"+name+"/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+filename[:-4]+".pkl", "rb") as f:
        segmentation = pickle.load(f)
 
    merged_segmentation = segmentation.copy()
    if(threshold_merge>0):
        merged_segmentation = numpy_merge(segmentation, image, number_of_regions)
    
    if(write):
        output(name,dataset,filename,image_file,merged_segmentation,threshold_merge,stamp,radius,threshold)

def get_graphs(method,name,dirpath,filename,dataset,write=False,weighted=True,stamp=None,radius=1,threshold=0):
    import networkx
    image_file = io.imread(dirpath+"/"+filename)
    image = image_file
    image = img_as_float(image_file)
    image = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
    vertices,arcs = baseline_radius_graph(image,radius=radius,threshold=threshold)

    G = networkx.Graph()
    G.add_nodes_from(vertices)
    G.add_weighted_edges_from(arcs)

    with open(absolute_path+"/graphs/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+filename[:-4]+".pkl", "wb") as f:
        pickle.dump(G,f)

def get_communities(method,name,dirpath,filename,dataset,write=False,weighted=True,stamp=None,radius=1,threshold=0):
    import networkit
    image_file = io.imread(dirpath+"/"+filename)
    image = image_file
    image = img_as_float(image_file)
    image = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]

    with open(absolute_path+"/graphs/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+filename[:-4]+".pkl", "rb") as f:
        G = pickle.load(f)

    initial_segmentation = method(G,image)
    
    # --- DEV --- output men region size after merging
    if(write):
        with open(absolute_path+"/communities/"+name+"/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+filename[:-4]+".pkl", "wb") as f:
            pickle.dump(initial_segmentation,f)

def LP(G,image):
    import networkit

    Gk = networkit.nxadapter.nx2nk(G,weightAttr="weight")

    initial_segmentation = networkit.community.detectCommunities(Gk,algo=networkit.community.PLP(Gk)).getVector()
    
    initial_segmentation = numpy.asarray(initial_segmentation)+1
    initial_segmentation=(1+initial_segmentation).reshape((image.shape[0],image.shape[1]))

    return initial_segmentation

def louvain(G,image):
    import networkit

    Gk = networkit.nxadapter.nx2nk(G,weightAttr="weight")

    initial_segmentation = networkit.community.detectCommunities(Gk,algo=networkit.community.PLM(Gk,True,gamma=1)).getVector()
    
    initial_segmentation = numpy.asarray(initial_segmentation)+1
    initial_segmentation = initial_segmentation.reshape(image.shape[0],image.shape[1])

    return initial_segmentation

def infomap(G,image):
    from infomap import Infomap

    im = Infomap()
    mapping = im.add_networkx_graph(G)
    im.run(no_file_output=True,silent=False,two_level=True)
    initial_segmentation = [ i for i in range(image.shape[0]*image.shape[1])  ]

    for node in im.nodes:
        initial_segmentation[mapping[node.node_id]]=node.module_id

    initial_segmentation = numpy.asarray(initial_segmentation)

    initial_segmentation = initial_segmentation.reshape(image.shape[0], image.shape[1])
    _, initial_segmentation = numpy.unique(initial_segmentation,return_inverse=1)
    initial_segmentation=(1+initial_segmentation).reshape((image.shape[0],image.shape[1]))

    return initial_segmentation

def numpy_merge(segmentation, image, number_of_regions):
    to_merge = numpy.copy(segmentation)
    min_size = (image.shape[0]*image.shape[1])//number_of_regions

    labels = dict.fromkeys(numpy.unique(to_merge))
    # Merging small-sized regions first
    while(True):
        merged=False
        g = graph.rag_mean_color(image,to_merge,connectivity=1,mode='similarity',sigma=125)
        for region in sorted(g.nodes):
            if(len(to_merge[to_merge==region])<=(min_size//10)):
                merged=True
                closest = max([(v,g[region][v]['weight']) for v in g.neighbors(region)],key=lambda x: x[1])
                to_merge[to_merge==region] = closest[0]
                if(region in labels.keys()):
                    labels.pop(region)
                    labels[closest[0]]=None
                if(len(labels)<=number_of_regions):
                    _, to_merge = numpy.unique(to_merge,return_inverse=1)
                    to_merge=(1+to_merge).reshape((image.shape[0],image.shape[1]))

                    return to_merge
        if(merged):
            continue
        break

    while(True):
        g = graph.rag_mean_color(image,to_merge,connectivity=1,mode='similarity',sigma=125)

        for region in sorted(g.nodes):
            if(len(to_merge[to_merge==region])<=min_size):
                closest = max([(v,g[region][v]['weight']) for v in g.neighbors(region)],key=lambda x: x[1])
                to_merge[to_merge==region] = closest[0]
                if(region in labels.keys()):
                    labels.pop(region)
                    labels[closest[0]]=None
                if(len(labels)<=number_of_regions):
                    _, to_merge = numpy.unique(to_merge,return_inverse=1)
                    to_merge=(1+to_merge).reshape((image.shape[0],image.shape[1]))

                    return to_merge

def output(name,dataset,filename,image,segmentation,number_of_regions,stamp,radius,threshold):
    import csv
    io.imsave(absolute_path+"/output/"+name+"/"+dataset+"/"+str(radius)+"/"+str(number_of_regions)+"/"+str(threshold)+"-"+stamp+"/"+filename[:-4]+".png",img_as_ubyte(mark_boundaries(img_as_float(image),segmentation,color=(0,0,0))))
    with open(absolute_path+"/csv/"+name+"/"+dataset+"/"+str(radius)+"/"+str(number_of_regions)+"/"+str(threshold)+"-"+stamp+"/"+filename[:-4]+".csv", "w", newline='') as csvfile:
        segwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in segmentation:
            segwriter.writerow(line)
