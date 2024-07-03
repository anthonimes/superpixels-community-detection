from skimage.segmentation import mark_boundaries, find_boundaries, slic

from statistics import mean
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure,img_as_ubyte,data,img_as_uint,filters
from skimage import graph
from skimage import filters,io,color
from skimage.util import img_as_float

from collections import defaultdict
import math

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

def study(name,dirpath,filename,dataset,numbers_of_regions=[1000],write=True,radius=1,threshold=0.98):
    image_file = io.imread(dirpath+"/"+filename)
    image = img_as_float(image_file)
    image = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
    
    with open(absolute_path+"/communities/"+name+"/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+filename[:-4]+".pkl", "rb") as f:
        segmentation = pickle.load(f)
    
    merged_segmentation = numpy_small_to_large_merge(segmentation, image, numbers_of_regions)
    
    if(write):
        output(name,dataset,filename,image_file,merged_segmentation,numbers_of_regions,radius,threshold)

def get_graphs(dirpath,filename,dataset,write=True,weighted=True,radius=1,threshold=0):
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

def get_communities(community_detection,name,dirpath,filename,dataset,write=True,weighted=True,radius=1,threshold=0):
    import networkit
    image_file = io.imread(dirpath+"/"+filename)
    image = image_file
    image = img_as_float(image_file)
    image = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]

    with open(absolute_path+"/graphs/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+filename[:-4]+".pkl", "rb") as f:
        G = pickle.load(f)

    initial_segmentation = community_detection(G,image)
    
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

def numpy_small_to_large_merge(segmentation, image, numbers_of_regions):
    to_merge = numpy.copy(segmentation)
    merge_result = {}
    avg_expected_size = (image.shape[0]*image.shape[1])//min(numbers_of_regions)

    g = graph.rag_mean_color(image,to_merge,connectivity=1,mode='similarity',sigma=125)

    # create a dictionary which holds a list of regions to consider for each size
    reg = [(len(to_merge[to_merge==r]), r) for r in g.nodes]
    r1 = defaultdict(list)
    for k, v in reg:
        r1[k].append(v)
    regions = dict((k, v) for k, v in r1.items())

    # dictionary used to keep track of whether a region still exists or not
    labels = dict.fromkeys(numpy.unique(to_merge))

    # dictionary used to solve region name change issues
    close_dict = dict()

    for k in range(1, avg_expected_size):

        if k in regions.keys(): # check if there is someone of this size (useful for big sizes mostly)

            if math.log(k, 2).is_integer(): # don't want to have to re-create the graph too often because it takes time
                g = graph.rag_mean_color(image,to_merge,connectivity=1,mode='similarity',sigma=125)

            # remove the regions that are in the wrong category
            filtered_regions = list(filter(lambda x: len(to_merge[to_merge==x]) == k, list(set(regions[k]))))

            for region in sorted(filtered_regions):

                closest = max([(v,g[region][v]['weight']) for v in g.neighbors(region)],key=lambda x: x[1])

                # look for the closest region, even if its name has changed
                c = closest[0]
                if len(to_merge[to_merge==c]) == 0:
                    while not len(to_merge[to_merge==c]) != 0:
                        c = close_dict[c]
                if len(to_merge[to_merge==region]) + len(to_merge[to_merge==c]) < 2*k:
                    print(region, len(to_merge[to_merge==region]), closest, len(to_merge[to_merge==c]))

                # merge the regions
                close_dict[region] = c
                to_merge[to_merge==region] = c
                l = len(to_merge[to_merge==c])

                # put the new region back in the "waiting list", to be considered again later
                if l not in regions.keys():
                    regions[l] = [c]
                else:
                    regions[l].append(c)

                # keep track of who has been deleted
                if(region in labels.keys()):
                    labels.pop(region)
                    labels[c]=None
                
                # end
                if(len(labels) in numbers_of_regions):
                    t_m = numpy.copy(to_merge)
                    unique, t_m = numpy.unique(t_m,return_inverse=1)
                    t_m=(1+t_m).reshape((image.shape[0],image.shape[1]))

                    merge_result[len(labels)] = numpy.copy(t_m)

                    if (len(labels) == min(numbers_of_regions)):
                        return merge_result

            regions.pop(k)

    # in case of failure
    unique, to_merge = numpy.unique(to_merge,return_inverse=1)
    print("FAIL nb regions ", len(unique))
    print(regions)

def output(name,dataset,filename,image,segmentation,numbers_of_regions,radius,threshold):
    import csv
    for number_of_regions in numbers_of_regions:
        io.imsave(absolute_path+"/output/"+name+"/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+str(number_of_regions)+"/"+filename[:-4]+".png",img_as_ubyte(mark_boundaries(img_as_float(image),segmentation[number_of_regions],color=(0,0,0))))
        with open(absolute_path+"/csv/"+name+"/"+dataset+"/"+str(radius)+"-"+str(threshold)+"/"+str(number_of_regions)+"/"+filename[:-4]+".csv", "w", newline='') as csvfile:
            segwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line in segmentation[number_of_regions]:
                segwriter.writerow(line)
