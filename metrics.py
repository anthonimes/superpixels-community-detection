from statistics import mean
from math import sqrt,floor
from os import path

import numpy

absolute_path = path.dirname(path.abspath(__file__))

# ----- SUPERPIXEL QUALITY METRICS -----
def intersection_matrix(labels,gt):
    max_gt = gt.max()
    max_labels = labels.max()
    result = numpy.zeros((max_gt,max_labels))

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            result[(gt[(i,j)]-1,labels[(i,j)]-1)]+=1

    return result

def UEIN(segmentation,gt):
    H = segmentation.shape[0]
    W = segmentation.shape[1]
    n = H*W

    im = intersection_matrix(segmentation,gt)
    error = 0.
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if(im[(i,j)]>0):
                error += min(im[(i,j)], len(segmentation[segmentation==j+1])-im[(i,j)])

    return error/n

def ASA(segmentation,gt):
    H = segmentation.shape[0]
    W = segmentation.shape[1]
    n = H*W

    im = intersection_matrix(segmentation,gt)
    accuracy = 0.
    for j in range(im.shape[1]):
        maximum = 0
        for i in range(im.shape[0]):
            if(im[(i,j)]>maximum):
                 maximum= im[(i,j)]
        accuracy += maximum

    return accuracy/n

def USE_ASA(gt,seg,img):
    return UEIN(seg,gt),ASA(seg,gt)

def is_4_connected(segmentation, i, j):
    if(i > 0):
        if(segmentation[i][j] != segmentation[i-1][j]):
            return True

    if(i < segmentation.shape[0]-1):
        if(segmentation[i][j] != segmentation[i+1][j]):
            return True

    if(j > 0):    
        if(segmentation[i][j] != segmentation[i][j-1]):
            return True

    if(j < segmentation.shape[1] - 1):
        if(segmentation[i][j] != segmentation[i][j+1]):
            return True

    return False

def boundary_recall(segmentation, gt, d=0.0025):
    import sys
    H = segmentation.shape[0]
    W = segmentation.shape[1]
    r = round(d*sqrt(H*H+W*W))

    tp = 0
    fn = 0

    for i in range(H):
        for j in range(W):
            if(is_4_connected(gt,i,j)):
                pos = False
                for k in range(max(0,i-r),min(H-1,i+r)+1):
                    for l in range(max(0,j-r),min(W-1,j+r)+1):
                        if(is_4_connected(segmentation,k,l)):
                            pos = True
                if(pos):
                    tp+=1
                else:
                    fn+=1

    if(tp+fn)>0:
        return tp/(tp+fn)
    return 0

def boundary_precision(segmentation, gt, d=0.0025):
    H = segmentation.shape[0]
    W = segmentation.shape[1]
    r = round(d*sqrt(H*H+W*W))

    tp = 0.
    fp = 0.

    for i in range(H):
        for j in range(W):
            if(is_4_connected(gt,i,j)):
                pos = False
                for k in range(max(0,i-r),min(H-1,i+r)+1):
                    for l in range(max(0,j-r),min(W-1,j+r)+1):
                        if(is_4_connected(segmentation,k,l)):
                            pos = True
                if(pos):
                    tp+=1
            elif(is_4_connected(segmentation,i,j)):
                pos = False
                # --- DEV --- might be going a bit too far here
                for k in range(max(0,i-r),min(H-1,i+r)+1):
                    for l in range(max(0,j-r),min(W-1,j+r)+1):
                        if(is_4_connected(gt,k,l)):
                            pos = True
                if(not pos):
                    fp+=1

    if(tp+fp)>0:
        return tp/(tp+fp)
    return 0

def compactness(segmentation,image):
    number_sp = numpy.max(segmentation)

    perimeters = [0]*number_sp
    areas = [len(segmentation[segmentation==i]) for i in range(1,number_sp+1)]

    H = segmentation.shape[0]
    W = segmentation.shape[1]

    for i in range(H):
        for j in range(W):
            count = 0
            if(i>0):
                if(segmentation[i,j]!=segmentation[i-1,j]):
                    count+=1
            else:
                count += 1

            if(i < H-1):
                if(segmentation[i,j]!=segmentation[i+1,j]):
                    count+=1
            else:
                count += 1

            if(j>0):
                if(segmentation[i,j]!=segmentation[i,j-1]):
                    count+=1
            else:
                count += 1

            if(j < W-1):
                if(segmentation[i,j]!=segmentation[i,j+1]):
                    count+=1
            else:
                count += 1

            perimeters[segmentation[i,j]-1]+=count

    compactness_value = 0.

    for i in range(number_sp):
        if (perimeters[i]>0):
            compactness_value += areas[i] * (4*numpy.pi*areas[i]) / (perimeters[i] * perimeters[i])

    return (compactness_value/(H*W))

def EV(segmentation,image):
    number_sp = numpy.max(segmentation)
    overall_mean = [0,0,0]
    sp_mean = [ [0,0,0] for i in range(number_sp) ]
    count = [ 0 ] * number_sp
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            for c in range(image.shape[2]):
                overall_mean[c]+=image[u][v][c]
                sp_mean[segmentation[u][v]-1][c]+=image[u][v][c]
            count[segmentation[u][v]-1] += 1

    for c in range(image.shape[2]):
        overall_mean[c] = overall_mean[c]/(image.shape[0]*image.shape[1])
       
    for i in range(number_sp):
        for c in range(image.shape[2]):
            sp_mean[i][c] = (sp_mean[i][c] / count[i])

    sum_tmp,sum_top,sum_bottom=0.,0.,0.
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            for c in range(image.shape[2]):
                sum_top+=(sp_mean[segmentation[u][v]-1][c]-overall_mean[c])*(sp_mean[segmentation[u][v]-1][c]-overall_mean[c])
                sum_bottom+=(image[u][v][c]-overall_mean[c])*(image[u][v][c]-overall_mean[c])

    return sum_top/sum_bottom

def compare(filename,segmentation,compute_groundtruth,gt_extension,dirpath,dataset,folder,several_gt=False):
    image_file = io.imread(dirpath+"/"+filename)
    image = img_as_float(image_file)
    image = (color.rgb2lab(image) + [0,128,128]) #// [1,1,1]
    uenp,asa,br,ev=0,0,0,0
    gt = compute_groundtruth(absolute_path+"/"+dataset+"/groundTruth/"+folder+"/"+filename[:-4]+gt_extension)

    if(several_gt==True):
        measures = [ (USE_ASA(gt[i],segmentation,image),boundary_recall(segmentation,gt[i])) for i in range(len(gt)) ] 
    else:
        measures = [ (USE_ASA(gt,segmentation,image),boundary_recall(segmentation,gt)) ]#,boundary_precision(segmentation,gt)) ] 

    uenp=[e[0][0] for e in measures]
    asa=[e[0][1] for e in measures]
    br=[e[1] for e in measures]
    bp=[e[2] for e in measures]
    ev=EV(segmentation,image)
    ev_rgb = EV(segmentation,image_file)
    compactness_value = compactness(segmentation,image_file)

    return (uenp,asa,br,bp,ev,ev_rgb,compactness_value)

def _get_groundtruth_csv(filepath):
    import csv
    segmentation = []

    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            segmentation.append(list(map(int,row)))
    segmentation = numpy.asarray(segmentation)
    return segmentation

def _get_groundtruth_BSDS(filepath):
    from scipy.io import loadmat
    groundtruth = loadmat(filepath)
    segmentation = []
    for i in range(len(groundtruth['groundTruth'][0])):
        # groundtruths boundaries and segmentation as numpy arrays
        segmentation.append(groundtruth['groundTruth'][0][i][0]['Segmentation'][0])
    return segmentation
