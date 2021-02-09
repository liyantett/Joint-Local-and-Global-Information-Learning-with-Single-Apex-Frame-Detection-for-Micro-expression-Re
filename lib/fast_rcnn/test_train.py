# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# R*CNN
# Written by Georgia Gkioxari, 2015.
# See LICENSE in the project root for license information.
# --------------------------------------------------------

"""Test a R*CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import scipy.io as sio
import utils.cython_bbox
import pickle
import matplotlib.pyplot as plt
import pylab

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None, 'secondary_rois': None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    blobs['secondary_rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes





# vissquare
def vissquare(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()
    
    print data.shape
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #
    print n
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.imshow(data); plt.axis('off')
    pylab.show()  
    #plt.imshow(data)




def visualize_feature_map_sum(feature_map):
    

    feature_map_combination = []

    
    num_pic = feature_map.shape[0]

    
    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]
        feature_map_combination.append(feature_map_split)


    feature_map_sum = sum(one for one in feature_map_combination)

    #plt.imshow(feature_map_sum)
    #plt.savefig('./mao_feature/feature_map_sum2.png')
    #plt.show()





def im_detect(net, im, boxes, gt_label):
    """Detect classes in an image given object proposals.
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)
    base_shape = blobs['data'].shape
    
    gt_inds = np.where(gt_label>-1)[0]
    num_rois = len(gt_inds)
    blobs_rois = blobs['rois'][gt_inds].astype(np.float32, copy=False)
    blobs_rois = blobs_rois[:, :, np.newaxis, np.newaxis]

    non_gt_inds = np.where(gt_label==-1)[0]
    num_sec_rois = len(non_gt_inds)
    blobs_sec_rois = blobs['secondary_rois'][non_gt_inds].astype(np.float32, copy=False)
    blobs_sec_rois = blobs_sec_rois[:, :, np.newaxis, np.newaxis]

    # reshape network inputs
    net.blobs['data'].reshape(base_shape[0], base_shape[1],
                              base_shape[2], base_shape[3])
    net.blobs['rois'].reshape(num_rois, 5, 1, 1)
    net.blobs['secondary_rois'].reshape(num_sec_rois, 5, 1, 1)
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs_rois,
                            secondary_rois = blobs_sec_rois)

    scores= blobs_out['cls_score']
    secondary_scores = blobs_out['context_cls_score']



   
    gt_boxes = boxes[gt_inds]
    sec_boxes = boxes[non_gt_inds]

    # Compute overlap
    boxes_overlaps = \
      utils.cython_bbox.bbox_overlaps(sec_boxes.astype(np.float), 
                                      gt_boxes.astype(np.float))

    selected_boxes = np.zeros((scores.shape[1], 4, gt_boxes.shape[0]))

    scores_gt=scores
    # MAX Here i==1   
    for i in xrange(gt_boxes.shape[0]):
        keep_inds = np.where((boxes_overlaps[:,i]>=cfg.TEST.IOU_LB) &
                             (boxes_overlaps[:,i]<=cfg.TEST.IOU_UB))[0]
        
        if keep_inds.size > 0:
            this_scores = np.amax(secondary_scores[keep_inds,:], axis=0)
            scores[i,:] = scores[i,:]+this_scores
            winner_ind  = np.argmax(secondary_scores[keep_inds,:], axis=0)            
            selected_boxes[:,:,i] = sec_boxes[keep_inds[winner_ind]]
            #print scores
       
       
    # Softmax 
    scores=scores_gt+scores
    scores = np.exp(scores-np.amax(scores))
    scores = scores / np.array(np.sum(scores, axis=1), ndmin=2).T

    #fc7= net.blobs_out['fc7']
    #context_fc7 = net.blobs_out['context_fc7']
    #save('fc7.mat',fc7)
    #save('context_fc7.mat',context_fc7)
     
    #visualization 
    feat = net.blobs['conv1_2'].data[0, :32]
    #with open('FirstLayerOutput.pickle','wb') as f:   
    #    pickle.dump(feat,f)
    #vissquare(feat,padval=1)
    visualize_feature_map_sum(feat)
    #pool = net.blobs['pool5'].data[0,:512]
    #with open('pool1.pickle','wb') as f:
    #    pickle.dump(pool,f)
    #vissquare(pool,padval=1)






    # Apply bounding-box regression deltas 
    #box_deltas = blobs_out['bbox_pred']
    #pred_boxes = _bbox_pred(gt_boxes, box_deltas)
    #pred_boxes = _clip_boxes(pred_boxes, im.shape)

    return scores, secondary_scores

def vis_detections(im, boxes, scores, classes):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(1):
        bbox = boxes[i, :4]
        sscore = scores[i, :]
        cls_ind = sscore.argmax()
        sscore = sscore.max()
        
        #plt.cla()
        plt.imshow(im)
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                           bbox[2] - bbox[0],
                           bbox[3] - bbox[1], fill=False,
                          edgecolor='r', linewidth=3)
                )
        plt.title('{}  {:.3f}'.format(classes[cls_ind], sscore))
        plt.show()

def test_net(net, imdb):
    """Test a R*CNN network on an image database."""
    num_images = len(imdb.image_index)
    num_classes = imdb.num_classes

    all_boxes = np.zeros((0, 2+num_classes), dtype = np.float32)
    all_selected_boxes = np.zeros((num_classes, 4, 0))


    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer()}

    roidb = imdb.roidb
    s=0
    s2=0
    print num_images
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        gt = np.where(roidb[i]['gt_classes']>-1)[0]
        #gt_boxes = roidb[i]['boxes'][gt]

        _t['im_detect'].tic()
        scores, secondary_scores = im_detect(net, im, roidb[i]['boxes'], roidb[i]['gt_classes'])
        _t['im_detect'].toc()
	maxscore=np.max(scores)
        ind=np.argwhere(scores==maxscore)
        indy=ind[0][1]
	if indy==roidb[i]['gt_classes'][0]:
           s=s+1
        print indy,roidb[i]['gt_classes'][0]
        if i+1==num_images:
	   print s,i+1

        #max2score=np.max(secondary_scores)   
        #print(max2score)
        #ind2=np.argwhere(secondary_scores==max2score)
        #indy2=ind2[0][1]
	#if indy2==roidb[i]['gt_classes'][0]:
        #   s2=s2+1
        #print s2



        #print roidb[i]['gt_classes'][0]
        #print indy
     
       

        #Visualize detections
        #vis_detections(im, gt_boxes, scores, imdb.classes)
    """

        for j in xrange(gt_boxes.shape[0]):
            # store image id and voc_id (1-indexed)
            temp = np.array([i+1, j+1], ndmin=2)
            print temp
            temp = np.concatenate((temp, np.array(scores[j,:],ndmin=2)), axis=1)
            print temp
            print all_boxes
            all_boxes = np.concatenate((all_boxes, temp), axis=0)

        all_selected_boxes = np.concatenate((all_selected_boxes, selected_boxes), axis = 2)

        print 'im_detect: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time)

    print 'Writing VOC results'
   
    imdb._write_voc_results_file(all_boxes)
    """
