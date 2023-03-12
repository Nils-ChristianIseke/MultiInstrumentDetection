import json
import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import utils
import _timing
from natsort import natsorted
import glob
from copy import deepcopy


class EndoSimData():
    """Dataset for endoscopic simulator dataset
    Basically pre-process the model GT and predictions
    And feed them into the HOTA script
    For this we need to compute the num GTs, num Dets, and the similarity function
    """

    def __init__(self):
        self.classes_to_eval = ['Nadelhalter', 'Nervhaken', 'Klappenschere', 'Atraum. Pinzette', 'Knotenschieber']

        self.valid_classes = ['nadelhalter', 'nervhaken', 'klappenschere', 'atraum. pinzette', 'knotenschieber']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.classes_to_eval]
        if not all(self.class_list):
            raise TrackEvalException('Attempted to evaluate an invalid class. '
                                     'Only classes [nadelhalter, atraum. Pinzette] are valid.')
        self.class_name_to_class_id = {'nadelhalter': '1', 'nervhaken': '2', 'klappenschere': '3', 'atraum. pinzette': '4', 'knotenschieber': '5', 'ignore': '6'} #CHANGED

    @staticmethod
    def _calculate_box_ious(bboxes1, bboxes2, box_format='xywh', do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """

        if len(bboxes1) == 0: #CHANGED
            bboxes1 = np.empty((0, 4)) #CHANGED
        if len(bboxes2) == 0: #CHANGED
            bboxes2 = np.empty((0, 4)) #CHANGED
            
        
        if box_format in 'xywh':
            # layout: (x0, y0, w, h)
            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in 'x0y0x1y1':
            raise (TrackEvalException('box_format %s is not implemented' % box_format))

        # layout: (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo('float').eps
            ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]

            return ioas
        else:
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
            intersection[union <= 0 + np.finfo('float').eps] = 0
            union[union <= 0 + np.finfo('float').eps] = 1
            ious = intersection / union
            return ious

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(bboxes1=np.asarray(gt_dets_t),
                                                     bboxes2=np.asarray(tracker_dets_t),
                                                     box_format='x0y0x1y1')
        return similarity_scores

    @staticmethod
    def _assign_ids(classes_list):
        ids = []
        for classes_t in classes_list:
            unique_classes_t = np.unique(classes_t)
            ids_t = [None] * len(classes_t)
            for class_ in unique_classes_t:
                id_ctr = 1
                for i, det in enumerate(classes_t):
                    if det == class_:
                        id_ = (det * 100) + id_ctr
                        ids_t[i] = id_
                        id_ctr += 1

            ids.append(np.asarray(ids_t))
        return ids

    def get_raw_seq_data(self):
        raw_gt_data = self._load_raw_gt_file()
        raw_tracker_data = self._load_raw_tracker_file()
        raw_data = {**raw_tracker_data, **raw_gt_data}

        # Calculate similarities for each timestep
        similarity_scores =[]
        for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
            ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)
            similarity_scores.append(ious)

        raw_data['similarity_scores'] = similarity_scores
        return raw_data


    def get_preprocessed_seq_data(self, raw_data, cls):
        # check unique IDs
        self._check_unique_ids(raw_data)
        cls_id = int(self.class_name_to_class_id[cls])

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}

        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = [raw_data['gt_dets'][t][ind] for ind in range(len(gt_class_mask)) if gt_class_mask[ind]]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = [raw_data['tracker_dets'][t][ind] for ind in range(len(tracker_class_mask)) if
                            tracker_class_mask[ind]]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Keep all tracker detections because we don't have an 'ignore' region
            # Ignore the ignore :)
            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets

            # Keep all ground truth detections
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['cls'] = cls

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)
        return data

    @staticmethod
    def _get_seq_lengths(seq_path):
        image_file_list = natsorted(glob.glob(os.path.join(seq_path, 'images', '*')))
        return len(image_file_list)

    @staticmethod
    def _check_unique_ids(data, after_preproc=False):
        """Check the requirement that the tracker_ids and gt_ids are unique per timestep"""
        gt_ids = data['gt_ids']
        tracker_ids = data['tracker_ids']
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            if len(tracker_ids_t) > 0:
                unique_ids, counts = np.unique(tracker_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Tracker predicts the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise TrackEvalException(exc_str)
            if len(gt_ids_t) > 0:
                unique_ids, counts = np.unique(gt_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = 'Ground-truth has the same ID more than once in a single timestep ' \
                                   '(seq: %s, frame: %i, ids:' % (data['seq'], t+1)
                    exc_str = ' '.join([exc_str_init] + [str(d) for d in duplicate_ids]) + ')'
                    if after_preproc:
                        exc_str_init += '\n Note that this error occurred after preprocessing (but not before), ' \
                                        'so ids may not be as in file, and something seems wrong with preproc.'
                    raise TrackEvalException(exc_str)

class TrackEvalException(Exception):
    """Custom exception for catching expected errors."""
    ...