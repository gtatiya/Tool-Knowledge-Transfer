# Author: Gyan Tatiya

import argparse
import csv
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import scipy.io
import matlab.engine
from scipy.io import loadmat

from utils import time_taken, get_config, save_config, get_classes_labels, get_classifier, split_train_test_trials, \
    get_split_data_objects, augment_trials, get_new_labels, check_kema_data, classifier, update_all_modalities, \
    compute_mean_accuracy, plot_fold_all_modalities, plot_each_modality, plot_all_modalities

if __name__ == '__main__':
    '''
    This script projects data from source context to target context.
    The contexts can be behavior and tool combination.
    There are 2 mappings (-across): 1. across tools and same behavior and 2. across behaviors and same tool.

    The shared objects between source and target is increased incrementally from 1 to 10.
    The shared objects are used to train the projection function (KEMA).
    The number of folds is set to 10 (-num-folds).

    Recognition task is novel object identity recognition.
    Trained projection function is used to generate features of 5 novel objects of target context using source context.

    For evaluation, randomly choose a trial fold for novel/test objects.
    An object recognition model (-classifier-name) is trained on the generated features for transfer condition and 
    on original target context data for baseline 1 condition (best case),
    and on original source context data for baseline 2 condition (worse case).
    Fuses the predictions of all the modalities by uniform combination, train score, and test score.

   For data augmentation (-augment-trials), augment trials of the test objects in source and target contexts.

    Assumptions:
    Modality is the same across each projection.
    Only using discretized modality.
    '''

    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Cross tool, and behavior knowledge transfer.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset'],
                        default='Tool_Dataset',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5'],
                        default='ur5',
                        help='robot name')
    parser.add_argument('-feature',
                        choices=['discretized-10-bins'],
                        default='discretized-10-bins',
                        help='feature type')
    parser.add_argument('-classifier-name',
                        choices=['SVM-RBF', 'SVM-LIN', 'KNN', 'DT', 'RF', 'AB', 'GN'],
                        default='SVM-RBF',
                        help='classifier')
    parser.add_argument('-num-folds',
                        default=10,
                        type=int,
                        help='number of folds')
    parser.add_argument('-increment-train-objects',
                        action='store_true',
                        help='increment train objects')
    parser.add_argument('-across',
                        choices=['tools', 'behaviors', 'tools_behaviors'],
                        default='tools',
                        help='transfer across')
    parser.add_argument('-augment-trials',
                        default=0,
                        type=int,
                        help='number of trials to augment')
    args = parser.parse_args()

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    recognition_task = 'object'

    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_path = 'results' + os.sep + 'transfer_' + args.feature + os.sep + 'across_' + args.across + \
                   ('_aug_' + str(args.augment_trials) + '_trials' if args.augment_trials else '') + os.sep
    os.makedirs(results_path, exist_ok=True)

    MATLAB_eng = matlab.engine.start_matlab()
    MATLAB_eng.cd(r'kema', nargout=0)

    data_path_KEMA = os.getcwd() + os.sep + results_path + os.sep + 'KEMA_data' + os.sep + time_stamp
    input_filename_KEMA = 'data_' + args.across + '.mat'
    output_filename_KEMA = 'projections_' + args.across + '.mat'
    os.makedirs(data_path_KEMA, exist_ok=True)

    log_path_name = results_path + time_stamp + '.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path_name), logging.StreamHandler()])
    logging.info("args: {}".format(args))

    # Writing log file for execution time
    time_log_path_name = results_path + time_stamp + '_time_log.txt'
    with open(time_log_path_name, 'w') as time_log_file:
        time_log_file.write('Time Log\n')
        main_start_time = time.time()

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    config.update(vars(args))
    logging.info('config: {}'.format(config))

    # robots = list(config['Tool_Dataset'].keys())
    behaviors = config['behaviors']
    modalities = config['modalities']
    # logging.info('robots: {}'.format(robots))
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))

    data_file_path = os.sep.join([r'data', args.dataset + '_Binary', args.robot, 'dataset_metadata.bin'])
    bin_file = open(data_file_path, 'rb')
    metadata = pickle.load(bin_file)
    bin_file.close()

    objects = metadata[behaviors[0]]['objects']
    tools = metadata[behaviors[0]]['tools']
    trials = metadata[behaviors[0]]['trials']
    logging.info('objects: {}'.format(objects))
    logging.info('tools: {}'.format(tools))
    logging.info('trials: {}'.format(trials))

    classes_labels = get_classes_labels(metadata[behaviors[0]][recognition_task + 's'])
    logging.info('classes_labels: {}'.format(classes_labels))

    objects_list = list(classes_labels.keys())

    dist_data = {}
    for behavior in behaviors:
        dist_data.setdefault(behavior, {})
        for tool in tools:
            dist_data[behavior].setdefault(tool, {})
            for modality in modalities:
                dist_data[behavior][tool].setdefault(modality, {})

                for obj in objects_list:
                    dist_data[behavior][tool][modality].setdefault(obj, {})

                    # logging.info('behavior: {}'.format(behavior))
                    # logging.info('tool: {}'.format(tool))
                    # logging.info('modality: {}'.format(modality))

                    t_train_data, t_train_y = get_split_data_objects(binary_dataset_path, trials, classes_labels,
                                                                     args.robot, behavior, modality, tool,
                                                                     [obj], args.feature)
                    # logging.info('t_train_data: {}'.format(t_train_data.shape))
                    # logging.info('t_train_y: {}, {}'.format(t_train_y.shape, t_train_y.flatten()[0:15]))

                    dist_data[behavior][tool][modality][obj]['X'] = t_train_data
                    dist_data[behavior][tool][modality][obj]['Y'] = t_train_y

    db_file_name = 'dist_data.bin'
    output_file = open(db_file_name, 'wb')
    pickle.dump(dist_data, output_file)
    output_file.close()
