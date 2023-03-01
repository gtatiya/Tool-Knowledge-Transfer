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

    binary_dataset_path = r'data' + os.sep
    recognition_task = 'object'

    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_path = 'results' + os.sep + 'across_' + args.across + \
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

    behaviors = config['behaviors']
    modalities = config['modalities']
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))

    data_file_path = os.sep.join([r'data', 'dataset_metadata.bin'])
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

    clf = get_classifier(args.classifier_name)

    folds_objects_split_all = get_config(r'configs' + os.sep + 'objects_splits.yaml')

    folds_objects_split = {'fold_' + str(object_fold): folds_objects_split_all['fold_' + str(object_fold)]
                           for object_fold in range(args.num_folds)}
    logging.info('folds_objects_split: {}'.format(folds_objects_split))

    num_of_test_examples = 2
    folds = len(trials) // num_of_test_examples
    folds_trials_split = split_train_test_trials(folds, len(trials))
    logging.info('folds_trials_split: {}'.format(folds_trials_split))

    num_objects_list = np.arange(len(folds_objects_split['fold_0']['train'])) + 1 \
        if args.increment_train_objects else [len(folds_objects_split['fold_0']['train'])]
    logging.info('num_objects_list: {}'.format(num_objects_list))

    projections = []
    for source_behavior in sorted(behaviors):
        for source_tool in sorted(tools):
            for target_behavior in sorted(behaviors):
                for target_tool in sorted(tools):
                    # Both behaviors and tools cannot be the same
                    if (source_behavior == target_behavior) and (source_tool == target_tool):
                        continue
                    # Either behavior or tool needs to be the same
                    if args.across == 'tools' and source_behavior == target_behavior and source_tool != target_tool:
                        projections.append({'source_behavior': source_behavior, 'source_tool': source_tool,
                                            'target_behavior': target_behavior, 'target_tool': target_tool})
                    elif args.across == 'behaviors' and source_behavior != target_behavior and source_tool == target_tool:
                        projections.append({'source_behavior': source_behavior, 'source_tool': source_tool,
                                            'target_behavior': target_behavior, 'target_tool': target_tool})
                    elif args.across == 'tools_behaviors':
                        projections.append({'source_behavior': source_behavior, 'source_tool': source_tool,
                                            'target_behavior': target_behavior, 'target_tool': target_tool})
    logging.info('projections: {}'.format(len(projections)))

    for p_i, projection in enumerate(projections):
        folds_proba_score_bl = {}
        folds_proba_score_bl2 = {}
        folds_proba_score_kt = {}

        source_behavior = projection['source_behavior']
        source_tool = projection['source_tool']
        target_behavior = projection['target_behavior']
        target_tool = projection['target_tool']

        projection_path = results_path + '_'.join([source_behavior, source_tool, 'TO',
                                                   target_behavior, target_tool]) + '_' + args.classifier_name + os.sep

        if os.path.exists(projection_path):
            logging.info('projection already exists: {}'.format(projection_path))
            continue
        else:
            os.makedirs(projection_path, exist_ok=True)

        for object_fold in sorted(folds_objects_split):
            folds_proba_score_bl.setdefault(object_fold, {})
            folds_proba_score_bl2.setdefault(object_fold, {})
            folds_proba_score_kt.setdefault(object_fold, {})

            for num_obj in num_objects_list:
                folds_proba_score_bl[object_fold].setdefault(num_obj, {})
                folds_proba_score_bl2[object_fold].setdefault(num_obj, {})
                folds_proba_score_kt[object_fold].setdefault(num_obj, {})

                for modality in modalities:
                    folds_proba_score_bl[object_fold][num_obj].setdefault(modality, {})
                    folds_proba_score_bl2[object_fold][num_obj].setdefault(modality, {})
                    folds_proba_score_kt[object_fold][num_obj].setdefault(modality, {})

                    logging.info('source_behavior: {}'.format(source_behavior))
                    logging.info('source_tool: {}'.format(source_tool))
                    logging.info('target_behavior: {}'.format(target_behavior))
                    logging.info('target_tool: {}'.format(target_tool))
                    logging.info('object_fold: {}'.format(object_fold))
                    logging.info('num_obj: {}'.format(num_obj))
                    logging.info('modality: {}'.format(modality))

                    # Get target train objects data
                    t_train_data, t_train_y = get_split_data_objects(binary_dataset_path, trials, target_behavior,
                                                                     modality, target_tool,
                                                                     folds_objects_split[object_fold]['train'][0:num_obj],
                                                                     )
                    logging.info('t_train_data: {}'.format(t_train_data.shape))
                    logging.info('t_train_y: {}, {}'.format(t_train_y.shape, t_train_y.flatten()[0:15]))

                    if args.augment_trials:
                        t_train_data, t_train_y = augment_trials(t_train_data, t_train_y, args.augment_trials)
                        logging.info('After Data Augmentations:')
                        logging.info('t_train_data: {}'.format(t_train_data.shape))
                        logging.info('t_train_y: {}, {}'.format(t_train_y.shape, t_train_y.flatten()[0:15]))

                    # Get target test objects data
                    t_test_data, t_test_y = get_split_data_objects(binary_dataset_path, trials, target_behavior,
                                                                   modality, target_tool,
                                                                   folds_objects_split[object_fold]['test'])
                    logging.info('t_test_data: {}'.format(t_test_data.shape))
                    logging.info('t_test_y: {}, {}'.format(t_test_y.shape, t_test_y.flatten()))

                    t_test_y2, objects_labels_target, old_labels_new_label = get_new_labels(t_test_y, classes_labels)
                    logging.info('t_test_y2: {}, {}'.format(t_test_y2.shape, t_test_y2.flatten()))
                    logging.info('objects_labels_target: {}'.format(objects_labels_target))
                    logging.info('old_labels_new_label: {}'.format(old_labels_new_label))

                    # Reshaping target test data to access trials for training and testing
                    t_test_data2 = t_test_data.reshape(len(folds_objects_split['fold_0']['test']), -1, t_test_data.shape[-1])
                    t_test_y2 = t_test_y2.reshape(len(folds_objects_split['fold_0']['test']), -1, t_test_y2.shape[-1])
                    logging.info('t_test_data2: {}'.format(t_test_data2.shape))
                    logging.info('t_test_y2: {}, {}'.format(t_test_y2.shape, t_test_y2.flatten()))

                    # Get source data of all objects to train the projection function
                    objects_list = list(classes_labels.keys())
                    s_data, s_y = get_split_data_objects(binary_dataset_path, trials, source_behavior, modality,
                                                         source_tool, objects_list)
                    logging.info('s_data: {}'.format(s_data.shape))
                    logging.info('s_y: {}, {}'.format(s_y.shape, s_y.flatten()[0:15]))

                    if args.augment_trials:
                        s_data, s_y = augment_trials(s_data, s_y, args.augment_trials, list(old_labels_new_label.keys()))
                        logging.info('After Data Augmentations:')
                        logging.info('s_data: {}'.format(s_data.shape))
                        logging.info('s_y: {}, {}'.format(s_y.shape, s_y.flatten()[0:15]))

                    # Get source test objects data for baseline 2
                    s_test_data, s_test_y = get_split_data_objects(binary_dataset_path, trials, source_behavior,
                                                                   modality, source_tool,
                                                                   folds_objects_split[object_fold]['test'])
                    logging.info('s_test_data: {}'.format(s_test_data.shape))
                    logging.info('s_test_y: {}, {}'.format(s_test_y.shape, s_test_y.flatten()))

                    # Reshaping source test data to access trials for training
                    s_test_data2 = s_test_data.reshape(len(folds_objects_split['fold_0']['test']), -1,
                                                       s_test_data.shape[-1])
                    logging.info('s_test_data2: {}'.format(s_test_data2.shape))

                    # Transfer Robot Knowledge
                    start_time = time.time()
                    # KEMA
                    KEMA_data = {'X2': t_train_data, 'X2_Test': t_test_data}
                    KEMA_data['Y2'] = t_train_y + 1  # add 1 as in KEMA (MATLAB) labels starts from 1

                    KEMA_data['X1'] = s_data
                    KEMA_data['Y1'] = s_y + 1  # add 1 as in KEMA (MATLAB) labels starts from 1

                    KEMA_data = check_kema_data(KEMA_data)
                    scipy.io.savemat(os.path.join(data_path_KEMA, input_filename_KEMA), mdict=KEMA_data)
                    MATLAB_eng.project2Domains_v2(data_path_KEMA, input_filename_KEMA, output_filename_KEMA, 1)

                    # In case Matlab messes up, we'll load and check these immediately, then delete them so we never read in an old file
                    projections = None
                    if os.path.isfile(os.path.join(data_path_KEMA, output_filename_KEMA)):
                        try:
                            projections = loadmat(os.path.join(data_path_KEMA, output_filename_KEMA))
                            Z1_train, Z2_train, Z2_Test = projections['Z1'], projections['Z2'], projections['Z2_Test']
                            os.remove(os.path.join(data_path_KEMA, output_filename_KEMA))
                            os.remove(os.path.join(data_path_KEMA, input_filename_KEMA))
                        except TypeError as e:
                            logging.info('loadmat failed: {}'.format(e))

                        if not np.isreal(Z1_train).all():
                            logging.info('Complex number detected in Z1_train')
                            Z1_train = Z1_train.real

                        if not np.isreal(Z2_Test).all():
                            logging.info('Complex number detected in Z2_Test')
                            Z2_Test = Z2_Test.real

                    logging.info('Z1_train: {}'.format(Z1_train.shape))
                    logging.info('Z2_train: {}'.format(Z2_train.shape))
                    logging.info('Z2_Test: {}'.format(Z2_Test.shape))

                    # Getting test objects data projected by source
                    Z1_train2 = []
                    s_train_y2 = []
                    for old_label in old_labels_new_label:
                        indices = np.where(s_y == old_label)
                        examples = Z1_train[indices[0]]
                        examples = examples.reshape(-1, examples.shape[-1])
                        Z1_train2.extend(examples)
                        label = old_labels_new_label[old_label]
                        s_train_y2.extend(np.repeat(label, len(examples)))
                    Z1_train2 = np.array(Z1_train2)
                    s_train_y2 = np.array(s_train_y2).reshape((-1, 1))
                    logging.info('Z1_train2: {}'.format(Z1_train2.shape))
                    logging.info('s_train_y2: {}, {}'.format(s_train_y2.shape, s_train_y2.flatten()))

                    # Randomly chosing a trial fold for training and testing model (baselines and transfer)
                    trial_fold = np.random.choice(list(folds_trials_split.keys()))
                    logging.info('{}: {}'.format(trial_fold, folds_trials_split[trial_fold]))

                    projection_log_path = projection_path + object_fold + os.sep + str(num_obj) + os.sep + modality + \
                                          os.sep
                    os.makedirs(projection_log_path, exist_ok=True)

                    results = []
                    with open(projection_log_path + os.sep + 'results.csv', 'w') as f:
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(['Baseline 1 (Test Accuracy)', 'Baseline 1 (Train Accuracy)',
                                         'Baseline 2 (Test Accuracy)', 'Baseline 2 (Train Accuracy)',
                                         'Transfer (Test Accuracy)', 'Transfer (Train Accuracy)'])

                    # Baseline 1 (target learns using target context original data - best case)
                    logging.info('Baseline 1 (target learns using target context original data - best case)')
                    # Get train data
                    t_train_data2 = t_test_data2[:, folds_trials_split[trial_fold]['train']].reshape((-1, t_test_data2.shape[-1]))
                    t_train_y2 = t_test_y2[:, folds_trials_split[trial_fold]['train']].reshape((-1, 1))
                    logging.info('t_train_data2: {}'.format(t_train_data2.shape))
                    logging.info('t_train_y2: {}, {}'.format(t_train_y2.shape, t_train_y2.flatten()))

                    # Get test data
                    t_test_data2_ = t_test_data2[:, folds_trials_split[trial_fold]['test']].reshape((-1, t_test_data2.shape[-1]))
                    t_test_y2_ = t_test_y2[:, folds_trials_split[trial_fold]['test']].reshape((-1, 1))
                    logging.info('t_test_data2_: {}'.format(t_test_data2_.shape))
                    logging.info('t_test_y2_: {}, {}'.format(t_test_y2_.shape, t_test_y2_.flatten()))

                    # Train and Test
                    y_acc, y_pred, y_proba = classifier(clf, t_train_data2, t_test_data2_, t_train_y2.ravel(),
                                                        t_test_y2_.ravel())
                    logging.info('y_pred: {}, {}, {}'.format(y_pred.shape, y_pred.flatten()[0:10], t_test_y2_.flatten()[0:10]))
                    logging.info('y_acc: {}'.format(y_acc))
                    results.append(y_acc)

                    folds_proba_score_bl[object_fold][num_obj][modality]['proba'] = y_proba
                    folds_proba_score_bl[object_fold][num_obj][modality]['test_acc'] = y_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(clf, t_train_data2, t_train_data2,
                                                                          t_train_y2.ravel(), t_train_y2.ravel())
                    logging.info('y_pred_train: {}, {}, {}'.format(y_pred_train.shape, y_pred_train.flatten()[0:10], t_train_y2.flatten()[0:10]))
                    logging.info('y_acc_train: {}'.format(y_acc_train))
                    results.append(y_acc_train)

                    folds_proba_score_bl[object_fold][num_obj][modality]['train_acc'] = y_acc_train

                    # Baseline 2 (target learns using source context original data - worst case)
                    logging.info('Baseline 2 (target learns using source context original data - worst case)')
                    # Get train data
                    s_train_data2 = s_test_data2[:, folds_trials_split[trial_fold]['train']].reshape(
                        (-1, s_test_data2.shape[-1]))
                    logging.info('s_train_data2: {}'.format(s_train_data2.shape))

                    # Train and Test
                    y_acc, y_pred, y_proba = classifier(clf, s_train_data2, t_test_data2_, t_train_y2.ravel(),
                                                        t_test_y2_.ravel())
                    logging.info('y_pred: {}, {}, {}'.format(y_pred.shape, y_pred.flatten()[0:10], t_test_y2_.flatten()[0:10]))
                    logging.info('y_acc: {}'.format(y_acc))
                    results.append(y_acc)

                    folds_proba_score_bl2[object_fold][num_obj][modality]['proba'] = y_proba
                    folds_proba_score_bl2[object_fold][num_obj][modality]['test_acc'] = y_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(clf, s_train_data2, s_train_data2,
                                                                          t_train_y2.ravel(), t_train_y2.ravel())
                    logging.info('y_pred_train: {}, {}, {}'.format(y_pred_train.shape, y_pred_train.flatten()[0:10], t_train_y2.flatten()[0:10]))
                    logging.info('y_acc_train: {}'.format(y_acc_train))
                    results.append(y_acc_train)

                    folds_proba_score_bl2[object_fold][num_obj][modality]['train_acc'] = y_acc_train

                    # Transfer (target learns using projected source data)
                    logging.info('Transfer (target learns using projected source data)')
                    # KEMA
                    # Reshaping Z2_Test to access trials for testing
                    Z2_Test2 = Z2_Test.reshape(len(folds_objects_split['fold_0']['test']), -1, Z2_Test.shape[-1])
                    logging.info('Z2_Test2: {}'.format(Z2_Test2.shape))

                    # Get test data
                    Z2_Test2 = Z2_Test2[:, folds_trials_split[trial_fold]['test']].reshape((-1, Z2_Test2.shape[-1]))
                    logging.info('Z2_Test2: {}'.format(Z2_Test2.shape))

                    y_acc, y_pred, y_proba = classifier(clf, Z1_train2, Z2_Test2, s_train_y2.ravel(),
                                                        t_test_y2_.ravel())
                    logging.info('y_pred: {}, {}, {}'.format(y_pred.shape, y_pred.flatten()[0:10], t_test_y2_.flatten()[0:10]))
                    logging.info('y_acc: {}'.format(y_acc))
                    results.append(y_acc)

                    folds_proba_score_kt[object_fold][num_obj][modality]['proba'] = y_proba
                    folds_proba_score_kt[object_fold][num_obj][modality]['test_acc'] = y_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(clf, Z1_train2, Z1_train2,
                                                                          s_train_y2.ravel(), s_train_y2.ravel())
                    logging.info('y_pred_train: {}, {}, {}'.format(y_pred_train.shape, y_pred_train.flatten()[0:10], s_train_y2.flatten()[0:10]))
                    logging.info('y_acc_train: {}'.format(y_acc_train))
                    results.append(y_acc_train)

                    folds_proba_score_kt[object_fold][num_obj][modality]['train_acc'] = y_acc_train

                    with open(projection_log_path + os.sep + 'results.csv', 'a') as f:  # append to the file created
                        writer = csv.writer(f, lineterminator='\n')
                        writer.writerow(results)

                    # Writing log file for execution time
                    file = open(time_log_path_name, 'a')  # append to the file created
                    end_time = time.time()
                    file.write('\n\n' + '_'.join([source_behavior, source_tool, 'TO', target_behavior, target_tool])
                               + '_' + args.classifier_name + os.sep + object_fold + os.sep + str(num_obj) + os.sep +
                               modality + os.sep)
                    file.write('\nTime: ' + time_taken(start_time, end_time))
                    file.write('\nTotal Time: ' + time_taken(main_start_time, end_time))
                    file.close()

                # End of `for modality in modalities` loop: combining all the modalities
                folds_proba_score_bl[object_fold][num_obj] = update_all_modalities(
                    folds_proba_score_bl[object_fold][num_obj], t_test_y2_)
                folds_proba_score_bl2[object_fold][num_obj] = update_all_modalities(
                    folds_proba_score_bl2[object_fold][num_obj], t_test_y2_)
                folds_proba_score_kt[object_fold][num_obj] = update_all_modalities(
                    folds_proba_score_kt[object_fold][num_obj], t_test_y2_)

            # End of `for num_obj in num_objects_list` loop: plot accuracy curve of all modality for each num_obj
            title_name = ' '.join([source_behavior, source_tool, 'TO', target_behavior, target_tool]) + \
                         '\n(All modalities combined - ' + object_fold + ')'
            xlabel = 'Number of Shared Objects'
            file_path = results_path + '_'.join([source_behavior, source_tool, 'TO',
                                                           target_behavior, target_tool]) + '_' + \
                                  args.classifier_name + os.sep + object_fold + os.sep

            plot_fold_all_modalities(folds_proba_score_bl[object_fold], folds_proba_score_bl2[object_fold],
                                     folds_proba_score_kt[object_fold], 'all_modalities', title_name, xlabel,
                                     file_path)
            plot_fold_all_modalities(folds_proba_score_bl[object_fold], folds_proba_score_bl2[object_fold],
                                     folds_proba_score_kt[object_fold], 'all_modalities_train', title_name, xlabel,
                                     file_path)
            plot_fold_all_modalities(folds_proba_score_bl[object_fold], folds_proba_score_bl2[object_fold],
                                     folds_proba_score_kt[object_fold], 'all_modalities_test', title_name, xlabel,
                                     file_path)

        # End of `for object_fold in sorted(folds_objects_split)` loop:
        # compute mean accuracy of each split in folds_objects_split
        # plot accuracy curve of all folds_objects_split for each num_obj
        folds_proba_score_bl = compute_mean_accuracy(folds_proba_score_bl, behavior_present=False)
        folds_proba_score_bl2 = compute_mean_accuracy(folds_proba_score_bl2, behavior_present=False)
        folds_proba_score_kt = compute_mean_accuracy(folds_proba_score_kt, behavior_present=False)

        # Save results
        all_modalities_type = 'all_modalities_train'
        with open(projection_path + os.sep + 'results_' + all_modalities_type + '.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['Number of Shared Objects', 'Baseline 1 Accuracy', 'Baseline 1 SD', 'Baseline 2 Accuracy',
                             'Baseline 2 SD', 'Transfer Accuracy', 'Transfer SD',
                             'Difference in Accuracy from Baseline 1', 'Difference in Accuracy from Baseline 2'])

            for num_obj in sorted(folds_proba_score_bl):
                logging.info('{}: {}'.format(num_obj, folds_proba_score_bl[num_obj]))
                logging.info('{}: {}'.format(num_obj, folds_proba_score_bl2[num_obj]))
                logging.info('{}: {}'.format(num_obj, folds_proba_score_kt[num_obj]))

                bl1_accuracy = folds_proba_score_bl[num_obj][all_modalities_type]['mean']
                bl2_accuracy = folds_proba_score_bl2[num_obj][all_modalities_type]['mean']
                transfer_accuracy = folds_proba_score_kt[num_obj][all_modalities_type]['mean']
                difference_in_accuracy1 = bl1_accuracy - transfer_accuracy
                difference_in_accuracy2 = bl2_accuracy - transfer_accuracy
                writer.writerow([num_obj, bl1_accuracy, folds_proba_score_bl[num_obj][all_modalities_type]['std'],
                                 bl2_accuracy, folds_proba_score_bl2[num_obj][all_modalities_type]['std'],
                                 transfer_accuracy, folds_proba_score_kt[num_obj][all_modalities_type]['std'],
                                 difference_in_accuracy1, difference_in_accuracy2])

        db_file_name = 'results.bin'
        output_file = open(projection_path + os.sep + db_file_name, 'wb')
        pickle.dump(folds_proba_score_bl, output_file)
        pickle.dump(folds_proba_score_bl2, output_file)
        pickle.dump(folds_proba_score_kt, output_file)
        output_file.close()

        xlabel = 'Number of Shared Objects'

        title_name = ' '.join([source_behavior, source_tool, 'TO', target_behavior, target_tool]) + \
                     '\nIndividual modality (Baseline 1 Condition)'
        plot_each_modality(folds_proba_score_bl, 'each_modality_baseline1', title_name, xlabel, projection_path)

        title_name = ' '.join([source_behavior, source_tool, 'TO', target_behavior, target_tool]) + \
                     '\nIndividual modality (Baseline 2 Condition)'
        plot_each_modality(folds_proba_score_bl2, 'each_modality_baseline2', title_name, xlabel, projection_path)

        title_name = ' '.join([source_behavior, source_tool, 'TO', target_behavior, target_tool]) + \
                     '\nIndividual modality (Transfer Condition)'
        plot_each_modality(folds_proba_score_kt, 'each_modality_transfer', title_name, xlabel, projection_path)

        title_name = ' '.join([source_behavior, source_tool, 'TO', target_behavior, target_tool]) + \
                     '\n(All modalities combined)'
        plot_all_modalities(folds_proba_score_bl, folds_proba_score_bl2, folds_proba_score_kt, 'all_modalities',
                            title_name, xlabel, projection_path, 'all_modalities')
        plot_all_modalities(folds_proba_score_bl, folds_proba_score_bl2, folds_proba_score_kt, 'all_modalities_train',
                            title_name, xlabel, projection_path, 'all_modalities_train')
        plot_all_modalities(folds_proba_score_bl, folds_proba_score_bl2, folds_proba_score_kt, 'all_modalities_test',
                            title_name, xlabel, projection_path, 'all_modalities_test')

    save_config(config, results_path + 'config.yaml')
