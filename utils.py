# Author: Gyan Tatiya
import os
import pickle

import yaml

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def time_taken(start, end):
    """Human-readable time between `start` and `end`
    :param start: time.time()
    :param end: time.time()
    :returns: day:hour:minute:second.millisecond
    """

    my_time = end - start
    day = my_time // (24 * 3600)
    my_time = my_time % (24 * 3600)
    hour = my_time // 3600
    my_time %= 3600
    minutes = my_time // 60
    my_time %= 60
    seconds = my_time
    milliseconds = ((end - start) - int(end - start))
    day_hour_min_sec = str('%02d' % int(day)) + ":" + str('%02d' % int(hour)) + ":" + str('%02d' % int(minutes)) + \
                       ':' + str('%02d' % int(seconds) + "." + str('%.3f' % milliseconds)[2:])

    return day_hour_min_sec


def fix_names(names):

    names = list(names)
    for i, name in enumerate(names):
        if name in ['1-look', '2-stirring-slow', '3-stirring-fast', '4-stirring-twist', '5-whisk', '6-poke']:
            names[i] = '-'.join([x.capitalize() for x in name[2:].split('-')])
        elif name in ['plastic-knife', 'metal-whisk', 'wooden-chopstick', 'plastic-spoon', 'metal-scissor',
                      'wooden-fork']:
            names[i] = '-'.join([x.capitalize() for x in name.split('-')])
        elif name in ['camera_depth_image', 'camera_rgb_image', 'touch_image', 'audio', 'gripper_joint_states',
                      'effort', 'position', 'velocity', 'torque', 'force']:
            if 'depth' in name:
                names[i] = 'Depth-Image'
            elif 'rgb' in name:
                names[i] = 'RGB-Image'
            elif 'gripper' in name:
                names[i] = 'Gripper'
            else:
                names[i] = name.capitalize()

    return names


def get_config(config_path):
    with open(config_path) as file:
        return yaml.safe_load(file)


def save_config(config, config_filepath):
    with open(config_filepath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_classes_labels(objects_list):

    classes_labels_ = {}
    for i, object_name in enumerate(sorted(objects_list)):
        classes_labels_[object_name] = i

    return classes_labels_


def get_classifier(name):

    if name == 'SVM-RBF':
        clf = SVC(gamma='auto', kernel='rbf', probability=True)
    elif name == 'SVM-LIN':
        clf = SVC(gamma='auto', kernel='linear', probability=True)
    elif name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=1)
    elif name == 'DT':
        clf = DecisionTreeClassifier()
    elif name == 'RF':
        clf = RandomForestClassifier()
    elif name == 'AB':
        clf = AdaBoostClassifier()
    elif name == 'GN':
        clf = GaussianNB()
    else:
        raise Exception(name + ' does not exits!')

    return clf


def split_train_test_trials(n_folds, trials_per_class):

    test_size = trials_per_class // n_folds
    tt_splits = {}

    for a_fold in range(n_folds):

        train_index = []
        test_index = np.arange(test_size * a_fold, test_size * (a_fold + 1))

        if test_size * a_fold > 0:
            train_index.extend(np.arange(0, test_size * a_fold))
        if test_size * (a_fold + 1) - 1 < trials_per_class - 1:
            train_index.extend(np.arange(test_size * (a_fold + 1), trials_per_class))

        tt_splits.setdefault('fold_' + str(a_fold), {}).setdefault('train', []).extend(train_index)
        tt_splits.setdefault('fold_' + str(a_fold), {}).setdefault('test', []).extend(test_index)

    return tt_splits


def get_split_data_objects(path, trials, behavior, modality, tool, objects):

    trials = [int(trial.split('-')[1]) for trial in sorted(trials)]

    trial_data_filepath = os.sep.join([path, 'dataset_discretized.bin'])
    bin_file = open(trial_data_filepath, 'rb')
    dataset = pickle.load(bin_file)
    bin_file.close()

    x_split = []
    y_split = []
    for object_name in sorted(objects):
        for trial_num in sorted(trials):
            x_split.append(dataset[behavior][tool][modality][object_name]['X'][trial_num])
            y_split.append(dataset[behavior][tool][modality][object_name]['Y'][trial_num])

    return np.array(x_split), np.array(y_split)


def augment_trials(X_data, y_object, num_trials_aug=5, object_labels=None, shuffle=True):
    # If object_labels is give then only augment these labels

    if object_labels is None:
        object_labels = set(y_object.flatten())

    X_data_aug = []
    y_object_aug = []
    for label in object_labels:
        indices = np.where(y_object == label)

        X_data_mean = np.mean(X_data[indices[0]], axis=0)
        X_data_std = np.std(X_data[indices[0]], axis=0)

        for _ in range(num_trials_aug):
            data_point = np.random.normal(X_data_mean, X_data_std)
            X_data_aug.append(data_point)
            y_object_aug.append(label)

    X_data_aug = np.array(X_data_aug)
    y_object_aug = np.array(y_object_aug).reshape((-1, 1))

    if len(X_data_aug) > 0:
        X_data = np.concatenate((X_data, X_data_aug), axis=0)
        y_object = np.concatenate((y_object, y_object_aug), axis=0)

    if shuffle:
        random_idx = np.random.permutation(X_data.shape[0])
        X_data = X_data[random_idx]
        y_object = y_object[random_idx]

    return X_data, y_object


def get_new_labels(y_object, objects_labels):

    label_count = 0
    y_object_new = []
    old_labels_new_label = {}
    objects_labels_new = {}
    for old_label in y_object.flatten():
        if old_label not in old_labels_new_label:
            old_labels_new_label[old_label] = label_count
            y_object_new.append(label_count)
            object_name_ = list(objects_labels.keys())[list(objects_labels.values()).index(old_label)]
            objects_labels_new[object_name_] = label_count
            label_count += 1
        else:
            y_object_new.append(old_labels_new_label[old_label])
    y_object_new = np.array(y_object_new).reshape((-1, 1))

    return y_object_new, objects_labels_new, old_labels_new_label


def check_kema_data(kema_data):

    for x_key in kema_data:
        if 'Test' not in x_key and x_key.startswith('X') and kema_data[x_key].shape[0] <= 10:
            y_key = 'Y' + x_key[1]
            print('<= 10 EXAMPLES FOR: ', x_key, y_key)

            while kema_data[x_key].shape[0] <= 10:
                idx = np.random.choice(kema_data[x_key].shape[0])
                kema_data[x_key] = np.append(kema_data[x_key], kema_data[x_key][idx].reshape(1, -1), axis=0)
                kema_data[y_key] = np.append(kema_data[y_key], kema_data[y_key][idx].reshape(1, -1), axis=0)

    return kema_data


def classifier(my_classifier, x_train, x_test, y_train, y_test):
    # Train a classifier on test data and return accuracy and prediction on test data

    # Fit the model on the training data
    my_classifier.fit(x_train, y_train.ravel())

    # See how the model performs on the test data
    probability = my_classifier.predict_proba(x_test)

    prediction = np.argmax(probability, axis=1)
    accuracy = np.mean(y_test.ravel() == prediction)

    return accuracy, prediction, probability


def combine_probability(proba_acc_list_, y_test_, acc=None):
    # For each classifier, combine weighted probability based on its accuracy score
    proba_list = []
    for proba_acc in proba_acc_list_:
        y_proba = proba_acc['proba']
        if acc and proba_acc[acc] > 0:
            # Multiply the score by probability to combine each classifier's performance accordingly
            # IMPORTANT: This will discard probability when the accuracy is 0
            y_proba = y_proba * proba_acc[acc]  # weighted probability
            proba_list.append(y_proba)
        elif not acc:
            proba_list.append(y_proba)  # Uniform combination, probability is combined even when the accuracy is 0

    # If all the accuracy is 0 in proba_acc_list_, the fill proba_list with chance accuracy
    if len(proba_list) == 0:
        num_examples, num_classes = proba_acc_list_[0]['proba'].shape
        chance_prob = (100 / num_classes) / 100
        proba_list = np.full((1, num_examples, num_classes), chance_prob)

    # Combine weighted probability of all classifiers
    y_proba_norm = np.zeros(len(proba_list[0][0]))
    for proba in proba_list:
        y_proba_norm = y_proba_norm + proba

    # Normalizing probability
    y_proba_norm_sum = np.sum(y_proba_norm, axis=1)  # sum of weighted probability
    y_proba_norm_sum = np.repeat(y_proba_norm_sum, len(proba_list[0][0]), axis=0).reshape(y_proba_norm.shape)
    y_proba_norm = y_proba_norm / y_proba_norm_sum

    y_proba_pred = np.argmax(y_proba_norm, axis=1)
    y_prob_acc = np.mean(y_test_ == y_proba_pred)

    return y_proba_norm, y_prob_acc


def update_all_modalities(modalities_proba_score, y_test_):
    # For each modality, combine weighted probability based on its accuracy score

    proba_acc_list = []
    for modality_ in modalities_proba_score:
        proba_acc = {'proba': modalities_proba_score[modality_]['proba'],
                     'train_acc': modalities_proba_score[modality_]['train_acc'],
                     'test_acc': modalities_proba_score[modality_]['test_acc']}
        proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    modalities_proba_score.setdefault('all_modalities', {})
    modalities_proba_score['all_modalities']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    modalities_proba_score.setdefault('all_modalities_train', {})
    modalities_proba_score['all_modalities_train']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    modalities_proba_score.setdefault('all_modalities_test', {})
    modalities_proba_score['all_modalities_test']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_test']['test_acc'] = y_prob_acc

    return modalities_proba_score


def compute_mean_accuracy(folds_behaviors_modalities_proba_score, acc='test_acc', vary_objects=True,
                          behavior_present=True):

    behaviors_modalities_score = {}
    for fold_ in folds_behaviors_modalities_proba_score:
        if vary_objects:
            for objects_per_label_ in folds_behaviors_modalities_proba_score[fold_]:
                behaviors_modalities_score.setdefault(objects_per_label_, {})
                if behavior_present:
                    for behavior_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                        if behavior_.startswith('all_behaviors_modalities'):
                            behaviors_modalities_score[objects_per_label_].setdefault(behavior_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][acc]
                            behaviors_modalities_score[objects_per_label_][behavior_].append(y_prob_acc)
                        else:
                            behaviors_modalities_score[objects_per_label_].setdefault(behavior_, {})
                            for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_]:
                                behaviors_modalities_score[objects_per_label_][behavior_].setdefault(modality_, [])
                                y_prob_acc = \
                                    folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][modality_][
                                        acc]
                                behaviors_modalities_score[objects_per_label_][behavior_][modality_].append(y_prob_acc)
                else:
                    for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                        behaviors_modalities_score[objects_per_label_].setdefault(modality_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][modality_][acc]
                        behaviors_modalities_score[objects_per_label_][modality_].append(y_prob_acc)
        else:
            if behavior_present:
                for behavior_ in folds_behaviors_modalities_proba_score[fold_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score.setdefault(behavior_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][acc]
                        behaviors_modalities_score[behavior_].append(y_prob_acc)
                    else:
                        behaviors_modalities_score.setdefault(behavior_, {})
                        for modality_ in folds_behaviors_modalities_proba_score[fold_][behavior_]:
                            behaviors_modalities_score[behavior_].setdefault(modality_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][modality_][acc]
                            behaviors_modalities_score[behavior_][modality_].append(y_prob_acc)
            else:
                for modality_ in folds_behaviors_modalities_proba_score[fold_]:
                    behaviors_modalities_score.setdefault(modality_, [])
                    y_prob_acc = folds_behaviors_modalities_proba_score[fold_][modality_][acc]
                    behaviors_modalities_score[modality_].append(y_prob_acc)

    if vary_objects:
        for objects_per_label_ in behaviors_modalities_score:
            if behavior_present:
                for behavior_ in behaviors_modalities_score[objects_per_label_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score[objects_per_label_][behavior_] = {
                            'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_]),
                            'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_])}
                    else:
                        for modality_ in behaviors_modalities_score[objects_per_label_][behavior_]:
                            behaviors_modalities_score[objects_per_label_][behavior_][modality_] = {
                                'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_][modality_]),
                                'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_][modality_])}
            else:
                for modality_ in behaviors_modalities_score[objects_per_label_]:
                    behaviors_modalities_score[objects_per_label_][modality_] = {
                        'mean': np.mean(behaviors_modalities_score[objects_per_label_][modality_]),
                        'std': np.std(behaviors_modalities_score[objects_per_label_][modality_])}
    else:
        if behavior_present:
            for behavior_ in behaviors_modalities_score:
                if behavior_.startswith('all_behaviors_modalities'):
                    behaviors_modalities_score[behavior_] = {
                        'mean': np.mean(behaviors_modalities_score[behavior_]),
                        'std': np.std(behaviors_modalities_score[behavior_])}
                else:
                    for modality_ in behaviors_modalities_score[behavior_]:
                        behaviors_modalities_score[behavior_][modality_] = {
                            'mean': np.mean(behaviors_modalities_score[behavior_][modality_]),
                            'std': np.std(behaviors_modalities_score[behavior_][modality_])}
        else:
            for modality_ in behaviors_modalities_score:
                behaviors_modalities_score[modality_] = {
                    'mean': np.mean(behaviors_modalities_score[modality_]),
                    'std': np.std(behaviors_modalities_score[modality_])}

    return behaviors_modalities_score


def plot_fold_all_modalities(folds_proba_score_bl, folds_proba_score_bl2, folds_proba_score_kt, all_modalities_type,
                             title_name, xlabel, file_path, ylim=True, xticks=True):

    acc_bl = []
    acc_bl2 = []
    acc_kt = []
    x_points = []
    for num_obj in folds_proba_score_bl:
        x_points.append(num_obj)
        acc_bl.append(folds_proba_score_bl[num_obj][all_modalities_type]['test_acc'])
        acc_bl2.append(folds_proba_score_bl2[num_obj][all_modalities_type]['test_acc'])
        acc_kt.append(folds_proba_score_kt[num_obj][all_modalities_type]['test_acc'])
    acc_bl = np.array(acc_bl) * 100
    acc_bl2 = np.array(acc_bl2) * 100
    acc_kt = np.array(acc_kt) * 100

    plt.plot(x_points, acc_bl, color='pink', label='Baseline Condition')
    plt.plot(x_points, acc_bl2, color='red', label='Baseline 2 Condition')
    plt.plot(x_points, acc_kt, color='blue', label='Transfer Condition')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='lower right')
    plt.savefig(file_path + all_modalities_type + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_each_modality(modalities_score, filename, title_name, xlabel, file_path, ylim=True, xticks=True):

    all_scores = {}
    x_points = []
    for num_obj in sorted(modalities_score):
        x_points.append(num_obj)
        for modality in modalities_score[num_obj]:
            all_scores.setdefault(modality, {'mean': [], 'std': []})
            all_scores[modality]['mean'].append(modalities_score[num_obj][modality]['mean'])
            all_scores[modality]['std'].append(modalities_score[num_obj][modality]['std'])

    for modality in sorted(all_scores):
        all_scores[modality]['mean'] = np.array(all_scores[modality]['mean']) * 100
        all_scores[modality]['std'] = np.array(all_scores[modality]['std']) * 100
        plt.plot(x_points, all_scores[modality]['mean'], label=modality.capitalize())
        plt.fill_between(x_points, all_scores[modality]['mean'] - all_scores[modality]['std'],
                         all_scores[modality]['mean'] + all_scores[modality]['std'], alpha=0.3)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='upper left')
    plt.savefig(file_path + filename + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_all_modalities(modalities_score_bl, modalities_score_bl2, modalities_score_kt, all_modalities_type,
                        title_name, xlabel, file_path, filename, ylim=True, xticks=True, errorbar=False,
                        plot_bl2=True):

    acc_bl = []
    std_bl = []
    acc_bl2 = []
    std_bl2 = []
    acc_kt = []
    std_kt = []
    x_points = []
    for num_obj in sorted(modalities_score_bl):
        x_points.append(num_obj)
        acc_bl.append(modalities_score_bl[num_obj][all_modalities_type]['mean'])
        std_bl.append(modalities_score_bl[num_obj][all_modalities_type]['std'])
        acc_bl2.append(modalities_score_bl2[num_obj][all_modalities_type]['mean'])
        std_bl2.append(modalities_score_bl2[num_obj][all_modalities_type]['std'])
        acc_kt.append(modalities_score_kt[num_obj][all_modalities_type]['mean'])
        std_kt.append(modalities_score_kt[num_obj][all_modalities_type]['std'])
    acc_bl = np.array(acc_bl) * 100
    std_bl = np.array(std_bl) * 100
    acc_bl2 = np.array(acc_bl2) * 100
    std_bl2 = np.array(std_bl2) * 100
    acc_kt = np.array(acc_kt) * 100
    std_kt = np.array(std_kt) * 100

    if errorbar:
        plt.errorbar(x=x_points, y=acc_kt, yerr=std_kt, fmt='-x', color='#89bc73',
                     label='Transfer Condition (Trained on common latent features)')
        plt.errorbar(x=x_points, y=acc_bl, yerr=std_bl, fmt='-.o', color='#ea52bf',
                     label='Ground Truth Features (Trained on target context)')
        if plot_bl2:
            plt.errorbar(x=x_points, y=acc_bl2, yerr=std_bl2, fmt='--D', color='#f18c5d',
                         label='Ground Truth Features (Trained on source context)')
    else:
        plt.plot(x_points, acc_kt, color='blue', label='Transfer Condition')
        plt.fill_between(x_points, acc_kt - std_kt, acc_kt + std_kt, color='blue', alpha=0.4)

        plt.plot(x_points, acc_bl, color='pink', label='Baseline 1 Condition')
        plt.fill_between(x_points, acc_bl - std_bl, acc_bl + std_bl, color='pink', alpha=0.4)

        if plot_bl2:
            plt.plot(x_points, acc_bl2, color='red', label='Baseline 2 Condition')
            plt.fill_between(x_points, acc_bl2 - std_bl2, acc_bl2 + std_bl2, color='red', alpha=0.4)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='lower right')
    plt.savefig(file_path + filename + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()
