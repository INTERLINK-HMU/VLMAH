import tensorflow as tf
import numpy as np
import os  
from itertools import product
import lmdb
import tensorflow.keras.backend as K
import csv
import random

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)


def Compute_class_weights(tokenizer):
    total_words = len(tokenizer.word_index)
    counts = tokenizer.word_counts
    words = tokenizer.word_index

    word_freq = np.zeros(total_words)

    # Fix the counts
    for k, v in counts.items():
        if k in words.keys():
            word_freq[words[k] - 1] = v

    total_presence = 0
    for i in range(len(word_freq)):

        if word_freq[i] <= np.mean(word_freq):
            total_presence += word_freq[i]

    weight_vec = np.ones(total_words + 1)

    for i in range(len(word_freq)):

        if word_freq[i] <=np.mean(word_freq):
           #weight_vec[i] = 1 + (total_presence - word_freq[i]) / total_presence
           weight_vec[i] = 1 + (((total_presence - word_freq[i]) / total_presence)*10)

    np.save('./weights_class/weights.npy', weight_vec)



    weights_vec_dic = {}

    keys = range(len(weight_vec)-1)
    for i in keys:
        weights_vec_dic[i] = float(weight_vec[i])
        type(i)

    #weight_vec_dic = {'Predictions':list(weight_vec[:-1])}

    return weights_vec_dic


def w_categorical_crossentropy(y_true, y_pred):
    class_weights = os.path.join('weights_class', 'weights.npy')
    weights = np.load(class_weights)
    nb_cl = len(weights)

    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)

    for c_t in range(nb_cl):
        final_mask += (K.cast(weights[c_t], K.floatx()) * K.cast(y_pred_max_mat[:, c_t], K.floatx()) * K.cast(
            y_true[:, c_t], K.floatx()))

    return K.categorical_crossentropy(y_pred, y_true) * final_mask


##------- Focal Loss ------------------------------------------------------------------------
def focal_loss(y_true, y_pred, gamma=3.0):
    class_weights = os.path.join('weights_class', 'weights.npy')

    weights = np.load(class_weights)
    nb_cl = len(weights)

    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    pt = K.sum(np.multiply(y_true, y_pred), axis=1)

    for c_t in range(nb_cl):
        final_mask += (K.cast(weights[c_t], K.floatx()) * K.cast(y_pred_max_mat[:, c_t], K.floatx()) * K.cast(
            y_true[:, c_t], K.floatx()))

    outi = K.sum(-(final_mask * ((1 - pt) ** gamma)) * K.log(pt))

    return outi




def import_pre_extracted_Assembly(sample, frames, modality, mode):
    """
    Input:
        frames: a list of frames
        modality: obj, gaze, hands, rgb, flow
        mode: Train_Val, Test

    Output:
        features: numpy array
    """

    path_to_lmdb = r'\\Assembly101\Dataset\TSM_e4\e4\\'

    lmdb_op = lmdb.open(path_to_lmdb, readonly=True, lock=False)

    path_to_lmdb1 = r'\Assembly101\Dataset\TSM_e4\HMC_21179183_mono10bit\\'

    lmdb_op1 = lmdb.open(path_to_lmdb1, readonly=True, lock=False)

    if len(frames)==0:
        print(sample)
        raise ValueError('Empty frame array!')

    if int(sample[-1]) < int(frames[-1]):
        frames = frames[:-1]

    features_left, features_right, features_diff = [], [], []
    # for each frame
    for f in frames:

        num = str(int(f)).rjust(10, '0')
        img = num+'.jpg'

        # ID sample split
        camids = sample[0].split('/')

        key = camids[0]+'/'+camids[1][:-4]+'/'+camids[1][:-4]+'_'+img

        print(key)
        print(LOLO)


        with lmdb_op.begin() as e:
  


            dd = e.get(key.encode('utf-8'))

        if dd is None:
            with lmdb_op1.begin() as e:
                dd = e.get(key.encode('utf-8'))

        # FIX ME AFTER THE .csv have been corrected
        if dd is None:
            print(f)
            print(key)
            dd = np.zeros(1024)
            print(sample)
            raise ValueError('Error')


        # convert to numpy array
        data = np.frombuffer(dd, 'float32')


        # take the first 1024
        data_left = data[:1024]

        # take the last 1024
        data_right = data[1024:]

        # sample 1024 with step 2
        if features_diff:
            data_diff = np.abs(data[0:2048:2] - features_diff[-1])
        else:

            data_diff = data[0:2048:2]

        # append to list
        features_left.append(data_left)
        features_right.append(data_right)
        features_diff.append(data_diff)

    if features_left :
        pass
    else:
        print(sample)
        print(frames)
        raise ValueError("list is empty")

    # convert list to numpy array
    features_left = np.array(features_left)
    features_right = np.array(features_right)
    features_diff = np.array(features_diff)

    return features_left, features_right, features_diff

def Add_Noise(word_em, percentage, unique_labels):
    """
       Input:
           word_em: the indexes of the verbs and nouns that are fed into the Language model
           percentage: the portion of labels to be affected
           unique_labels: the combinations of verb+nouns (i.e. the labels) that we have found in the dataset

       Output:
           new_seq: the new sequence of the noisy labels
       """

    # Find the history that we have so far

    get_indexes_temp = np.where(word_em==1)
    if len(word_em)-1 in list(get_indexes_temp[0]):
        get_indexes = get_indexes_temp[0][:-1]
    else:
        get_indexes = get_indexes_temp[0]

    if len(get_indexes)>1:
        # Number of labels inside my history
        num_label = len(get_indexes)-1 # count the 1 and subtract one due to the case "start 1"

        # How many labels we need to change
        lbl_affect = round(num_label*(percentage/100))

        if lbl_affect>1:

            for k in range(lbl_affect):
                # generate a random position
                idx_label = random.randint(0,len(get_indexes)-1)

                # Get the verb idx and noun idx to be changed
                v_idx = get_indexes[idx_label]+1
                n_idx = get_indexes[idx_label]+2
                if v_idx==len(word_em) or n_idx==len(word_em):
                    return word_em
                else:
                    if word_em[v_idx] in [1,51, 52] or word_em[n_idx] in [1,51,52]:
                        return word_em
                     
                    else:
                        # Get a random new label (verb and noun pair)
                        idx_new_lbl = random.choice(unique_labels)

                  
                        if idx_new_lbl[0] in [1,51,52] or idx_new_lbl[1] in [1,51,52]:
                            print(idx_new_lbl)
                            raise ValueError('You are planning to add an index that corresponds to a) start, b) end, c) ''')

                        word_em[v_idx]=idx_new_lbl[0]
                        word_em[n_idx]=idx_new_lbl[1]


            new_seq = word_em
        else:
            new_seq = word_em

    else:
        new_seq = word_em


    return new_seq

def Add_Noise_ShortLong(word_em, percentage, unique_labels,shortlong):
    """
       Input:
           word_em: the indexes of the verbs and nouns that are fed into the Language model
           percentage: the portion of labels to be affected
           unique_labels: the combinations of verb+nouns (i.e. the labels) that we have found in the dataset

       Output:
           new_seq: the new sequence of the noisy labels
       """

    # Find the history that we have so far
    get_indexes_temp = np.where(word_em == 1)

    if len(word_em)-1 in list(get_indexes_temp[0]):
        get_indexes = get_indexes_temp[0][:-1]
    else:
        get_indexes = get_indexes_temp[0]


    if len(get_indexes)>1:
        # Number of labels inside my history
        num_label = len(get_indexes)-1 # count the 1 and subtract one due to the case "start 1"

        # How many labels we need to change
        if shortlong in ['short']:
            lbl_affect = round(num_label * (percentage / 100))
        else:  # for the long-range we do 100 - percentage
            lbl_affect = round(num_label * ((100 - percentage) / 100))

        # recognizer percentange
        correct_class_prob = 60 # HERE add the percentage of erroneous classification #DO NOT GET Confused by the correct :P


        if lbl_affect>1:

            for k in range(lbl_affect-1):
                # Check if we will add noise or not
                Irecon = random.random() * 100
                if Irecon > correct_class_prob:  # Add noise
                    pass
                else:
                    if shortlong in ['short']:
                        k += 2
                        idx_label = k
                    else:
                        idx_label = k + (len(get_indexes) - lbl_affect)  # 0 ews N + skip ta short-term labels

                    if idx_label > len(get_indexes):
                        raise ValueError('Check your index computations')

                    # Get the verb idx and noun idx to be changed


                    v_idx = get_indexes[idx_label] + 1
                    n_idx = get_indexes[idx_label] + 2
                    if v_idx == len(word_em) or n_idx == len(word_em):
                        return word_em
                    else:
                        if word_em[v_idx] in [1, 51, 52] or word_em[n_idx] in [1, 51, 52]:
                            return word_em
                          
                            print(get_indexes[idx_label])
                        
                        else:
                            # Get a random new label (verb and noun pair)
                            idx_new_lbl = random.choice(unique_labels)

                           
                            if idx_new_lbl[0] in [1, 51, 52] or idx_new_lbl[1] in [1, 51, 52]:
                                print(idx_new_lbl)
                                raise ValueError(
                                    'You are planning to add an index that corresponds to a) start, b) end, c) ''')

                            word_em[v_idx] = idx_new_lbl[0]
                            word_em[n_idx] = idx_new_lbl[1]






            new_seq = word_em
        else:
            new_seq = word_em

    else:
        new_seq = word_em


    return new_seq


def Add_Noise_Action(word_em, percentage):
    """
       Input:
           word_em: the indexes of the verbs and nouns that are fed into the Language model
           percentage: the portion of labels to be affected
           unique_labels: We dont want that because we know that there are 960 classes, we should exclude 1, 29,30 (i.e. ', Start, end)

       Output:
           new_seq: the new sequence of the noisy labels
       """

    # Find the history that we have so far
    get_indexes_temp = np.where(word_em==1)
    if len(word_em)-1 in list(get_indexes_temp[0]):
        get_indexes = get_indexes_temp[0][:-1]
    else:
        get_indexes = get_indexes_temp[0]

    if len(get_indexes)>1:
        # Number of labels inside my history
        num_label = len(get_indexes)-1 # count the 1 and subtract one due to the case "start 1"

        # How many labels we need to change
        lbl_affect = round(num_label*(percentage/100))

        if lbl_affect>1:

            for k in range(lbl_affect):
                # generate a random position
                idx_label = random.randint(0,len(get_indexes)-1)

                # Get the verb idx and noun idx to be changed
                v_idx = get_indexes[idx_label]+1

                if v_idx==len(word_em) :
                    return word_em
                else:
                    if word_em[v_idx] in [29, 30]:
                    
                        raise ValueError('The verb or noun index corresponds to a) start, b) end, c) ''')
                    else:
                       # Check if we will add noise or not
                        Irecon = random.random() * 100
                        if Irecon > correct_class_prob:  # Add noise
                            # Get a random new label (verb and noun pair)
                            idx_new_lbl = random.randint(0,960)
                            while idx_new_lbl in [1,29,30,960]:
                                idx_new_lbl = random.randint(0, 960)

                            if idx_new_lbl in [1,29,30]:
                                print(idx_new_lbl)
                                raise ValueError('You are planning to add an index that corresponds to a) start, b) end, c) ''')

                            word_em[v_idx]=idx_new_lbl
                        else:
                            pass

            new_seq = word_em
        else:
            new_seq = word_em

    else:
        new_seq = word_em


    return new_seq

def Add_Noise_Action_ShortLong(word_em, percentage, shortlong):
    """
       Input:
           word_em: the indexes of the verbs and nouns that are fed into the Language model
           percentage: the portion of labels to be affected
           unique_labels: We dont want that because we know that there are 960 classes, we should exclude 1, 29,30 (i.e. ', Start, end)
           shortlong: 'short' for short-range, 'long' for longrange

       Output:
           new_seq: the new sequence of the noisy labels
       """

    # Find the history that we have so far
    get_indexes_temp = np.where(word_em==1)
    if len(word_em)-1 in list(get_indexes_temp[0]):
        get_indexes = get_indexes_temp[0][:-1]
    else:
        get_indexes = get_indexes_temp[0]

    if len(get_indexes)>1:
        # Number of labels inside my history
        num_label = len(get_indexes)-1 # count the 1 and subtract one due to the case "start 1"

        # How many labels we need to change
        if shortlong in ['short']:
            lbl_affect = round(num_label*(percentage/100))
        else: # for the long-range we do 100 - percentage
            lbl_affect = round(num_label * ((100-percentage) / 100))

        # recognizer percentange
        correct_class_prob = 40 # HERE add the percentage of correct classification


        if lbl_affect>1:

            for k in range(lbl_affect):
                # generate a random position
                if shortlong in ['short']:
                    idx_label = k
                else:
                    idx_label = k+(len(get_indexes)-lbl_affect) # 0 ews N + skip ta short-term labels

                if idx_label>len(get_indexes):
                    raise ValueError('Check your index computations')

                # Get the action idx to be changed
                v_idx = get_indexes[idx_label]+1

                if v_idx==len(word_em) :
                    return word_em
                else:
                    if word_em[v_idx] in [1,29, 30]:
                     

                        #pass # DO NOT CHANGE THE START AND END
                         pass
#                        raise ValueError('The verb or noun index corresponds to a) start, b) end, c) ''')
                    else:
                        # Check if we will add noise or not
                        Irecon = random.random() *100

                        if Irecon > correct_class_prob: #Add noise

                            # Get a random new label (verb and noun pair)
                            idx_new_lbl = random.randint(0,960)
                            while idx_new_lbl in [1,29,30,960]:
                                idx_new_lbl = random.randint(0, 960)

                          
                            if idx_new_lbl in [1,29,30]:
                                print(idx_new_lbl)
                                raise ValueError('You are planning to add an index that corresponds to a) start, b) end, c) ''')

                            word_em[v_idx]=idx_new_lbl
                        else:

                            pass
            new_seq = word_em
        else:
            new_seq = word_em

    else:
        new_seq = word_em


    return new_seq


def calculate_timesteps(video_id, end_frame, mode, dataset="Assembly"):
    """
       Input:
           video_id: the video number - from 1 to 20
           end_frame: the current frame
           mode: train,test,val to open the relative csv

       Output:
           frames: frames that correspond to timesteps in form video_framenumber
           sgm_back: int, how many segments you need to go back
       """


    time_step = 0.25
    fps = 30
    point = (int(end_frame))
    sequence_length = 8
    frames_out =  []
    # generate the timestamps
    time_stamps = np.arange(time_step, time_step * (sequence_length + 1), time_step)[::-1]

    # compute the time stamp corresponding to the beginning of the action
    end_time_stamp = point / fps

    # subtract time stamps to the timestamp of the last frame
    time_stamps = end_time_stamp - time_stamps

    # convert timestamps to frames
    frames = np.floor(time_stamps * fps).astype(int)
    if len(frames)<8:
        print(len(frames))
        print('frames0:', frames)
    # sometimes there are not enough frames before the beginning of the action
    # in this case, we just pad the sequence with the first frame
    if frames.max() >= 1:
        frames[frames < 1] = frames[frames >= 1].min()
    elif frames.max() == 0:
        frames = frames + 1
        frames[frames < 1] = frames[frames >= 1].min()

    # numberOfVideo_numberOfFrame p.x. 03_00967
    # frames = frames + 1
    if dataset  in ['Meccano']:
        frames = np.array(list(map(lambda x: str(video_id).zfill(2) + "_" + str(x), frames)))
    else:
        pass

        #frames = np.array(list(map(lambda x: str(video_id).zfill(2) + "_" + str(x), frames)))
    if len(frames) < 8:
        print("frames1:", len(frames), frames)

    path = os.path.join('splits')
    if mode == 'train':
        path = path + '/train.csv'
    elif mode == 'val':

        path = path + '/val.csv'
    else:

        path = path + '/test.csv'
    #

    with open(path, encoding='utf-8', mode='r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    sgm_list = []

    for i in range(0, len(data)):

        if str(video_id) == data[i][0]:
            data[i][3] = data[i][3].split('.')[0]
            data[i][4] = data[i][4].split('.')[0]

            for j in range(0, len(frames)):
                if dataset in ['Meccano']:
                    parts = frames[j].split('_')

                    
                    frames_out.append(format(int(parts[1]), '05d'))

                    if format(int(parts[1]), '05d')>= data[i][3] and format(int(parts[1]), '05d')<=data[i][4]:
                       sgm_list.append(data[i])
                       continue
                else:
                    frames_out = frames.copy()
                    if int(frames_out[0]) >= int(data[i][3]) and int(frames_out[-1]) <= int(data[i][4]):
                        sgm_list.append(data[i])
                        continue



    sgm_list2array =np.array(sgm_list)
   
    if sgm_list2array.ndim>1:
        sgm_back = np.zeros((1,8))
        sgmnts = np.unique(sgm_list2array[:,3])
        for sg in range(0, len(np.unique(sgm_list2array[:,3]))):
            for ts in range(0, 8):

                if int(sgmnts[sg]) > int(frames_out[ts]):
                    sgm_back[0, ts] += 1

        
        out_sgmt = int(sgm_back[0][4])
    else:

        out_sgmt = 0
        

    # Fix the frames
    frames_end = frames_out[4]


    return frames_end, out_sgmt
