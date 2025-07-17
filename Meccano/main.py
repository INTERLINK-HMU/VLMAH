import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import os
from data_loader import DataSet
from models import ResearchModels
from utils import LearningRateReducerCb, Compute_class_weights
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model



def trainANDtest(data_type, modeTrainTest = 'train', modelName=None, saved_model=None, PoS_focus = 'noun', image_shape=None, batch_size=1,
          nb_epoch=None, epoch_last=None, frame_seq_len = 8, Multi_GPU = False, loss_weights = False):

    path2save = os.path.join('weights')

    if Multi_GPU:
        # ----- Multi-gpu ------------
        strategy = tf.distribute.MirroredStrategy(devices=None)
        GLOBAL_BATCH_SIZE = strategy.num_replicas_in_sync * batch_size

        print(f'Number of devices: {strategy.num_replicas_in_sync}')

        batchs = GLOBAL_BATCH_SIZE
        
    else:
        batchs = batch_size

    # Errors
    if modeTrainTest not in ['train', 'retrain', 'test']:
        raise ValueError('Specify an appropriate learning mode (train, retraim, test)')

        # Two modes: if true then we are in the test mode, else we move to train or val
    if modeTrainTest is 'test':
        scores_runs, scores_runs_top5 = [], []

        for posa_it in range(10):
            data = DataSet(seq_length=frame_seq_len, data_type=data_type, sampling_type=0, TrainValTest='test',
                           model_name=modelName)

            if PoS_focus in ['noun', 'verb']:
                val_generator = data.frame_generator_PredictionNLP(batchs, data_type, modelName, PoS_focus)
                test_generator = data.frame_generator_PredictionNLP(batchs, data_type, modelName, PoS_focus)
                steps_todo = len(data.data) // (batchs * 2)

            elif PoS_focus in ['both']:
                val_generator = data.frame_generator_PredictionNLP_original(batchs, data_type, modelName)
                test_generator = data.frame_generator_PredictionNLP_original(batchs, data_type, modelName)
                steps_todo = len(data.data) // (batchs)

            else:
                val_generator = data.frame_generator_PredictionNLP_Action(batchs, data_type, modelName)
                test_generator = data.frame_generator_PredictionNLP_Action(batchs, data_type, modelName)
                steps_todo = len(data.data) // (batchs)

            model = ResearchModels(saved_model=path2save + '/model_' + modelName + '_' + str(epoch_last) + '.h5')

            score = model.model.evaluate_generator(val_generator, steps=steps_todo - 3, use_multiprocessing=False,
                                                   verbose=1)
            scores_runs.append((score[1]))
            scores_runs_top5.append((score[2]))

            print('Prediction Accuracy, :', posa_it, score[1])

        print('The mean acc for split is:', sum(scores_runs) / len(scores_runs))
        np.save('scores_overruns.npy', np.array(scores_runs))
        np.save('scores_overrunsTop5.npy', np.array(scores_runs_top5))
        print('The max acc for split is:', max(scores_runs))

        # Prediction probabilities for confusion matrix
        pred_prob = model.model.predict_generator(test_generator, steps=700, use_multiprocessing=False, verbose=1)

        y_prediction = np.argmax(pred_prob, axis=1)
        np.save('y_pred.npy', np.array(y_prediction))


    else:
        
               
        data = DataSet(seq_length = frame_seq_len, data_type = data_type,sampling_type = 0,TrainValTest='train', model_name=modelName)

        steps_per_epoch = (len(data.data)//batchs)-10

        if PoS_focus in ['verb','noun']:
            _,_,_,total_words, max_sequence_len,tokenizer,_ = data.get_words_ActionPrediction()
        else:
            _, _, _, total_words, max_sequence_len, tokenizer, _ = data.get_words_ActionPrediction_Action()

        
        weights = Compute_class_weights(tokenizer) if loss_weights else None


        if modeTrainTest is 'retrain':
           
            model= ResearchModels(saved_model=path2save+'/model_'+modelName+'_'+str(epoch_last)+'.h5')

            if PoS_focus in ['noun', 'verb']:
                generator = data.frame_generator_PredictionNLP(batchs, data_type, modelName, PoS_focus)
                steps_todo = len(data.data) // (batchs * 2)
            else:
                generator = data.frame_generator_PredictionNLP_Action(batchs, data_type, modelName)
                steps_todo = len(data.data) // (batchs)

            

            for ep in range(nb_epoch):
                    
                # Use fit generator.
                hist = model.model.fit_generator(
                      generator=generator,
                      steps_per_epoch=50,#steps_per_epoch,
                      epochs=ep+1,
                      verbose=1,
                      #callbacks=[LearningRateReducerCb()],
                      use_multiprocessing=False,  
                      workers=1,  initial_epoch = ep)
                
                model.model.save(path2save+'/model_'+modelName+'_'+str(ep+int(epoch_last)+1)+'.h5')
                if os.path.exists(path2save + '/model_'+modelName+'_' + str(ep+int(epoch_last)) + '.h5'):
                    os.remove(path2save + '/model_'+modelName+'_' + str(ep+int(epoch_last)) + '.h5')
        else:
          #  with strategy.scope():
            # If you want custom weighted cross entropy or focal loss then 1. set Custom_Cross: True, set to Loss_req: with 'w_crossen': for weighted cross-entropy, 'focal': for focal loss
            model = ResearchModels(nb_classes=total_words, model=modelName, batch_size = batchs, Word_seq_len = max_sequence_len.shape[1], Frame_seq_len = frame_seq_len, features_length=None, lossweights = loss_weights, weights=weights, Custom_cros=False, Loss_req = 'focal')

            if Multi_GPU:
              appear = data.frame_generator_PredictionNLPMULTI(GLOBAL_BATCH_SIZE, data_type, modelName)
              train_data = tf.data.Dataset.from_tensor_slices((appear[0], appear[1]))

              train_data = train_data.batch(batch_size)

              # Disable AutoShard.
              options = tf.data.Options()
              options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
              train_data = train_data.with_options(options)
              print('HERE_I_am')

              parallel_model = multi_gpu_model(model, gpus=2)
              parallel_model.compile(loss='categorical_crossentropy', optimizer='adam',
                            metrics=['accuracy', 'top_k_categorical_accuracy'])
              hist = parallel_model.fit(train_data,verbose=1, epochs=nb_epoch)

              parallel_model.save(path2save+'/model_'+modelName+'_'+str(nb_epoch)+'.h5')
            
              model.model.compile(loss='categorical_crossentropy', optimizer='adam',
                              metrics=['accuracy', 'top_k_categorical_accuracy'])
            else:

              if PoS_focus in ['noun','verb']:
                generator = data.frame_generator_PredictionNLP(batchs, data_type, modelName, PoS_focus)
                steps_todo = len(data.data) // (batchs * 2)
              else:
                  generator = data.frame_generator_PredictionNLP_Action(batchs, data_type, modelName)
                  steps_todo = len(data.data) // (batchs)

              for ep in range(nb_epoch):
                  # Use fit generator.
                  hist = model.model.fit_generator(
                        generator=generator,
                        steps_per_epoch= 50,#steps_per_epoch,
                        epochs=ep+1,
                        verbose=1,
                        use_multiprocessing=False,
                        workers=1,  initial_epoch = ep, class_weight= weights)
                  model.model.save(path2save + '/model_'+modelName+'_' + str(ep + 1) + '.h5')
                  if os.path.exists(path2save + '/model_'+modelName+'_' + str(ep) + '.h5'):
                      os.remove(path2save + '/model_'+modelName+'_' + str(ep) + '.h5')


############################################################################
#             MAIN 
############################################################################

def main():

    print(keras.__version__)
    print(sys.version)
    print(np.__version__)

    tf.test.is_built_with_cuda()
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1" # first gpu

    batch_size = 32
    Trained = 'test' # train or test or retrain

    epoch_last = 400 # input('Epoch checkpoint:') if Trained in ['retrain','test'] else None

    # MODELS and their permissions:
    # Data_type: images-> Supported models: a) Vid (only video), b) VNLP (nlp + video)
    #            features-> Supported models: a) Objs_Only, b)Objs_NLP c) ObjsGazeHands_Only, d) ObjsGazeHands_NLP
    #
    # To try only NLP then use model: NLP and for the data_type type images
    model = 'ObjsGazeHands_NLP' #'NLP' # Vid #VNLP #Obj_Only # Obj_NLP # ObjsGazeHands_Only, # ObjsGazeHands_NLP # VNLP_Both, # ObjsGazeTSNRGB_NLP
    PoS = 'YOLO' #noun, verb, both or action 
    
    seq_length = 8

    data_type = 'features' # Refers to the input for the vision-net : a. images, b. features
    nb_epoch = 400
    loss_weights = False# if true then it will compute class weights and apply weighted cross-entropy loss



    trainANDtest(data_type, modeTrainTest = Trained, modelName=model, PoS_focus = PoS, image_shape=None, batch_size=batch_size,
          nb_epoch=nb_epoch, epoch_last=epoch_last, frame_seq_len = seq_length, Multi_GPU = False, loss_weights=loss_weights)

if __name__ == '__main__':
    main()