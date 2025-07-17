
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import functools
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input, Concatenate, ZeroPadding3D, Conv3D, Flatten,Dropout, MaxPool3D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from utils import w_categorical_crossentropy, focal_loss


class ResearchModels():

    def __init__(self, nb_classes=None, model=None, batch_size = 2,Word_seq_len = None, Frame_seq_len = None, features_length=None, 
    lossweights = False, saved_model=None,weights=None, Custom_cros=False, Loss_req=None):


        self.batch_size = batch_size
        self.Word_seq_len = Word_seq_len
        self.Frame_seq_len = Frame_seq_len
        self.nb_classes = nb_classes
        self.lossWeights = lossweights
        self.saved_model = saved_model
        self.weights= weights
        self.Custom_cros= Custom_cros
        self.Loss_req = Loss_req

        # --------- Metrics------------
        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy','top_k_categorical_accuracy']

        # Model Loading and definition

        # --------- Get the appropriate model. -----------
        if self.saved_model is not None:
            print("Loading model")
            self.model = load_model(self.saved_model, custom_objects={'w_categorical_crossentropy':w_categorical_crossentropy})

        elif model in ['VNLP']:
            print("Setting up Video + Language in a C3D + LSTM model.")
            self.model = self.NLP_Video()
        elif model in ['NLP']:
            print("Setting up Language LSTM model.")
            self.model =  self.NLP_simple()
        elif model in ['Vid']:
            print('Setting up Video C3D model.')
            self.model = self.Vid_simple()
        elif model in ['Obj_Only']:
            print('Setting up Objs features model.')
            self.model = self.Obj_simple()
        elif model in ['Obj_NLP']:
            print('Setting up Objs features + NLP model.')
            self.model = self.Obj_NLP()
        elif model in ['ObjsGazeHands_Only']:
            print("Setting up Objs + Gaze+ Hands in an BiLSTM model.")
            self.model = self.ObjGazeHand_simple()

        elif model in ['ObjsGazeHands_NLP']:
            print("Setting up Objs + Gaze+ Hands +NLP in an BiLSTM model.")
            self.model = self.ObjGazeHand_NLP()

        elif model in ['VNLP_Both']:
            print('Setting up Video+NLP model that jointly predicts verb and noun')
            self.model = self.NLP_VideoVerbNoun()
        elif model in ['ObjsGazeTSNRGB_NLP']:
            print('Setting up a model with RGB (TSN), Objs, Gaze and NLP')
            self.model = self.ObjsGazeTSNRGB_NLP()

        else:
            print("Unknown network.")
            sys.exit()


        # Now compile the network.
        optimizer = 'sgd'#tf.keras.optimizers.Adam(learning_rate=0.1)

        lossy = 'categorical_crossentropy'

        if self.lossWeights:

           # Call it to load them

           if self.Custom_cros:

               if self.Loss_req in ['w_crossen']:

                   print('Using weighted categorical cross-entropy loss')
                   ncce = functools.partial(w_categorical_crossentropy)
                   ncce.__name__ = 'w_categorical_crossentropy'

                   lossy = ncce

                   self.model.compile(loss=lossy, optimizer=optimizer, metrics=metrics)

               elif self.Loss_req in ['focal']:
                   print('Using focal loss')
                   ncce = functools.partial(focal_loss())
                   ncce.__name__ = 'focal_loss'

                   lossy = ncce

                   self.model.compile(loss=lossy, optimizer=optimizer, metrics=metrics)
               else:
                   raise ValueError('Unsupported loss, please add w_crossen: for weighted cross-entropy, focal: for focal loss!')

           else:
               self.model.compile(loss=lossy, optimizer=optimizer, metrics=metrics)
        else:
          
           self.model.compile(loss=lossy, optimizer=optimizer, metrics=metrics)

      #  print(self.model.summary())

    ########################################################################################################################
    # MODEL DEFINITIONS
    ########################################################################################################################
    """# **Define Models**"""

    # NLP + VIDEO with Conv3D, and frames as inputs
    def NLP_Video(self):
        
        weight_decay = 0.001

        inputsWords = Input(batch_input_shape=(self.batch_size, self.Word_seq_len))

        model_words_new = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works_new1 = Embedding(self.nb_classes, 100, input_length=self.Word_seq_len)(inputsWords)

        model_words = Concatenate()([model_works_new1,model_words_new])

        model_words = Bidirectional(LSTM(150,return_sequences=True))(model_words)

        model_words = Bidirectional(LSTM(50))(model_words)

        model_words_A = Dense(256, activation='relu')(model_words)

        # ---- Video -----------------------
        inputsFrames = Input(batch_input_shape=(self.batch_size, self.Frame_seq_len,224,224,3))
        model_visual = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
                    activation='relu',kernel_regularizer=l2(weight_decay))(inputsFrames)
        model_visual = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(model_visual)

        model_visual = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
                    activation='relu',kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(model_visual)

        model_visual = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
                    activation='relu',kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(model_visual)

        model_visual = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
                    activation='relu',kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(model_visual)

        model_visual = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                    activation='relu',kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Flatten()(model_visual)
        model_visual = Dense(512,activation='relu',kernel_regularizer=l2(weight_decay))(model_visual)
       
        model_visual_A = Dense(256,activation='relu',kernel_regularizer=l2(weight_decay))(model_visual)
        

        model_concat = Concatenate()([model_words_A, model_visual_A])

        model_out = Dense(self.nb_classes, activation='softmax', name='Predictions')(model_concat)

        model = Model(inputs= [inputsWords,inputsFrames], outputs= [model_out],name="NLP_Video")

        return model


    def NLP_simple(self):

        inputsWords = Input(shape=(self.Word_seq_len))

        model_works_new = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works = Concatenate()([model_works,model_works_new])

        model_works = Bidirectional(LSTM(150,return_sequences=True))(model_works)

        model_works = Bidirectional(LSTM(50))(model_works)

        model_works = Dense(self.nb_classes, activation='softmax')(model_works)

        model = Model(inputs= inputsWords, outputs= model_works,name="NLP")

        return model

    def Obj_simple(self):

        inputsObjs = Input(shape=(self.Frame_seq_len,20))



        model_works = Bidirectional(LSTM(150,return_sequences=True), batch_input_shape=( None, self.Frame_seq_len,20))(inputsObjs)

        model_works = Bidirectional(LSTM(50))(model_works)

        model_works = Dense(self.nb_classes, activation='softmax')(model_works)

        model = Model(inputs= inputsObjs, outputs= model_works,name="Obj_Only")
        return model


    def ObjGazeHand_simple(self):

        inputsObjs = Input(shape=(self.Frame_seq_len,20))

        inputsGaze = Input(shape=(self.Frame_seq_len, 20))

        inputsHands = Input(shape=(self.Frame_seq_len, 20))


        model_objs = Bidirectional(LSTM(150,return_sequences=True), batch_input_shape=( None, self.Frame_seq_len,20))(inputsObjs)

        model_gaze = Bidirectional(LSTM(150, return_sequences=True), batch_input_shape=(None, self.Frame_seq_len, 20))(inputsGaze)

        model_hands = Bidirectional(LSTM(150, return_sequences=True), batch_input_shape=(None, self.Frame_seq_len, 20))(inputsHands)

        model_objs = Bidirectional(LSTM(50))(model_objs)

        model_gaze = Bidirectional(LSTM(50))(model_gaze)

        model_hands = Bidirectional(LSTM(50))(model_hands)

        model_hands = Dense(256, activation='relu')(model_hands)

        model_gaze = Dense(256, activation='relu')(model_gaze)

        model_objs = Dense(256, activation='relu')(model_objs)


        concatted = tf.keras.layers.Concatenate()([model_objs, model_gaze, model_hands])

        model_works = Dense(128, activation='relu')(concatted)

        model_works = Dense(self.nb_classes, activation='softmax')(model_works)

        model = Model(inputs= [inputsObjs, inputsGaze, inputsHands], outputs= model_works,name="ObjsGazeHands_Only")
        return model

    def ObjGazeHand_NLP(self):

        inputsObjs = Input(shape=(self.Frame_seq_len,20))

        inputsGaze = Input(shape=(self.Frame_seq_len, 20))

        inputsHands = Input(shape=(self.Frame_seq_len, 20))

        inputsWords = Input(shape=(self.Word_seq_len))

        model_objs = Bidirectional(LSTM(150,return_sequences=True), batch_input_shape=( None, self.Frame_seq_len,20))(inputsObjs)

        model_gaze = Bidirectional(LSTM(150, return_sequences=True), batch_input_shape=(None, self.Frame_seq_len, 20))(inputsGaze)

        model_hands = Bidirectional(LSTM(150, return_sequences=True), batch_input_shape=(None, self.Frame_seq_len, 20))(inputsHands)

        model_objs = Bidirectional(LSTM(50))(model_objs)

        model_gaze = Bidirectional(LSTM(50))(model_gaze)

        model_hands = Bidirectional(LSTM(50))(model_hands)

        model_hands = Dense(256, activation='relu')(model_hands)

        model_gaze = Dense(256, activation='relu')(model_gaze)

        model_objs = Dense(256, activation='relu')(model_objs)


        concatted = tf.keras.layers.Concatenate()([model_objs, model_gaze, model_hands])

        model_Feats = Dense(256, activation='relu')(concatted)

        # ------------------- Words ------------------------------------------------------------
        model_works_new = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_words = Concatenate()([model_works, model_works_new])

        model_words = Bidirectional(LSTM(150, return_sequences=True))(model_words)

        model_words = Bidirectional(LSTM(50))(model_words)
        model_words = Dense(256, activation='relu')(model_words)

        # ------------------ Combine -------------------------------
        model_concat = Concatenate()([model_words, model_Feats])
        model_concat = Dense(256, activation='relu')(model_concat)

        model_pout = Dense(self.nb_classes, activation='softmax')(model_concat)

        model = Model(inputs= [inputsWords, inputsObjs, inputsGaze, inputsHands], outputs= model_pout,name="ObjsGazeHands_NLP")

        return model

    def ObjsGazeTSNRGB_NLP(self):

        inputsObjs = Input(shape=(self.Frame_seq_len,20))

        inputsGaze = Input(shape=(self.Frame_seq_len, 20))

        inputsTSN = Input(shape=(self.Frame_seq_len, 1024))

        inputsWords = Input(shape=(self.Word_seq_len))

        model_objs = Bidirectional(LSTM(150,return_sequences=True), batch_input_shape=( None, self.Frame_seq_len,20))(inputsObjs)

        model_gaze = Bidirectional(LSTM(150, return_sequences=True), batch_input_shape=(None, self.Frame_seq_len, 20))(inputsGaze)

        model_TSN = Bidirectional(LSTM(512, return_sequences=True), batch_input_shape=(None, self.Frame_seq_len, 1024))(inputsTSN)

        model_objs = Bidirectional(LSTM(50))(model_objs)

        model_gaze = Bidirectional(LSTM(50))(model_gaze)

        model_TSN = Bidirectional(LSTM(256))(model_TSN)

        model_TSN = Dense(256, activation='relu')(model_TSN)

        model_gaze = Dense(256, activation='relu')(model_gaze)

        model_objs = Dense(256, activation='relu')(model_objs)


        concatted = tf.keras.layers.Concatenate()([model_objs, model_gaze, model_TSN])

        model_Feats = Dense(256, activation='relu')(concatted)

        # ------------------- Words ------------------------------------------------------------
        model_works_new = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_words = Concatenate()([model_works, model_works_new])

        model_words = Bidirectional(LSTM(150, return_sequences=True))(model_words)

        model_words = Bidirectional(LSTM(50))(model_words)
        model_words = Dense(256, activation='relu')(model_words)

        # ------------------ Combine -------------------------------
        model_concat = Concatenate()([model_words, model_Feats])
        model_concat = Dense(256, activation='relu')(model_concat)

        model_pout = Dense(self.nb_classes, activation='softmax')(model_concat)

        model = Model(inputs= [inputsWords, inputsObjs, inputsGaze, inputsTSN], outputs= model_pout,name="ObjsGazeTSNRGB_NLP")

        return model



    def Obj_NLP(self):

        inputsObjs = Input(shape=(self.Frame_seq_len,20))
        inputsWords = Input(shape=(self.Word_seq_len))

        #------------------- OBJECTS ------------------------------------------------------------
        model_objs = Bidirectional(LSTM(150,return_sequences=True), batch_input_shape=( None, self.Frame_seq_len,20))(inputsObjs)

        model_objs = Bidirectional(LSTM(50))(model_objs)

        model_objs = Dense(256, activation='relu')(model_objs)

        # ------------------- Words ------------------------------------------------------------
        model_works_new = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_words = Concatenate()([model_works, model_works_new])

        model_words = Bidirectional(LSTM(150, return_sequences=True))(model_words)

        model_words = Bidirectional(LSTM(50))(model_words)

        model_words = Dense(256, activation='relu')(model_words)

        #------------------ Combine -------------------------------
        model_concat = Concatenate()([model_words, model_objs])
        model_concat = Dense(256, activation='relu')(model_concat)

        model_out = Dense(self.nb_classes, activation='softmax')(model_concat)

        model = Model(inputs= [inputsWords, inputsObjs], outputs= model_out,name="Obj_NLP")
        return model


    def Vid_simple(self):

        weight_decay = 0.001

        inputsFrames = Input(shape=(self.Frame_seq_len, 224, 224, 3))

        # ---- Video -----------------------
        model_visual = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                            activation='relu', kernel_regularizer=l2(weight_decay))(inputsFrames)
        model_visual = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(model_visual)

        model_visual = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                            activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                            activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                            activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                            activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Flatten()(model_visual)
        model_visual = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = Dropout(0.5)(model_visual)
        model_visual = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = Dropout(0.5)(model_visual)

        model_out = Dense(self.nb_classes, activation='softmax')(model_visual)

        model = Model(inputs=[inputsFrames], outputs=[model_out], name="Video_Only")

        return model

        # NLP + VIDEO with Conv3D, and frames as inputs
    def NLP_VideoVerbNoun(self):
        weight_decay = 0.001

        inputsWords = Input(batch_input_shape=(self.batch_size, self.Word_seq_len))

        model_words_new = Embedding(self.nb_classes, 200, input_length=self.Word_seq_len)(inputsWords)

        model_works_new1 = Embedding(self.nb_classes, 100, input_length=self.Word_seq_len)(inputsWords)

        model_words = Concatenate()([model_works_new1, model_words_new])

        model_words = Bidirectional(LSTM(150, return_sequences=True))(model_words)

        model_words = Bidirectional(LSTM(50))(model_words)

        model_words_A = Dense(256, activation='relu')(model_words)

        # ---- Video -----------------------
        inputsFrames = Input(batch_input_shape=(self.batch_size, self.Frame_seq_len, 224, 224, 3))
        model_visual = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(weight_decay))(inputsFrames)
        model_visual = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(model_visual)

        model_visual = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
                              activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        model_visual = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(model_visual)

        model_visual = Flatten()(model_visual)
        model_visual = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        
        model_visual_A = Dense(256, activation='relu', kernel_regularizer=l2(weight_decay))(model_visual)
        

        model_concat = Concatenate()([model_words_A, model_visual_A])

        model_verb = Dense(128, activation='relu', kernel_regularizer=l2(weight_decay))(model_concat)

        model_noun = Dense(128, activation='relu', kernel_regularizer=l2(weight_decay))(model_concat)

        model_outVerb = Dense(self.nb_classes, activation='softmax', name='PredictionsVerb')(model_verb)

        model_outNoun = Dense(self.nb_classes, activation='softmax', name='PredictionsNoun')(model_noun)

        model = Model(inputs=[inputsWords, inputsFrames], outputs=[model_outVerb, model_outNoun], name="NLP_Video_Both")

        return model
