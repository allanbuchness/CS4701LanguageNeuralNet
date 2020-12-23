# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:03:23 2020

@author: Allan
"""
import random   #standard library
import csv

from nltk import TweetTokenizer    #3rd party packages
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt

#Global Variables
random.seed('CS4701')

#Pre-Processing Set-up
stop_words = set(stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish') + ['rt', 'u'])
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case = False)
stemmer = SnowballStemmer("english")


#Functions

def processcsv(file):    
    """processescsv takes a file name and outputs a list containing rows of the
    csv file"""
    a = []
    with open(file, 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        lines = 0
        for row in reader:
            a.append(row)
            lines += 1
        return a, lines
    
def split_data(inputs, outputs, train_percent):
    """porbabalistically assign inputs and outputs to test or train"""
    trainin = []
    trainout = []
    testin = []
    testout = []
    for i in range(len(inputs)):
        if (random.uniform(0,1)) > train_percent:
            testin.append(inputs[i])
            testout.append(outputs[i])
        else: 
            trainin.append(inputs[i])
            trainout.append(outputs[i])
    return trainin, trainout, testin, testout
        
def pre_process(line, stops=True, stemming=True, hashtag=True):
    """
    Processes and returns a string by stemming, removing stopwords and punctuation,
    and leaving hashtags.

    Parameters
    ----------
    line : string
        A string to be pre-processed.
    stops : bool, optional
        Whether to remove stopwords from nltk Corpus English, Spanish, French, and
        custom twitter words. The default is True.
    stemming : bool, optional
        Whether to stem words (e.g. running -> run) according to the Snowball
        Stemmer. The default is True.
    hashtag : bool, optional
        Whether to include hashtags rather than delete them as punctuation. The 
        default is True.

    Returns
    -------
    output : string
        pre-processed string.

    """
    tokenized = tknzr.tokenize(line)
    hashtags = [word for word in tokenized if (word[0] == "#")]
    nopunct = [word for word in tokenized if (word.isalpha())]
    if stops:
        nostops = [w for w in nopunct if not w in stop_words]
    else: nostops = nopunct
    if stemming:
        stemmed = [stemmer.stem(word) for word in nostops]
    else: stemmed = nostops
    output = ' '.join((stemmed + hashtags) if hashtag else (stemmed))
    return output

def make_model(vector_train, max_tokens, output_seq_len, num_hidden, size_hidden,
                hidden_activ='relu', output_activ='sigmoid', loss='binary_crossentropy',
                optimizer='adam', embed=True):

    vectorizer = TextVectorization(max_tokens = max_tokens,
                                    output_sequence_length = output_seq_len)
    vectorizer.adapt(vector_train)
    model = keras.Sequential()
    model.add(layers.Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer)                                               #Vectorizer Layer
    if embed: model.add(layers.Embedding(max_tokens+1, size_hidden))    #Embedded Layer
    for i in range(num_hidden):
        model.add(layers.Dense(size_hidden, activation=hidden_activ))   #hidden layers
    model.add(layers.Dense(1, activation=output_activ))                 #output layer
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def get_inputs_outputs(file, ratio, stops=True, stemming=True, hashtag=True):
     #languageclass.csv pre-processing
    a = processcsv(file)
    #manipulate: col 0 is text, col 1 is classification
    a = a[0]
    b = []
    for i in range(len(a)):
        b.append([pre_process(a[i][5], stops, stemming, hashtag), a[i][4]])
    inputsdata = [i[0] for i in b]
    outputs = [i[1] for i in b]
    outputsbinary = [int((i=='2')) for i in outputs]    #1 is not offensive, 0 is
    trainin, trainout, testin, testout = split_data(inputsdata, outputsbinary, ratio)
    return trainin, trainout, testin, testout
        
def plot_history(model):
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
#Testing and Demo Scripts start here

def rand_model():
    """
    A test with unstructured data. 50% of data 
    """
    trainin, trainout, testin, testout = get_inputs_outputs('test.csv', 0.67)
    model = make_model(trainin, 5000, 10, 20, 20, embed=True)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=32, validation_data=(testin, testout), epochs=10)    

def baseline_model():
    """
    Baseline Neural Net Model Demonstration:
        Preprocess:                 hashtags, stemming, stopwords
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             False
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     5
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
        
        

    Returns
    -------
    None.

    """
    print("""
        Baseline Neural Net Model Demonstration:
        Preprocess:                 hashtags, stemming, stopwords
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             False
        Dataset Size:               24,784 Tweets
        Training Data:              60%
        Testing Data:               40%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     5
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
        """)
    #make and train model
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv', 0.60, True, True, True)
    opt = keras.optimizers.Adam(learning_rate = 0.0005)
    model = make_model(trainin, 7000, 35, 3, 600, embed=True, optimizer=opt)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=32, validation_data=(testin, testout), epochs=5)    

def embedded_model():
    """
    Baseline Neural Net Model Demonstration:
        Preprocess:                 hashtags, stemming, stopwords
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             True
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     5
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
        
        

    Returns
    -------
    None.

    """
    print("""
        Baseline Neural Net Model Demonstration:
        Preprocess:                 hashtags, stemming, stopwords
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             True
        Dataset Size:               24,784 Tweets
        Training Data:              67%
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     5
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
        """)
    #make and train model
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=5)    

def dataset_info():
    """
    Prints three random Tweets stripped of non-alpha content
    Prints the portion of data that is objectionable and the portion that is not
    Prints size of Dataset, in Tweets

    Returns
    -------
    None.

    """
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv', 1.0, False, False, False)
    inputs = trainin + testin
    outputs = trainout + testout
    for i in range(3):
        rand = random.randrange(1, len(inputs))
        print('Tweet:  ' + inputs[rand])
    offensive = 0
    for i in range(len(inputs)):
        offensive += outputs[i]
    offense_fraction = offensive / len(outputs)
    print('Objectionable:     ' + str(1 - offense_fraction))
    print('Not Objectionable: ' + str(offense_fraction))
    print('Dataset Size:      ' + str(len(inputs)) + ' Tweets')
    return

def pre_process_demonstration():
    """
    Prints, for three random tweets:
    First, the raw tweet text, 
    then, non-alpha removed, 
    then stopwords removed, 
    finally, words stemmed
    
    It also prints token number metrics for each type of pre-processing

    Returns
    -------
    None.

    """
    
    a, length = processcsv('languageclass.csv')
    original_len = 0
    punct_r_len = 0
    stopword_len = 0
    for i in range(3):
        index = random.randrange(1, length)
        print("Raw Tweet:            " + str(a[index][5]))
        original_len += len(a[index][5].split())
        print("Punctuation Removed:  " + pre_process((a[index][5]), stemming=False, stops=False))
        punct_r_len += len(pre_process((a[index][5]), stemming=False, stops=False).split())
        print("Stopwords Removed:    " + pre_process((a[index][5]), stemming=False, stops=True))
        stopword_len += len(pre_process((a[index][5]), stemming=False, stops=True).split())
        print("Words Stemmed:        " + pre_process((a[index][5]), stemming=True, stops=True) + '\n')
    original_len = original_len / 3
    punct_r_len = punct_r_len / 3
    stopword_len = stopword_len / 3
    print("Average token length, raw tweet:         " + str(original_len))
    print("Average token length, only alpha:        " + str(punct_r_len))
    print("Average Token length, stopwords removed: " + str(stopword_len))
    
def pre_process_baseline():
    """
    Neural Net Demonstration: Changing Pre_process Parameters
        Preprocess:                 varies
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             False
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     3
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
    
    Run 1: Remove Punctuation
    Run 2: Remove Stopwords
    Run 3: Stemming
    Run 4: Remove Stopwords and Flatten Stemwords
        

    Returns
    -------
    None.

    """
    print("""
          
        Neural Net Demonstration: Changing Pre_process Parameters
          
        Preprocess:                 varies
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             False
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     3
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
    
    Run 1: Remove Punctuation
    Run 2: Remove Stopwords
    Run 3: Stemming
    Run 4: Remove Stopwords and Flatten Stemwords)
    """)
    print("Run 1: Only Alphas, no stemming or stopword removal\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, False, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=False)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)    

    print("\n\nRun 2: Stopword Removal in effect, no stemming\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, True, False, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=False)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=('languageclass.csv',testin, testout), epochs=3)   
    
    print("\n\nRun 2: Stemming in effect, no stopword removal\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=False)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)    
    
    
    print("\n\nRun 3: Stopword Removal and Stemming in effect\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, False, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=False)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)    

def pre_process_embedded():
    """
    Neural Net Demonstration: Changing Pre_process Parameters
        Preprocess:                 varies
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             True
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     3
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
    
    Run 1: Remove Punctuation
    Run 2: Remove Stopwords   
    Run 3: Stemming
    Run 4: Remove Stopwords and Flatten Stemwords
        

    Returns
    -------
    None.

    """
    print("""
          
        Neural Net Demonstration: Changing Pre_process Parameters
          
        Preprocess:                 varies
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             True
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     3
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
    
    Run 1: Remove Punctuation
    Run 2: Remove Stopwords
    Run 3: Stemming
    Run 4: Remove Stopwords and Flatten Stemwords)
    """)
    print("Run 1: Only Alphas, no stemming or stopword removal\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, False, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)    

    print("\n\nRun 2: Stopword Removal in effect, no stemming\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs(0.67, True, False, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)    
    
    print("\n\nRun 2: Stemming in effect, no stopword Removal\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)
    
    print("\n\nRun 3: Stopword Removal and Stemming in effect\n")
    
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, False, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)    

def hashtag_test():
    """
    Neural Net Demonstration: Changing Pre_process Parameters
    Preprocess:                 Stemming, No Stopword Removal, Hashtag and no Hashtag
    Vectorizer:                 Integer, 1-gram Tokenized
    Embedded Layer:             True
    Dataset Size:               24,784 Tweets
    Training Data:              67% 
    Testing Data:               33%
    Corpus Limit:               5000
    Input Sequence Size Limit:  30
    Hidden Layers:              2
    Hidden Layer Size:          10
    Batch Size:                 8
    Epochs:                     3
    Activation Function:        ReLU
    Output Activation Function: Sigmoid
    Loss Function:              Binary Cross-Entropy
    Optimizer:                  Adam
    
    Run 1: Include Hashtags
    Run 2: Remove Hashtags   
    
    Returns
    -------
    final test model.
    
    """
    print(     """
    Neural Net Demonstration: Changing Pre_process Parameters
    
        Preprocess:                 Stemming, No Stopword Removal, Hashtag and no Hashtag
        Vectorizer:                 Integer, 1-gram Tokenized
        Embedded Layer:             True
        Dataset Size:               24,784 Tweets
        Training Data:              67% 
        Testing Data:               33%
        Corpus Limit:               5000
        Input Sequence Size Limit:  30
        Hidden Layers:              2
        Hidden Layer Size:          10
        Batch Size:                 8
        Epochs:                     3
        Activation Function:        ReLU
        Output Activation Function: Sigmoid
        Loss Function:              Binary Cross-Entropy
        Optimizer:                  Adam
    
    Run 1: Include Hashtags
    Run 2: Remove Hashtags   
    """)
    
    print("\n\nRun 1: Embedding, Stemming, Include Hashtags\n")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, True)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 2: Embedding, Stemming, NO Hashtags\n")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 2, 10, embed=True)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)
    
def more_hidden_layers():
    """
    Neural Net Demonstration: Tuning Number of Hidden Layers
    Preprocess:                 Stemming, No Stopword Removal, Hashtag and no Hashtag
    Vectorizer:                 Integer, 1-gram Tokenized
    Embedded Layer:             True
    Dataset Size:               24,784 Tweets
    Training Data:              67% 
    Testing Data:               33%
    Corpus Limit:               5000
    Input Sequence Size Limit:  30
    Hidden Layers:              0, 1, 3, 6
    Hidden Layer Size:          10
    Batch Size:                 8
    Epochs:                     3
    Activation Function:        ReLU
    Output Activation Function: Sigmoid
    Loss Function:              Binary Cross-Entropy
    Optimizer:                  Adam
    
    Run 1: 0 hidden layers
    Run 2: 1 hidden layers
    Run 3: 3 hidden layers
    Run 4: 6 hidden layers
    
    Returns
    -------
    None.
    
    """
    print("""
    Neural Net Demonstration: Tuning Number of Hidden Layers
    Preprocess:                 Stemming, No Stopword Removal, Hashtag and no Hashtag
    Vectorizer:                 Integer, 1-gram Tokenized
    Embedded Layer:             True
    Dataset Size:               24,784 Tweets
    Training Data:              67% 
    Testing Data:               33%
    Corpus Limit:               5000
    Input Sequence Size Limit:  30
    Hidden Layers:              0, 1, 3, 6
    Hidden Layer Size:          10
    Batch Size:                 8
    Epochs:                     3
    Activation Function:        ReLU
    Output Activation Function: Sigmoid
    Loss Function:              Binary Cross-Entropy
    Optimizer:                  Adam
    
    Run 1: 0 hidden layers
    Run 2: 1 hidden layers
    Run 3: 3 hidden layers
    Run 4: 6 hidden layers
    """)
    
    print("\n\nRun 1: 0 Hidden Layers")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 0, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 2: 1 Hidden Layer")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 1, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 3: 3 Hidden Layers")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 3, 10, embed=True)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 4: 6 Hidden Layers")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 6, 10, embed=True)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=5) 

def more_hidden_no_embed():
    """
    Neural Net Demonstration: Optimizing hidden layers runtime by removing embedded layer
    Preprocess:                 Stemming, No Stopword Removal, Hashtag and no Hashtag
    Vectorizer:                 Integer, 1-gram Tokenized
    Embedded Layer:             False
    Dataset Size:               24,784 Tweets
    Training Data:              67% 
    Testing Data:               33%
    Corpus Limit:               5000
    Input Sequence Size Limit:  30
    Hidden Layers:              0, 1, 3, 6
    Hidden Layer Size:          10
    Batch Size:                 8
    Epochs:                     3
    Activation Function:        ReLU
    Output Activation Function: Sigmoid
    Loss Function:              Binary Cross-Entropy
    Optimizer:                  Adam
    
    Run 1: 0 hidden layers
    Run 2: 1 hidden layers
    Run 3: 3 hidden layers
    Run 4: 6 hidden layers
    
    Returns
    -------
    None.
    
    """
    
    print("""
    Neural Net Demonstration: Optimizing hidden layers runtime by removing embedded layer
    Preprocess:                 Stemming, No Stopword Removal, Hashtag and no Hashtag
    Vectorizer:                 Integer, 1-gram Tokenized
    Embedded Layer:             False
    Dataset Size:               24,784 Tweets
    Training Data:              67% 
    Testing Data:               33%
    Corpus Limit:               5000
    Input Sequence Size Limit:  30
    Hidden Layers:              0, 1, 3, 6
    Hidden Layer Size:          10
    Batch Size:                 8
    Epochs:                     3
    Activation Function:        ReLU
    Output Activation Function: Sigmoid
    Loss Function:              Binary Cross-Entropy
    Optimizer:                  Adam
    
    Run 1: 0 hidden layers
    Run 2: 1 hidden layers
    Run 3: 3 hidden layers
    Run 4: 6 hidden layers
    """)
    
    print("\n\nRun 1: 0 Hidden Layers")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 0, 10, embed=False)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 2: 1 Hidden Layer")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 1, 10, embed=False)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 3: 3 Hidden Layers")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 3, 10, embed=False)    
    model.summary()
    model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=3)

    print("\n\nRun 4: 6 Hidden Layers")
    trainin, trainout, testin, testout = get_inputs_outputs('languageclass.csv',0.67, False, True, False)
    model = make_model(trainin, 5000, 30, 6, 10, embed=False)    
    model.summary()
    return model.fit(trainin, trainout, batch_size=8, validation_data=(testin, testout), epochs=5)



#Run Scripts
def run_all():
    ab = rand_model()
    plot_history(ab)
    
    a = baseline_model()
    plot_history(a)
    
    b = embedded_model()
    plot_history(b)
    
    dataset_info()
    pre_process_demonstration()
    
    e = pre_process_baseline()
    plot_history(e)
    
    f = pre_process_embedded()
    plot_history(f)
    
    a = hashtag_test() 
    plot_history(a)
    
    g = more_hidden_layers()
    plot_history(g)
    
    h = more_hidden_no_embed()
    plot_history(h)
    return

# ab = rand_model()
# plot_history(ab)

a = baseline_model()
plot_history(a)

# b = embedded_model()
# plot_history(b)

# dataset_info()
# pre_process_demonstration()

# e = pre_process_baseline()
# plot_history(e)

# f = pre_process_embedded()
# plot_history(f)

# a = hashtag_test() 
# plot_history(a)

# g = more_hidden_layers()
# plot_history(g)

# h = more_hidden_no_embed()
# plot_history(h)

def help():
    print("""
          List of available Demos:
          
          run_all()
              
          a = baseline_model()
          plot_history(a)

          b = embedded_model()
          plot_history(b)

          dataset_info()
          pre_process_demonstration()

          e = pre_process_baseline()
          plot_history(e)

          f = pre_process_embedded()
          plot_history(f)

          a = hashtag_test() 
          plot_history(a)

          g = more_hidden_layers()
          plot_history(g)

          h = optim_hidden_layers()
          plot_history(h)    
          """)