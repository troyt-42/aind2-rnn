import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Loop from s_0 to s_(p - window_size)
    for i in range(0, len(series) - window_size - 1):
        # Declare a temp variable
        X_temp = []
        # Put window_size number of elements following i-th element
        # into X_temp
        for p in range(0, window_size):
            X_temp.append(series[i + p])
        # Add X_temp to output X
        X.append(X_temp)
        # Add corresponding output value to y
        y.append(series[window_size + i])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # Declare a sequential model
    model = Sequential()
    # Add LSTM layer with 5 hidden units
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # Add a Dense layer with 1 node
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    print(sorted(list(set(text))))

    # remove as many non-english characters and character sequences as you can
    text = text.replace('-', ' ')
    text = text.replace('&', ' ')
    text = text.replace('à', 'a')
    text = text.replace('â', 'a')
    text = text.replace('è', 'e')
    text = text.replace('é', 'e')


    # shorten any extra dead space created above
    text = text.replace('  ',' ')

    print(sorted(list(set(text))))

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    inputs= [text[i:i+window_size] for i in range(0,len(text)-window_size, step_size)]
    outputs= [text[i+window_size] for i in range(0,len(text)-window_size, step_size)]

    # reshape each
    inputs = np.asarray(inputs)
    inputs.shape = (np.shape(inputs)[0:2])
    outputs = np.asarray(outputs)

    return inputs,outputs
