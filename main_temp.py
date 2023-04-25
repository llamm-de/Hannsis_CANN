import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from cann import tempCANN
import matplotlib.pyplot as plt

def preprocess_data(data):
    # Split data into train and test data
    data_test = data[2]
    data_test_in = np.ndarray((data_test.shape[0], 2))
    data_test_label = np.ndarray((data_test.shape[0]))
    data_test_in[:,0:2] = data_test[:,0:2]
    data_test_label = data_test[:,2]
    test_dataset = tf.data.Dataset.from_tensor_slices((data_test_in, data_test_label))

    pred_x = np.linspace(data_test_in[:,0].max(), data_test_in[:,0].min(), 200)
    tmp = data_test_in[0,1] * np.ones((1, pred_x.size))
    prediction_inputs = np.hstack((pred_x.reshape(1, 200), tmp))
    pred_dataset = tf.data.Dataset.from_tensor_slices((prediction_inputs))

    data_train = np.vstack((data[0],data[1],data[3]))
    data_train_in = np.ndarray((data_train.shape[0], 2))
    data_train_label = np.ndarray((data_train.shape[0]))
    data_train_in[:,0:2] = data_train[:,0:2]
    data_train_label = data_train[:,2]
    train_dataset = tf.data.Dataset.from_tensor_slices((data_train_in, data_train_label))
    train_dataset = train_dataset.shuffle(data_train.shape[0] + 1)

    return test_dataset, train_dataset, pred_dataset, pred_x

def main(epochs:int = 3000, train_model:bool = True, save_model:bool = False) -> None:
    
    # Open data from Lion 1997
    # Array containing 4 numpy arrays with columns: ['stretch', 'temperture', '1. PK stress']
    with open('data/Lion1997.pkl', 'rb') as f:
        data = pickle.load(f)

    test_dataset, train_dataset, pred_dataset, pred_x = preprocess_data(data)

    # Train or load model
    if train_model:
        model = tempCANN()
        model.compile(optimizer=keras.optimizers.Adam(), 
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()]
                      )
        
        hist = model.fit(train_dataset, epochs=epochs)

        if save_model:
            model.save('model/tempCANN.tf')
    
    else:
        model = keras.models.load_model('model/tempCANN.tf')
    
    # Evaluate model
    y_pred = model.predict(pred_dataset)

    # Plot results
    fig, ax = plt.subplots(1,2,figsize=(10,6))
    if train_model:
        ax[0].plot(hist.history["loss"])
    ax[0].grid()
    ax[0].set_title('Training Error')
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE")
    ax[1].plot(pred_x, y_pred)
    # ax[1].plot(x, y, marker='x', markeredgecolor='r', linewidth=0)
    ax[1].set_title('Stress-strain prediction')
    ax[1].set_xlabel("stretch")
    ax[1].set_ylabel("stress")
    ax[1].legend(['predicted', 'experiment'])
    ax[1].grid()
    plt.show()


if __name__ == '__main__':
    main()