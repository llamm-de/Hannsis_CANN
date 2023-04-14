from cann import CANN
from tensorflow import keras
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt

def main() -> None:
    # Load filenames with weights of intermediate checkpoints
    weight_files = glob.glob('checkpoints/*.hdf5')
    weight_files.sort()

    # Load dataset from steinmann paper (Treloar data)
    with open('data/Treloar_steinmann.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data[0] # Uniaxial is saved at position 0 of the dataset
        x = np.float32(data[:,0])
        y = np.float32(data[:,1])
        x_test = np.linspace(x.min(), x.max(), 200)

    # Load pretrained model
    model = keras.models.load_model('model/cann.tf')

    # Predict results for intermediate checkpoints
    y_results = []
    for f in weight_files:
        model.load_weights(f)
        y_test = model.predict(x_test)
        y_results.append(y_test)

    # Plot results
    plt.plot(x, y, marker='x', markeredgecolor='r', linewidth=0, label="experiment")
    for i, y_test in enumerate(y_results):
        plt.plot(x_test, y_test, label=f"cp {i}")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()