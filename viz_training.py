import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

def create_checkpoint_predictions():
    from cann import CANN
    from tensorflow import keras
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

    with open('checkpoints/intermediate_results.pkl', 'wb') as f:
        pickle.dump([x_test, y_results], f)


def create_animation(factor, save_gif):
    # Load dataset from steinmann paper (Treloar data)
    with open('data/Treloar_steinmann.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data[0] # Uniaxial is saved at position 0 of the dataset
        x = np.float32(data[:,0])
        y = np.float32(data[:,1])
        x_test = np.linspace(x.min(), x.max(), 200)

    with open('checkpoints/intermediate_results.pkl', 'rb') as f:
        data = pickle.load(f)
        x_test = data[0]
        y_results = data[1]

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='x', markeredgecolor='r', linewidth=0, label="experiment")
    ax.legend()
    line, = ax.plot(x_test, y_results[0], label=f"cann simulation")
    text = ax.text(5.5, 0, '')

    def animate(i):
        line.set_ydata(y_results[i])
        ax.legend()
        text.set_text(f"Training Epoch: {i*factor}")
        return line, text,

    ani = animation.FuncAnimation(fig, animate, frames=len(y_results), repeat=False, interval=200, blit=True, save_count=50)

    plt.show()

    if save_gif:
        ani.save('checkpoints/animation.gif', writer='imagemagick', fps=10)

def main() -> None:

    NEW_PREDICTIONS = False
    FACTOR = 10
    SAVE_GIF = True
    
    if NEW_PREDICTIONS:
        create_checkpoint_predictions()
    create_animation(FACTOR, SAVE_GIF)

if __name__ == '__main__':
    main()