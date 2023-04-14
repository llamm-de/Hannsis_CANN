from cann import CANN
import pickle
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

TRAIN_MODEL = True
SAVE_MODEL = True
TRAINING_EPOCHS = 3000

def main() -> None:
    # Load dataset from steinmann paper (Treloar data)
    with open('data/Treloar_steinmann.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data[0] # Uniaxial is saved at position 0 of the dataset
        x = np.float32(data[:,0])
        y = np.float32(data[:,1])


    if TRAIN_MODEL:
        # Create CANN model
        model = CANN()
        model.compile(optimizer=keras.optimizers.Adam(), 
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()]
                      )
        
        # Set checkpoint callback to save intermediate results (e.g. for visualization of training process)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/weights-{epoch:02d}.hdf5', 
            save_weights_only=True,
            save_freq=250
        )
        
        # Train the model
        model_hist = model.fit(x, y, batch_size=1, epochs=TRAINING_EPOCHS, callbacks=[checkpoint_callback])

        # Save trained model
        if SAVE_MODEL:
            model.save('model/cann.tf')
    else:
        # Load pretrained model
        model = keras.models.load_model('model/cann.tf')

    # Predict data with trained model
    x_test = np.linspace(x.min(), x.max(), 200)
    y_test = model.predict(x_test)

    # Plot results
    fig, ax = plt.subplots(1,2,figsize=(10,6))
    if TRAIN_MODEL:
        ax[0].plot(model_hist.history["loss"])
    ax[0].grid()
    ax[0].set_title('Training Error')
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE")
    ax[1].plot(x_test, y_test)
    ax[1].plot(x, y, marker='x', markeredgecolor='r', linewidth=0)
    ax[1].set_title('Stress-strain prediction')
    ax[1].set_xlabel("stretch")
    ax[1].set_ylabel("stress")
    ax[1].legend(['predicted', 'experiment'])
    ax[1].grid()
    plt.show()
       

if __name__ == '__main__':
    main()
