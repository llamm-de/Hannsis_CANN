from tensorflow import keras
import tensorflow as tf

def activation_exp(x, a = 1.0, b = 0.0):
    """
    Exponential activation function: a*exp(x)+b
    """
    return a*tf.math.exp(x) + b


class Invariants(keras.layers.Layer):
    """
    Custom layer to compute invariants from stretch for uniaxial tension experiment
    """
    def __init__(self) -> None:
        super().__init__()

    def call(self, stretch):
        stretch_pow = tf.math.pow(stretch, 2)
        invariant_1 = stretch_pow + 2.0/stretch
        invariant_2 = 2*stretch + 1/stretch_pow
        return invariant_1, invariant_2
    

class PsiNet(keras.layers.Layer):
    """
    Custom layer to compute the strain energy density from invariants
    """
    def __init__(self, l2_factor = 0.001) -> None:
        super().__init__()

        # Weights for identity transformation
        self.w_identity = self.add_weight(shape = (4,1), 
                                 initializer = keras.initializers.GlorotNormal(), 
                                 constraint = tf.keras.constraints.NonNeg(), 
                                 regularizer = keras.regularizers.l2(l2_factor),
                                 trainable = True,
                                 name='w_identity')                         
        # Weights for exponential transformation
        self.w_exp = self.add_weight(shape = (4,1), 
                                 initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.00001), 
                                 constraint = tf.keras.constraints.NonNeg(), 
                                 regularizer = keras.regularizers.l2(l2_factor),
                                 trainable = True,
                                 name='w_exp')
        # Weights for summation to psi
        self.w_psi = self.add_weight(shape = (8,1), 
                                 initializer = keras.initializers.GlorotNormal(), 
                                 constraint = tf.keras.constraints.NonNeg(), 
                                 regularizer = keras.regularizers.l2(l2_factor),
                                 trainable = True,
                                 name='w_psi')  
        # Activation function for exponential activation
        self.activation_exp = keras.layers.Lambda(lambda x: activation_exp(x, b=-1.0))

    def call(self, invars):
        """
        Forward pass of the network
        """
        # get invariants in reference config
        invars_reference = invars - 3

        # Raise invariants to powers
        invars_pow_2 = tf.math.pow(invars_reference, 2.0)
        invar_powers = tf.concat([invars_reference, invars_pow_2], 0) # [I1-3, I2-3, (I1-3)^2, (I2-3)^2]

        # Multiply by weights
        powers_identity = tf.math.multiply(invar_powers, self.w_identity)
        powers_exp = tf.math.multiply(invar_powers, self.w_exp)
        
        # Apply activation functions to powers of invariants (only for exponential, since identity is directly given)
        powers_exp = self.activation_exp(powers_exp)

        # Concat results
        active_results = tf.concat([powers_identity, powers_exp], 0)

        # Multiply results with weights and add to strain energy using dot product (tf.tensordot)
        psi = tf.tensordot(tf.transpose(active_results), self.w_psi, axes=1)
        
        return psi


class CANN(keras.Model):
    """
    The CANN model
    """
    def __init__(self) -> None:
        super().__init__()
        # Include the predefined layers for invariant and psi computation to model
        self.invariant_layer = Invariants()
        self.psi_layer = PsiNet()

    def call(self, stretch):
        """
        Forward pass of the model
        """
        # Track invariants for later computaion of derivatives
        with tf.GradientTape(persistent=True) as tape:
            # Calculate invariants
            I1, I2 = self.invariant_layer(stretch)
            tape.watch(I1)
            tape.watch(I2)
            invars = tf.stack([I1, I2])
        
            # Calculate strain energy
            psi = self.psi_layer(invars)

        # Get derivatives as matrix including dPsi_dI1 and dPsi_dI2
        dPsi_dI1 = tape.gradient(psi, I1)
        dPsi_dI2 = tape.gradient(psi, I2)

        del tape

        # Calculate stress response
        P1 = 2*(dPsi_dI1 + 1/stretch * dPsi_dI2)*(stretch - 1/tf.math.pow(stretch, 2.0))

        return P1