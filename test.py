from cann import Invariants, PsiNet, CANN
import tensorflow as tf
from tensorflow import keras
import numpy as np

def test_invars():
    stretch = tf.constant(2.0)
    invars = Invariants()(stretch)
    tf.debugging.assert_equal(invars, tf.constant([5.0, 4.25]))

def test_PsiNet():
    invars = tf.constant([4.0, 4.0])
    psiNet = PsiNet()
    psiNet.set_weights([np.ones(4), np.ones(4), np.ones(8)])
    psi = psiNet(invars)
    tf.debugging.assert_equal(psi, tf.constant(10.873127))

def test_invars_ref():
    stretch = tf.constant(2.0)
    invars = Invariants()(stretch)
    invars = invars - 3
    tf.debugging.assert_equal(invars, tf.constant([2.0, 1.25]))

def test_model():
    stretch = tf.constant(2.0)
    psiNet = PsiNet()
    P1 = psiNet(stretch)

    print(P1)

def main() -> None:
    test_invars()
    test_PsiNet()
    test_invars_ref()
    test_model()

if __name__ == '__main__':
    main()