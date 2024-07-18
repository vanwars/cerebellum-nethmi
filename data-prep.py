from keras.datasets.mnist import load_data
from brian2 import seed
import utils
from time import sleep
import models

# loading the mnist dataset
# help(keras.datasets.mnist.load_data)
'''
1. Load the MNIST Dataset.
   This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
   along with a test set of 10,000 images. More info can be found at the MNIST homepage:
   http://yann.lecun.com/exdb/mnist/.

   Returns a tuple of numpy arrays of training and test data. Pixel values range from 0 to 255
'''
(training_bitmaps, training_labels), (test_bitmaps, test_labels) = load_data()

'''
2. Filter out some digits (only keep 4, 7, and 9):
'''
digits_to_exclude = [0, 1, 2, 3, 5, 6, 8]  # (only 4, 7, and 9 remain)
for digit in digits_to_exclude:
    training_bitmaps = training_bitmaps[(training_labels != digit)]
    training_labels = training_labels[(training_labels != digit)]
    test_bitmaps = test_bitmaps[(test_labels == digit)]
    test_labels = test_labels[(test_labels == digit)]

'''
3. Convert the bitmaps to Hz by dividing each pixel value by .
   Pixel intensity to Hz (255 becoms ~63Hz)
'''
training_bitmaps = training_bitmaps / 4
test_bitmaps = test_bitmaps / 4

'''
4. Display the first 5 digits in the set:
'''
utils.show_first_n_digits(training_bitmaps, training_labels,
                          n=5, sigfigs=2, convert_to_int=True)

'''
5. Define the training and testing proceedure:
'''
def train_model(train_items=1000, eval_items=1):

    seed(0)

    print('setting up the model...')
    model = models.CerebellarCircuitModel(debug=True)

    print('training the model...')
    model.train(training_bitmaps[:train_items], epoch=1)

    print('writing the model to disk...')
    model.net.store('train', 'train_4_7_9_epoch_1_eineuron_1000')
    # model.net.restore('train', 'trainall_but_0_epoch_1_eineuron_1000')

    print('evaluating the model with test data...')
    model.evaluate(test_bitmaps[:eval_items], test_bitmaps)


train_model()
