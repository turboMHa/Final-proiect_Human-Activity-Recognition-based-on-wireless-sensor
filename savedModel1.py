import tensorflow as tf
from lstm_architecture import one_hot
from sliding_window import sliding_window
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

meta_path = 'D:/gitclone/model/checkpoint/model3.ckpt.meta'
ckpt_path = 'D:/gitclone/model/checkpoint/model3.ckpt'
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

DATA_PATH = "data/"
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"

TRAIN = "train/"
TEST = "test/"

def opp_sliding_window(data_x, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_x= data_x.astype(np.float32)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
    return data_x


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001
    x /= (std * 2)  # 2 is for having smaller values
    return x


def load_X(X_signals_paths):
    """
    Given attribute (train or test) of feature, read all 9 features into an
    np ndarray of shape [sample_sequence_idx, time_step, feature_num]
        argument:   X_signals_paths str attribute of feature: 'train' or 'test'
        return:     np ndarray, tensor of features
    """
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_test = load_X(X_train_signals_paths)


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    """
    Read Y file of values to be predicted
        argument: y_path str attibute of Y: 'train' or 'test'
        return: Y ndarray / tensor of the 6 one_hot labels of each sample
    """
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_-1

y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_test = load_y(y_test_path)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess,ckpt_path)
    graph = tf.get_default_graph()

    X = graph.get_operation_by_name('X').outputs[0]
    Y = graph.get_operation_by_name('Y').outputs[0]
    is_train = graph.get_operation_by_name('is_train').outputs[0]



    feed_dict = {
        X: X_test,
        #Y: y_test,
        is_train: False
    }
    p = tf.get_collection('prob')[0]


    prediction = sess.run(p,feed_dict=feed_dict)
    final = np.array(prediction,dtype=np.int32)
    result = np.argmax(final,axis=1)+1
    result1 = np.argmax(final, axis=1)
    result = result.astype(np.int32)
    print(result)
    np.savetxt('3.txt',result)
    walking,walking_upstairs,walking_downstairs,sitting,standing,laying = 0,0,0,0,0,0
    for i in range(0,len(result)):
        if(result[i]==1):
            walking+=1
        if(result[i]==2):
            walking_upstairs+=1
        if(result[i]==3):
            walking_downstairs+=1
        if(result[i]==4):
            sitting+=1
        if(result[i]==5):
            standing +=1
        if(result[i]==6):
            laying +=1
    walking = walking*2.56
    walking_upstairs = walking_upstairs*2.56;
    walking_downstairs= walking_downstairs*2.56;
    sitting = sitting*2.56;
    standing = standing*2.56;
    laying = laying*2.56;
    if(walking!=0):
        print("You have been walking for %d seconds"%walking)
    if(walking_upstairs!=0):
        print("You have been walking upstairs for %d seconds"%walking_upstairs)
    if (walking_downstairs != 0):
        print("You have been walking downstairs for %d seconds" % walking_downstairs)
    if (sitting != 0):
        print("You have been sitting for %d seconds" % sitting)
    if (standing != 0):
        print("You have been standing for %d seconds" % standing)
    if (laying != 0):
        print("You have been laying for %d seconds" % laying)

print("Confusion Matrix:")
#result = load_y("D:/gitclone/3.txt")
print(y_test)
confusion_matrix = metrics.confusion_matrix(y_test, result1)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results:
width = 6
height = 6
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(6)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()