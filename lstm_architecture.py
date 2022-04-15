import tensorflow as tf
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


def one_hot(y):

    y = y.reshape(len(y))
    n_values = np.max(y) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]


def batch_norm(input_tensor, config, i):

    if config.n_layers_in_highway == 0:
        return input_tensor

    with tf.variable_scope("batch_norm") as scope:
        if i != 0:
            scope.reuse_variables()

        axes = list(range(len(input_tensor.get_shape())))

        mean, variance = tf.nn.moments(input_tensor, axes=axes, shift=None, name=None, keep_dims=False)
        stdev = tf.sqrt(variance+0.001)

        # Rescaling
        bn = input_tensor - mean
        bn /= stdev

        bn *= tf.get_variable("a_noreg", initializer=tf.random_normal([1], mean=0.5, stddev=0.0))
        bn += tf.get_variable("b_noreg", initializer=tf.random_normal([1], mean=0.0, stddev=0.0))

    return bn

def relu_fc(input_2D_tensor_list, features_len, new_features_len, config):
    """make a relu fully-connected layer, mainly change the shape of tensor
       both input and output is a list of tensor
        argument:
            input_2D_tensor_list: list shape is [batch_size,feature_num]
            features_len: int the initial features length of input_2D_tensor
            new_feature_len: int the final features length of output_2D_tensor
            config: Config used for weights initializers
        return:
            output_2D_tensor_list lit shape is [batch_size,new_feature_len]
    """

    W = tf.get_variable(
        "relu_fc_weights",

        initializer=tf.random_normal(
            [features_len, new_features_len],
            mean=0.0,
            stddev=float(config.weights_stddev)
        )
    )
    b = tf.get_variable(
        "relu_fc_biases_noreg",
        initializer=tf.random_normal(
            [new_features_len],
            mean=float(config.bias_mean),
            stddev=float(config.weights_stddev)
        )
    )
    print(input_2D_tensor_list)

    output_2D_tensor_list = [
        tf.nn.relu(tf.matmul(input_2D_tensor, W) + b)
            for input_2D_tensor in input_2D_tensor_list
    ]

    return output_2D_tensor_list


def single_LSTM_cell(input_hidden_tensor, n_outputs):
    with tf.variable_scope("lstm_cell"):
        #lstm_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(n_outputs, state_is_tuple=True, forget_bias=0.999)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_outputs, state_is_tuple=True, forget_bias=0.999)
        outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, input_hidden_tensor, dtype=tf.float32)
    return outputs


def bi_LSTM_cell(input_hidden_tensor, n_inputs, n_outputs, config):
    n_outputs = int(n_outputs/2)

    print("bidir:")

    with tf.variable_scope(name_or_scope='pass_forward',reuse=tf.AUTO_REUSE) as scope2:

        hidden_forward = relu_fc(input_hidden_tensor, n_inputs, n_outputs, config)
        forward = list(single_LSTM_cell(list(hidden_forward), n_outputs))

    print (len(hidden_forward), str(hidden_forward[0].get_shape()))

    # Backward pass is as simple as surrounding the cell with a double inversion:
    with tf.variable_scope('pass_backward',reuse=tf.AUTO_REUSE) as scope2:
        hidden_backward = relu_fc(input_hidden_tensor, n_inputs, n_outputs, config)
        backward = list(reversed(single_LSTM_cell(list(reversed(hidden_backward)), n_outputs)))

    with tf.variable_scope('bidir_concat',reuse=tf.AUTO_REUSE) as scope:
        # Simply concatenating cells' outputs at each timesteps on the innermost dimension, like if the two cells acted as one cell with twice the n_hidden size:
        layer_hidden_outputs = [
            tf.concat([f, b],len(f.get_shape()) - 1)
                for f, b in zip(forward, backward)]

    return layer_hidden_outputs


def residual_bidirectional_LSTM_layers(input_hidden_tensor, n_input, n_output, layer_level, config, keep_prob_for_dropout):

    with tf.variable_scope('layer_{}'.format(layer_level)) as scope:

        if config.use_bidirectionnal_cells:
            get_lstm = lambda input_tensor: bi_LSTM_cell(input_tensor, n_input, n_output, config)
        else:
            get_lstm = lambda input_tensor: single_LSTM_cell(relu_fc(input_tensor, n_input, n_output, config), n_output)
        def add_highway_redisual(layer, residual_minilayer):
            return [a + b for a, b in zip(layer, residual_minilayer)]

        hidden_LSTM_layer = get_lstm(input_hidden_tensor)
        for i in range(config.n_layers_in_highway - 1):
            with tf.variable_scope('LSTM_residual_{}'.format(i)) as scope2:
                hidden_LSTM_layer = add_highway_redisual(
                    hidden_LSTM_layer,
                    get_lstm(input_hidden_tensor)
                )

        if config.also_add_dropout_between_stacked_cells:
            hidden_LSTM_layer = [tf.nn.dropout(out, keep_prob_for_dropout) for out in hidden_LSTM_layer]

        return [batch_norm(out, config, i) for i, out in enumerate(hidden_LSTM_layer)]


def LSTM_network(feature_mat, config, keep_prob_for_dropout):

    with tf.variable_scope('LSTM_network') as scope:  # TensorFlow graph naming

        feature_mat = tf.nn.dropout(feature_mat, keep_prob_for_dropout)

        feature_mat = tf.transpose(feature_mat, [1, 0, 2])
        print (feature_mat.get_shape())
        # New feature_mat's shape: [time_steps, batch_size, n_inputs]

        # Temporarily crush the feature_mat's dimensions
        feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
        print (feature_mat.get_shape())
        # New feature_mat's shape: [time_steps*batch_size, n_inputs]

        # Split the series because the rnn cell needs time_steps features, each of shape:
        hidden = tf.split(feature_mat, config.n_steps, 0)
        print (len(hidden), str(hidden[0].get_shape()))
        # New shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

        # Stacking LSTM cells, at least one is stacked:
        print ("\nCreating hidden #1:")
        hidden = residual_bidirectional_LSTM_layers(hidden, config.n_inputs, config.n_hidden, 1, config, keep_prob_for_dropout)
        print (len(hidden), str(hidden[0].get_shape()))

        for stacked_hidden_index in range(config.n_stacked_layers - 1):
            # If the config permits it, we stack more lstm cells:
            print ("\nCreating hidden #{}:".format(stacked_hidden_index+2))
            hidden = residual_bidirectional_LSTM_layers(hidden, config.n_hidden, config.n_hidden, stacked_hidden_index+2, config, keep_prob_for_dropout)
            print (len(hidden), str(hidden[0].get_shape()))

        print ("")

        last_hidden = tf.nn.dropout(hidden[-1], keep_prob_for_dropout)
        last_logits = relu_fc(
            [last_hidden],
            config.n_hidden, config.n_classes, config
        )[0]
        return last_logits


def run_with_config(Config, X_train, y_train, X_test, y_test):
    tf.reset_default_graph()  # To enable to run multiple things in a loop

    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    with tf.device("/cpu:0"):  # Remove this line to use GPU. If you have a too small GPU, it crashes.
        X = tf.placeholder(tf.float32, [
                           None, config.n_steps, config.n_inputs], name="X")
        Y = tf.placeholder(tf.float32, [
                           None, config.n_classes], name="Y")

        # is_train for dropout control:
        is_train = tf.placeholder(tf.bool, name="is_train")
        keep_prob_for_dropout = tf.cond(is_train,
            lambda: tf.constant(
                config.keep_prob_for_dropout,
                name="keep_prob_for_dropout"
            ),
            lambda: tf.constant(
                1.0,
                name="keep_prob_for_dropout"
            )
        )

        pred_y = LSTM_network(X, config, keep_prob_for_dropout)
        tf.add_to_collection('prob', pred_y)
        # Loss, optimizer, evaluation

        # Softmax loss with L2 and L1 layer-wise regularisation
        print ("Unregularised variables:")
        for unreg in [tf_var.name for tf_var in tf.trainable_variables() if ("noreg" in tf_var.name or "Bias" in tf_var.name)]:
            print (unreg)
        l2 = config.lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred_y, labels=Y)) + l2  # + l1

        optimize = tf.contrib.layers.optimize_loss(
            loss,
            global_step=tf.Variable(0),
            learning_rate=config.learning_rate,
            optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
            clip_gradients=config.clip_gradients,
            gradient_noise_scale=config.gradient_noise_scale
        )

        correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


    losses = []
    accuracies = []
    saver_path = 'D:/gitclone/model/checkpoint/model3.ckpt'
    saver = tf.train.Saver()
    sessconfig = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=sessconfig) as sess:
        tf.initialize_all_variables().run()

        best_accuracy = (0.0, "iter: -1")
        best_f1_score = (0.0, "iter: -1")

        # Start training for each batch and loop epochs

        worst_batches = []

        for i in range(config.training_epochs):

            # Loop batches for an epoch:
            shuffled_X, shuffled_y = shuffle(X_train, y_train, random_state=i*42)
            for start, end in zip(range(0, config.train_count, config.batch_size),
                                  range(config.batch_size, config.train_count + 1, config.batch_size)):

                _, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, loss, pred_y],
                    feed_dict={
                        X: shuffled_X[start:end],
                        Y: shuffled_y[start:end],
                        is_train: True
                    }
                )

                worst_batches.append(
                    (train_loss, shuffled_X[start:end], shuffled_y[start:end])
                )
                worst_batches = list(sorted(worst_batches))[-5:]  # Keep 5 poorest

            # Train F1 score is not on boosting
            train_f1_score = metrics.f1_score(
                shuffled_y[start:end].argmax(1), train_pred.argmax(1), average="weighted"
            )

            for _, x_, y_ in worst_batches:

                _, train_acc, train_loss, train_pred = sess.run(
                    [optimize, accuracy, loss, pred_y],
                    feed_dict={
                        X: x_,
                        Y: y_,
                        is_train: True
                    }
                )

            pred_out, accuracy_out, loss_out = sess.run(
                [pred_y, accuracy, loss],
                feed_dict={
                    X: X_test,
                    Y: y_test,
                    is_train: False
                }
            )


            f1_score_out = metrics.f1_score(
                y_test.argmax(1), pred_out.argmax(1), average="weighted"
            )
            losses.append(loss_out)
            accuracies.append(accuracy_out)
            print (
                "iter: {}, ".format(i) + \
                "train loss: {}, ".format(train_loss) + \
                "train accuracy: {}, ".format(train_acc) + \
                "train F1-score: {}, ".format(train_f1_score) + \
                "test loss: {}, ".format(loss_out) + \
                "test accuracy: {}, ".format(accuracy_out) + \
                "test F1-score: {}".format(f1_score_out)
            )

            best_accuracy = max(best_accuracy, (accuracy_out, "iter: {}".format(i)))
            best_f1_score = max(best_f1_score, (f1_score_out, "iter: {}".format(i)))


        print("")
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("final F1 score: {}".format(f1_score_out))
        print("best epoch's F1 score: {}".format(best_f1_score))
        print("")
        saved_path = saver.save(sess, saver_path)
        print("Model saved in file:", saver_path)

        font = {
            'family': 'Bitstream Vera Sans',
            'weight': 'bold',
            'size': 18
        }
        matplotlib.rc('font', **font)

        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        indep_test_axis = np.append(
            np.array(range(100, len(losses) * 250, 250)[:-1]),
            [250]
        )
        plt.plot(indep_test_axis, np.array(losses), "b-", label="Test losses")

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Loss)')
        plt.xlabel('Training iteration')

        plt.show()

        font = {
            'family': 'Bitstream Vera Sans',
            'weight': 'bold',
            'size': 18
        }
        matplotlib.rc('font', **font)

        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        indep_test_axis = np.append(
            np.array(range(100, len(losses) * 250, 250)[:-1]),
            [250]
        )
        plt.plot(indep_test_axis, np.array(accuracies), "g-", label="Test accuracies")

        plt.title("Training session's progress over iterations")
        plt.legend(loc='upper right', shadow=True)
        plt.ylabel('Training Progress (Accuracy)')
        plt.xlabel('Training iteration')

        plt.show()
    return accuracy_out, best_accuracy, f1_score_out, best_f1_score, pred_out

