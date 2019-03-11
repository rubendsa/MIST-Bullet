import tensorflow as tf 
import numpy as np 
from pathlib import Path

#takes a placeholder with input size, and makes hidden layers based off of hidden_size array
def get_FC_model(input, scope, trainable, hidden_size=[64,64], output_size=4):
    with tf.variable_scope(scope):
        x = input 
        for size in hidden_size:
            x = tf.layers.dense(x, size, activation=tf.nn.tanh, trainable=trainable)
        output = tf.layers.dense(x, output_size, trainable=trainable)
    
    return output

def save_most_recent(save_path, saver, sess):
    save_name = str(
        save_path + "/" + "weights._most_recent_.ckpt"
    )
    saver.save(sess, save_name)
def save_model(save_path, saver, sess, cost):
    save_name = str(
        save_path + "/" + "weights.cost_{:.4f}_.ckpt".format(cost)
    )
    saver.save(sess, save_name)
    print("Model Saved with cost {}".format(cost))

def restore_most_recent(save_path, saver, sess):
    save_name = str(
        save_path + "/" + "weights._most_recent_.ckpt"
    )
    saver.restore(sess, save_name)

def restore_from_lowest_cost(save_path, saver, sess):
    save_path = Path(save_path)
    best_cost = 100000000
    try:
        if not save_path.exists():
            save_path.mkdir()

        weights_files = [wf for wf in save_path.glob("*.ckpt*") if not "_most_recent_" in str(wf)]
        if len(weights_files) == 0:
            raise IOError(
                "No weights to restore from at {0}".format(str(save_path)))
        weights_cost = [float(wf.name.split("_")[1]) for wf in weights_files]
        min_idx = np.argmin(weights_cost)
        best_cost = weights_cost[min_idx]
        best_weights_file = str(weights_files[min_idx])
        rmvidx = best_weights_file.index(".ckpt")
        best_weights_file = best_weights_file[:rmvidx+5]
        saver.restore(sess, best_weights_file)
        print("Successfully restored model with cost {:.4f}".format(
            best_cost))

    except IOError as e:
        print(e)

    return best_cost

def restore_from_highest_cost(save_path, saver, sess):
    save_path = Path(save_path)
    best_cost = -100000000
    try:
        if not save_path.exists():
            save_path.mkdir()

        weights_files = [wf for wf in save_path.glob("*.ckpt*") if not "_most_recent_" in str(wf)]
        if len(weights_files) == 0:
            raise IOError(
                "No weights to restore from at {0}".format(str(save_path)))
        weights_cost = [float(wf.name.split("_")[1]) for wf in weights_files]
        max_idx = np.argmax(weights_cost)
        best_cost = weights_cost[max_idx]
        best_weights_file = str(weights_files[max_idx])
        rmvidx = best_weights_file.index(".ckpt")
        best_weights_file = best_weights_file[:rmvidx+5]
        saver.restore(sess, best_weights_file)
        print("Successfully restored model with cost {:.4f}".format(
            best_cost))

    except IOError as e:
        print(e)

    return best_cost

if __name__ == "__main__":
    # sess = tf.Session()
    restore_from_lowest_cost(Path("./weights/"), None, None)
    



