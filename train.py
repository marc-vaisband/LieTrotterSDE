"""
This file contains the top-level utilities for training the models
used in the study. The main function is `run_procedure_outer`, which
relies on a lot of default parameters and simply generates a model given
a problem and a method key.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import os
from utils.input_preparation import sample_matrix_to_det_NN_input
import sys
import argparse
from utils import *
from loss_terms import *

LRPLATEAU_DELTA = 0.2

"""
The train function is the main function for training the models. For 
the Euler-Maruyama, Lie-Trotter and Wasserstein losses, it uses the more involved
utilities from loss_terms.py. The Moment and Autocorrelation losses are implemented
directly in the function. 
"""
def train(model, train_samples, data_means, data_stds, y0_reserve, delta_t, tspan, opt,
                      train_x, train_y, epochs=100, n_samples_generated=100, batch_size=1024,
                      euler_term=None, lt_term=None, moment_term=None, autocorrelation_term=None, wasserstein_term=None):
    
    generate_samples = (moment_term is not None or wasserstein_term is not None or autocorrelation_term is not None)
    
    
    data_means = tf.convert_to_tensor(data_means, dtype=tf.float32)
    data_stds = tf.convert_to_tensor(data_stds, dtype=tf.float32)
    data_vars = tf.math.square(data_stds)
    y0_tensor = tf.convert_to_tensor(y0_reserve[:], dtype=tf.float32)
    universal_logits = tf.ones((1, y0_tensor.shape[0]))
    
    train_samples_tensor = tf.convert_to_tensor(train_samples, dtype=tf.float32)
    train_sample_autocorrelations = tfp.stats.auto_correlation(train_samples_tensor, axis=1, max_lags=10)

    train_x_tensor = tf.convert_to_tensor(train_x, dtype=tf.float32)
    train_y_tensor = tf.convert_to_tensor(train_y, dtype=tf.float32)
    
    @tf.function
    def universal_loss(y_true_dummy, y_pred_dummy):
        y_true_ints = tf.cast(y_true_dummy[:, 0], tf.int32)
        # y_true_dummy are the indices of the batch!
        batch_x_tensor = tf.gather(train_x_tensor, y_true_ints)
        batch_y_tensor = tf.gather(train_y_tensor, y_true_ints)

        current_loss_value = 0.0
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:                               
            tape.watch(batch_x_tensor)
            # Feed forward
            output = tf.cast(model(batch_x_tensor, training=True), tf.float32)

            a_output = output[:, 0]
            b_output = output[:, 1]

            if euler_term is not None:
                euler_loss_addition = euler_loss(batch_y_tensor, output)
                current_loss_value += euler_term * euler_loss_addition
                
                
            if lt_term is not None:
                # LT splitting requires the gradient of a_output w.r.t. batch_x_tensor
                del_x_a = tape.gradient(a_output, batch_x_tensor)[:, 0]
                lt_loss_addition = lt_splitting_loss(batch_y_tensor, output, del_x_a)
                current_loss_value += lt_term * lt_loss_addition
                
            if generate_samples:
                y0_idc = tf.random.categorical(logits=universal_logits, num_samples=n_samples_generated)[0]
                new_samples = tensorflow_model_euler_tih(model, delta_t, tspan, tf.gather(y0_tensor, y0_idc))
                train_samples_used = tf.gather(train_samples_tensor, y0_idc)
                
                if moment_term is not None:
                    new_sample_means = tf.math.reduce_mean(new_samples, axis=0)
                    new_sample_variances = tf.math.reduce_variance(new_samples, axis=0)

                    expectation_loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(
                                            new_sample_means,
                                            data_means
                                        )))

                    variance_loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(
                                            new_sample_variances,
                                            data_vars
                                        )))
                    
                    current_loss_value += moment_term * (expectation_loss + variance_loss)
                    
                if autocorrelation_term is not None:
                    
                    new_sample_autocorrelations = tfp.stats.auto_correlation(new_samples, axis=1, max_lags=10)
                    
                    autocorrelation_loss = tf.math.reduce_sum(tf.math.square(
                        tf.math.reduce_mean(new_sample_autocorrelations, axis=0) - tf.math.reduce_mean(train_sample_autocorrelations, axis=0)
                    ))
                
                    current_loss_value += autocorrelation_term * autocorrelation_loss
                    
                    
                if wasserstein_term is not None:
                    wassersteins = get_wassersteins(new_samples, train_samples_used)
                    
                    wasserstein_loss = tf.math.reduce_mean(wassersteins) + tf.math.reduce_max(wassersteins)      
                                      
                    current_loss_value += wasserstein_term * wasserstein_loss
                    
        return current_loss_value
    
    
    if generate_samples:
        pretrain_model_for_init(model, tspan)
        print("Pre-training done")
    
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=1/np.sqrt(2), patience=10, verbose=1, min_delta=0.2,
                                               mode='min')
    model.compile(loss=universal_loss, optimizer=opt)
    
    idx_tensor = tf.range(train_x.shape[0], dtype=tf.int32)
    doubled_idx_tensor = tf.stack([idx_tensor, idx_tensor], axis=1)
    t0 = time.time()
    _ = model.fit(train_x, doubled_idx_tensor, epochs=epochs, batch_size=batch_size, callbacks=[lr_callback])
    t1 = time.time()
    print(f"Time elapsed: {t1 - t0}")
    return _       


def run_procedure(
                  train_samples_path, 
                  test_samples_path, 
                  tspan_path, 
                  test_bms_path,
                  generated_samples_out_path, 
                  model_save_path, 
                  unit_nums=None, 
                  n_epochs=200, 
                  batch_size=16*1024,
                  n_samples_generated=200, 
                  opt=None,
                  euler_term=None,
                  lt_term=None,
                  moment_term=None,
                  wasserstein_term=None,
                  autocorrelation_term=None):
    
    if opt is None:
        opt = tf.keras.optimizers.Adam()
    if unit_nums is None:
        unit_nums = [16, 64, 64, 16]

    train_samples = np.load(train_samples_path)
    train_y0 = train_samples[:, 0]
    test_samples = np.load(test_samples_path)
    t_hrs = np.load(tspan_path)
    test_bms = np.load(test_bms_path)
    delta_t = tf.cast(t_hrs[1] - t_hrs[0], tf.float32)
    tspan = tf.convert_to_tensor(t_hrs)

    train_sample_maxval = np.amax(np.abs(train_samples).flatten())

    train_samples = train_samples / train_sample_maxval
    test_samples = test_samples / train_sample_maxval

    data_means = np.average(train_samples, axis=0)
    data_stds = np.std(train_samples, axis=0)

    np.random.seed(2024)
    tf.keras.utils.set_random_seed(2024)

    train_x, train_y = sample_matrix_to_det_NN_input(train_samples, t_hrs, time_homogenous=False)

    naive_dense_model = get_naive_dense_model(delta_t=delta_t, unit_nums=unit_nums, time_homogenous=False,
                                              optimizer="adam")


    _ = train(model=naive_dense_model,
              train_samples=train_samples,
              data_means=data_means,
              data_stds=data_stds,
              y0_reserve=train_y0,
              delta_t=delta_t,
              tspan=tspan,
              opt=opt,
              train_x=train_x,
              train_y=train_y,
              epochs=n_epochs,
              n_samples_generated=n_samples_generated,
              batch_size=batch_size,
              euler_term=euler_term,
              lt_term=lt_term,
              moment_term=moment_term,
              wasserstein_term=wasserstein_term,
              autocorrelation_term=autocorrelation_term)

    batch_model_a = lambda x_vec, t: naive_dense_model.predict(np.stack([x_vec, t * np.ones(x_vec.shape)], axis=1),
                                                               verbose=False)[:, 0]
    batch_model_b = lambda x_vec, t: naive_dense_model.predict(np.stack([x_vec, t * np.ones(x_vec.shape)], axis=1),
                                                                   verbose=False)[:, 1]

    test_u0 = test_samples[:, 0] / train_sample_maxval
    synth_samples = one_step_euler(batch_model_a, batch_model_b, t_hrs, test_bms, test_u0)
    synth_samples = synth_samples * train_sample_maxval

    naive_dense_model.save(model_save_path)
    np.save(generated_samples_out_path, synth_samples)
                                                    

N_SAMPLES_GENERATED = 200

loss_term_dict = {
    "euler": {
        "euler_term": 10.0,
    },
    "lt": {
        "lt_term": 10.0,
    },
    "moment": {
        "moment_term": 1.0,
    },    
    "wasserstein": {
        "wasserstein_term": 1.0,
    },
    "corr": {
        "autocorrelation_term": 1.0,
    },
    
    "euler_moment": {
        "euler_term": 10.0,
        "moment_term": 1.0,
    },
    "euler_wasserstein": {
        "euler_term": 10.0,
        "wasserstein_term": 1.0,
    },
    "euler_corr": {
        "euler_term": 10.0,
        "autocorrelation_term": 1.0,
    },
    
    "lt_moment": {
        "lt_term": 10.0,
        "moment_term": 1.0,
    },
    "lt_wasserstein": {
        "lt_term": 10.0,
        "wasserstein_term": 1.0,
    },    
    "lt_corr": {
        "lt_term": 10.0,
        "autocorrelation_term": 1.0,
    },
    
    "euler_moment_corr": {
        "euler_term": 10.0,
        "moment_term": 1.0,
        "autocorrelation_term": 1.0,
    },
    "euler_wasserstein_corr": {
        "euler_term": 10.0,
        "wasserstein_term": 1.0,
        "autocorrelation_term": 1.0,
    },
    
    "lt_moment_corr": {
        "lt_term": 10.0,
        "moment_term": 1.0,
        "autocorrelation_term": 1.0,
    },
    "lt_wasserstein_corr": {
        "lt_term": 10.0,
        "wasserstein_term": 1.0,
        "autocorrelation_term": 1.0,
    }
}

def run_procedure_outer(problem, key, **kwargs):
    print(f"Running procedure {key}")
    
    if not os.path.exists(f"./results/{problem}_test"):
        os.makedirs(f"./results/{problem}_test")
        
    if not os.path.exists(f"./results/{problem}_test/{key}"):
        os.makedirs(f"./results/{problem}_test/{key}")
    
    units_string = "_".join([str(x) for x in kwargs["unit_nums"]])
    epochs_string = str(kwargs["n_epochs"])
    
    run_procedure(
        train_samples_path=f"./problems/{problem}_test/{problem}_train_samples.npy",
        test_samples_path=f"./problems/{problem}_test/{problem}_test_samples.npy",
        tspan_path=f"./problems/{problem}_test/{problem}_tspan.npy",
        train_bms_path=f"./problems/{problem}_test/{problem}_train_bms.npy",
        test_bms_path=f"./problems/{problem}_test/{problem}_test_bms.npy",
        generated_samples_out_path=f"./results/{problem}_test/{key}/generated_samples.npy",
        model_save_path=f"./results/{problem}_test/{key}/model_{units_string}_{epochs_string}ep",
        **kwargs
    )

"""
Admissible problem keys are:
- "ou"
- "cir"
- "sin",
- "sit",
- "gfp"

Admissible method keys are the keys of the loss_term_dict dictionary.

Example usage:
python train.py ou euler
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="Specify the problem")
    parser.add_argument("keys", nargs='+', help="Specify the method keys")
    args = parser.parse_args()

    for key in args.keys:
        vals = loss_term_dict.get(key)
        if vals is not None:
            run_procedure_outer(
                args.problem,
                key, 
                unit_nums=[16, 64, 64, 16],
                opt=tf.keras.optimizers.Adam(learning_rate=1e-3),
                batch_size=16 * 1024,
                n_samples_generated=N_SAMPLES_GENERATED,
                n_epochs=100,
                **vals)
        else:
            print(f"Invalid key: {key}")