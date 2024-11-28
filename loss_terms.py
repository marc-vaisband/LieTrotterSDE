"""
This file contains the implementations of the different loss terms used.
"""

import tensorflow as tf
import numpy as np

"""
For simulating new paths within the loss function evaluation,
we need to use a simlator that can be compiled into a TensorFlow graph.
This motivates the slightly ugly implementation via tf.scan below.
"""
@tf.function()
def tensorflow_model_euler_tih(model, delta_t, tspan, y0):
    sqrt_delta_t = tf.math.sqrt(delta_t)
    def evolve(a, x):
        model_prediction = model(tf.transpose(a), training=True)
        drift_addition = tf.scalar_mul(delta_t, model_prediction[:, 0])
        diff_addition = tf.multiply(tf.random.normal(shape=y0.shape, mean=0.0, stddev=sqrt_delta_t, dtype=tf.float32), model_prediction[:, 1])

        return a + tf.stack([drift_addition + diff_addition, tf.scalar_mul(delta_t, tf.ones(y0.shape))], axis=0)

    return tf.transpose(tf.scan(evolve, tspan, initializer=tf.stack([y0, tf.zeros(y0.shape)], axis=0))[:, 0])

"""
For the Euler-Maruyama loss, we only need to calculate the residuals of the Euler-Maruyama scheme,
as they are assumed, by the pseudo-likelihood formulation, to be Gaussian.
"""
@tf.function()
def get_variance_penalty(y_true, y_pred):
    var_penalty_indep = tf.math.reduce_mean(
        tf.math.log(
            tf.math.scalar_mul(
                2 * np.pi,
                y_true[:, 1],  # delta_t
            )
        )
    )
    var_penalty_dep = tf.math.reduce_mean(
        tf.math.scalar_mul(
            2.0,
            tf.math.log(y_pred[:, 1])
        )
    )
    var_penalty = var_penalty_dep + var_penalty_indep

    return var_penalty

@tf.function()
def get_residuals(y_true, y_pred):
    return tf.cast(tf.math.divide(
                        tf.math.subtract(
                            y_true[:, 0],  # d, i.e. x increments
                            tf.multiply(
                                y_true[:, 1],  # t increments
                                y_pred[:, 0]  # a
                            )
                        ),
                        tf.math.multiply(
                            tf.math.sqrt(y_true[:, 1]),  # For std. dev, we have sqrt(delta)
                            y_pred[:, 1]
                        )
                    ), tf.float64)


@tf.function()
def square_and_penalise_residuals(residuals):
    squared_residuals = tf.math.square(residuals)
    return tf.reduce_mean(squared_residuals)

@tf.function()
def get_residual_penalty(y_true, y_pred):
    residual_penalty = square_and_penalise_residuals(get_residuals(y_true, y_pred))
    return residual_penalty
        
@tf.function()
def euler_loss(y_true, y_pred):
    return tf.math.add(
                tf.cast(get_variance_penalty(y_true, y_pred), tf.float32),
                tf.cast(get_residual_penalty(y_true, y_pred), tf.float32)
        ) / tf.cast(tf.shape(y_true)[0], tf.float32)


"""
The implementation of the Lie-Trotter splitting pseudo-likelihood is slightly more complex
as it involves the derivative of a with respect to x. For this, we need to use a gradient tape
"""
@tf.function()
def lt_splitting_loss(y_true, y_pred, grad_a_pred):
    # y_true[:, 0] are the increments in x, as before
    # y_true[:, 1] are the delta_t
    
    variance_penalty_indep = tf.math.reduce_sum(tf.math.log(y_true[:, 1]))

    log_b = tf.math.log(y_pred[:, 1])

    variance_penalty_dep = tf.math.scalar_mul(2.0, 
                                              tf.math.reduce_sum(log_b))

    increments = tf.cast(y_true[:, 0], tf.float32)
    
    first_order_diff = increments - tf.math.multiply(tf.cast(y_true[:, 1], tf.float32), y_pred[:, 0])

    second_order_diff = first_order_diff - tf.math.scalar_mul(0.5, tf.math.multiply(tf.math.square(y_true[:, 1]), tf.math.multiply(y_pred[:, 0], grad_a_pred))) 

    increment_penalties = tf.math.divide(
        tf.math.square(second_order_diff),
        tf.cast(tf.math.multiply(
            tf.cast(y_true[:, 1], tf.float32),
            tf.cast(tf.math.square(y_pred[:, 1]), tf.float32)
        ), tf.float32)
    )
    
    increment_penalty = tf.math.reduce_sum(increment_penalties)
    
    result = (variance_penalty_indep + variance_penalty_dep + increment_penalty) / tf.cast(tf.shape(y_true)[0], tf.float32)

    return result


"""
To calculate the Wasserstein distance between two empirical distributions
in an autograph-compatible way, we adapt the standard Scipy implementation
into TensorFlow functions.
"""
@tf.function()
def tf_wasserstein1D(u_values, v_values):
    u_sorter = tf.argsort(u_values, axis=-1)
    v_sorter = tf.argsort(v_values, axis=-1)

    all_values = tf.concat([u_values, v_values], axis=-1)
    all_values = tf.sort(all_values, axis=-1)

    deltas = tf.experimental.numpy.diff(all_values, axis=-1)

    u_cdf_indices = tf.searchsorted(tf.gather(u_values, u_sorter), all_values[:-1], 'right')
    v_cdf_indices = tf.searchsorted(tf.gather(v_values, v_sorter), all_values[:-1], 'right')

    u_cdf = u_cdf_indices / tf.shape(u_values)[-1]
    v_cdf = v_cdf_indices / tf.shape(v_values)[-1]

    return tf.reduce_sum(tf.cast(tf.abs(u_cdf - v_cdf), tf.float32) * deltas)

@tf.function()
def wasserstein_1D_on_concat(uv_tensor):
    return tf_wasserstein1D(uv_tensor[:, 0], uv_tensor[:, 1])


@tf.function()
def get_wassersteins(X, Y):
    return tf.map_fn(wasserstein_1D_on_concat, elems=tf.transpose(tf.stack([X, Y], axis=2), perm=[1, 0, 2]))