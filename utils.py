import numpy as np
import tensorflow as tf

def sample_matrix_to_det_NN_input(sample_matrix, timepoints, time_homogenous=True):
    """
    In the deterministic NN, we train for the likelihood of increments given functions a, b output by the NN.

    :param sample_matrix:
    :param timepoints:
    :return:
    """
    x_plus = sample_matrix[:, 1:]  # without first timepoint
    x_minus = sample_matrix[:, :-1]  # without last timepoint
    d = x_plus - x_minus

    n_samples = sample_matrix.shape[0]
    n_timepoints = len(timepoints)

    x = np.empty(shape=(n_samples * (n_timepoints - 1), 2), dtype=np.float64)
    y = np.empty(shape=(n_samples * (n_timepoints - 1), 2), dtype=np.float64)
    # The Input of our NN is
    x[:, 0] = x_plus.flatten(order="C")  # X_t sample values #flatten: copy array, C for row
    x[:, 1] = np.tile(timepoints[:-1], n_samples)  # t-values
    # tile: Construct an array by repeating time_sequence the number of times given by n_samples

    if time_homogenous:
        x = x[:, 0]  # If we don't need the t values, we just forget about them

    # The increments of X-Values and t-values are stored in
    y[:, 0] = d.flatten(order="C")  # X_j - X_{j-1}
    y[:, 1] = np.tile(timepoints[1:] - timepoints[:-1], n_samples)  # delta_t

    return x, y

def one_step_euler(a_fun, b_fun, tspan, bms, y0, assert_positive=False):
    assert y0.shape[0] == bms.shape[0]

    res = np.zeros(shape=bms.shape)
    res[:, 0] = y0

    for i in range(1, bms.shape[1]):
        delta_t = tspan[i] - tspan[i-1]
        diff_add = delta_t * a_fun(res[:, i-1], tspan[i-1])
        if assert_positive:
            diff_add[res[:, i-1] < 0.0] = 0.0
        res[:, i] = res[:, i-1] + diff_add + b_fun(res[:, i-1], tspan[i-1]) * (bms[:, i] - bms[:, i-1])

    return res

def get_naive_dense_model(unit_nums, delta_t, loss="naive", time_homogenous=True, loss_kwargs=None, **compile_kwargs):
    if loss_kwargs is None:
        loss_kwargs = {}

    in_shape = (1,) if time_homogenous else (2,)

    input_layer = tf.keras.layers.Input(shape=in_shape)
    running_layer = input_layer
    for k, n_unit in enumerate(unit_nums):
        running_layer = tf.keras.layers.Dense(n_unit, activation="swish")(running_layer)

    mean = tf.keras.layers.Dense(1)(running_layer)
    variance = tf.keras.layers.Dense(1, activation="softplus")(running_layer)

    output = tf.keras.layers.concatenate([mean, variance])

    naive_dense_model = tf.keras.models.Model(input_layer, output)

    return naive_dense_model


def pretrain_model_for_init(model, tspan, space_grid=np.linspace(-2.0, 2.0, 1001), n_steps=100, time_homogenous=False, learning_rate=1e-3):
    grid = np.stack(len(tspan) * [space_grid], axis=1)
    grid_x, grid_y = sample_matrix_to_det_NN_input(grid, tspan, time_homogenous=time_homogenous)
    # Now, grid_x[:, 0] are x values

    grid_x = tf.convert_to_tensor(grid_x, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if time_homogenous:
        x_vals = grid_x
    else:
        x_vals = grid_x[:, 0]

    @tf.function
    def make_optimiser_step():
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(grid_x)
            output = model(grid_x, training=True)

            loss_value = tf.math.reduce_mean(tf.math.square(tf.math.add(
                output[:, 0],
                x_vals
            ))) + tf.math.reduce_mean(tf.math.square(output[:, 1] - 1.0))

        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        opt.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value

    for i in range(n_steps):
        _ = make_optimiser_step()