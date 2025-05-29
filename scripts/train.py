from deepreg.callback import build_checkpoint_callback
from deepreg.registry import REGISTRY
from deepreg.util import build_dataset
import deepreg.model.optimizer as opt
import deepreg.train as train
import tensorflow as tf
from tensorflow.python import debug as tf_debug


def set_GPU(device):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[device], 'GPU')
        except RuntimeError as e:
            print(e)


def fit_deepreg(
        config_path: str,
        log_dir: str,
        experiment_name: str,
        max_epochs: int,
        initial_epoch: int,
        ckpt_path: str = ""):
    """ Fit deepreg model.

    Args:
        config_path (str): Path to network configuration file.
        log_dir (str): Directory where the results will be saved.
        experiment_name (str): DeepReg automatically creates a folder titled 
            'experiment_name' under 'log_dir'.
        max_epochs (int): Maximum number of epochs for training.
        initial_epoch (int): Epoch where training starts.
        ckpt_path (str, optional): Path to the latest checkpoint, if any.
            Defaults to an empty string.
    """

    config, log_dir, ckpt_path = train.build_config(
        config_path=config_path,
        log_dir=log_dir,
        exp_name=experiment_name,
        ckpt_path=ckpt_path,
        max_epochs=max_epochs,
    )
    data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        split="train",
        training=True,
        repeat=True,
    )
    data_loader_val, dataset_val, steps_per_epoch_val = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        split="valid",
        training=False,
        repeat=True,
    )
    model: tf.keras.Model = REGISTRY.build_model(
        config=dict(
            name=config["train"]["method"],
            moving_image_size=data_loader_train.moving_image_shape,
            fixed_image_size=data_loader_train.fixed_image_shape,
            moving_label_size=tuple(config["dataset"]["moving_label_shape"]),
            fixed_label_size=tuple(config["dataset"]["fixed_label_shape"]),
            index_size=data_loader_train.num_indices,
            labeled=config["dataset"]["train"]["labeled"],
            batch_size=config["train"]["preprocess"]["batch_size"],
            config=config["train"],
        )
    )
    # tf.config.run_functions_eagerly(True)
    optimizer = opt.build_optimizer(optimizer_config=config["train"]["optimizer"])
    model.compile(optimizer=optimizer)
    # model.compile(optimizer=optimizer, run_eagerly=True) # BRIAN
    model.plot_model(output_dir=log_dir)
    # model.compile()
    # model.run_eagerly = True

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=config["train"]["save_period"],
        update_freq=config["train"].get("update_freq", "epoch"),
    )
    ckpt_callback, initial_epoch = build_checkpoint_callback(
        model=model,
        dataset=dataset_train,
        log_dir=log_dir,
        save_period=config["train"]["save_period"],
        ckpt_path=ckpt_path,
    )
    callbacks = [tensorboard_callback, ckpt_callback]

    # BRIAN
    # with tf_debug.LocalCLIDebugWrapperSession(tf.compat.v1.Session()) as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    history = model.fit(
        x=dataset_train,
        steps_per_epoch=steps_per_epoch_train,
        initial_epoch=initial_epoch,
        epochs=config["train"]["epochs"],
        validation_data=dataset_val,
        validation_steps=steps_per_epoch_val,
        callbacks=callbacks,
    )

