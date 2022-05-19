from typing import Optional, Sequence, Union
import sonnet as snt
import tensorflow as tf
import functools


Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]


def ensure_accelerator(accelerator: str) -> str:
    """Checks for the existence of the expected accelerator type.
    Args:
        accelerator: 'CPU', 'GPU' or 'TPU'.
    Returns:
        The validated `accelerator` argument.
    Raises:
        RuntimeError: Thrown if the expected accelerator isn't found.
    """
    devices = tf.config.get_visible_devices(device_type=accelerator)

    if devices:
        return accelerator
    else:
        error_messages = [f'Couldn\'t find any {accelerator} devices.',
                        'tf.config.get_visible_devices() returned:']
        error_messages.extend([str(d) for d in devices])
        raise RuntimeError('\n'.join(error_messages))


def get_first_available_accelerator_type(
    wishlist: Sequence[str] = ('TPU', 'GPU', 'CPU')) -> str:
    """Returns the first available accelerator type listed in a wishlist.
    Args:
        wishlist: A sequence of elements from {'CPU', 'GPU', 'TPU'}, listed in
        order of descending preference.
    Returns:
        The first available accelerator type from `wishlist`.
    Raises:
        RuntimeError: Thrown if no accelerators from the `wishlist` are found.
    """
    get_visible_devices = tf.config.get_visible_devices

    for wishlist_device in wishlist:
        devices = get_visible_devices(device_type=wishlist_device)
        if devices:
            return wishlist_device

    available = ', '.join(
        sorted(frozenset([d.type for d in get_visible_devices()])))
    raise RuntimeError(
        'Couldn\'t find any devices from {wishlist}.' +
        f'Only the following types are available: {available}.')

# Only instantiate one replicator per (process, accelerator type), in case
# a replicator stores state that needs to be carried between its method calls.
@functools.lru_cache()
def get_replicator(accelerator: Optional[str]) -> Replicator:
    """Returns a replicator instance appropriate for the given accelerator.
    This caches the instance using functools.cache, so that only one replicator
    is instantiated per process and argument value.
    Args:
        accelerator: None, 'TPU', 'GPU', or 'CPU'. If None, the first available
        accelerator type will be chosen from ('TPU', 'GPU', 'CPU').
    Returns:
        A replicator, for replciating weights, datasets, and updates across
        one or more accelerators.
    """
    if accelerator:
        accelerator = ensure_accelerator(accelerator)
    else:
        accelerator = get_first_available_accelerator_type()

    if accelerator == 'TPU':
        tf.tpu.experimental.initialize_tpu_system()
        return snt.distribute.TpuReplicator()
    else:
        return snt.distribute.Replicator()


def average_gradients_across_replicas(replica_context, gradients):
    """Computes the average gradient across replicas.
    This computes the gradient locally on this device, then copies over the
    gradients computed on the other replicas, and takes the average across
    replicas.
    This is faster than copying the gradients from TPU to CPU, and averaging
    them on the CPU (which is what we do for the losses/fetches).
    Args:
        replica_context: the return value of `tf.distribute.get_replica_context()`.
        gradients: The output of tape.gradients(loss, variables)
    Returns:
        A list of (d_loss/d_varabiable)s.
    """

    # We must remove any Nones from gradients before passing them to all_reduce.
    # Nones occur when you call tape.gradient(loss, variables) with some
    # variables that don't affect the loss.
    # See: https://github.com/tensorflow/tensorflow/issues/783
    # print("*" * 32)
    # print("gradients")
    # print(gradients)
    gradients_without_nones = [g for g in gradients if g is not None]
    original_indices = [i for i, g in enumerate(gradients) if g is not None]
    # print("*" * 32)
    # print("gradients_without_nones")
    # print(gradients_without_nones)
    results_without_nones = replica_context.all_reduce('mean',
                                                        gradients_without_nones)
    results = [None] * len(gradients)
    for ii, result in zip(original_indices, results_without_nones):
        results[ii] = result

    return results 


if __name__ == '__main__':
    print(get_first_available_accelerator_type())