"""Shared helpers for different experiment flavours."""

from typing import Mapping, Sequence, Optional

from Environments import specs
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt


def make_policy_network(
        action_spec: specs.BoundedArray,
        policy_layer_sizes: Sequence[int] = (256, 256, 256),
    ) -> types.TensorTransformation:
        """Creates the networks used by the agent."""

        # Get total number of action dimensions from action spec.
        num_dimensions = np.prod(action_spec.shape, dtype=int)

        # Create the policy network.
        policy_network = snt.Sequential([
            networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
            networks.NearZeroInitializedLinear(num_dimensions),
            networks.TanhToSpec(action_spec),
        ])

        return policy_network


def make_default_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])

    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        networks.DiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


def make_default_D3PGNetworks(
    vehicle_action_spec: Optional[specs.BoundedArray] = None,
    vehicle_policy_layer_sizes: Sequence[int] = (256, 256, 256),
    vehicle_critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vehicle_vmin: float = -150.,
    vehicle_vmax: float = 150.,
    vehicle_num_atoms: int = 51,
    edge_action_spec: Optional[specs.BoundedArray] = None,
    edge_policy_layer_sizes: Sequence[int] = (256, 256, 256),
    edge_critic_layer_sizes: Sequence[int] = (512, 512, 256),
    edge_vmin: float = -150.,
    edge_vmax: float = 150.,
    edge_num_atoms: int = 51,
):
    from Agents.MAD3PG.agent import D3PGNetworks

    # Get total number of action dimensions from action spec.
    vehicle_num_dimensions = np.prod(vehicle_action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    vehicle_observation_network = tf2_utils.batch_concat

    # Create the policy network.
    vehicle_policy_network = snt.Sequential([
        networks.LayerNormMLP(vehicle_policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(vehicle_num_dimensions),
        networks.TanhToSpec(vehicle_action_spec),
    ])

    # Create the critic network.
    vehicle_critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(vehicle_critic_layer_sizes, activate_final=True),
        networks.DiscreteValuedHead(vehicle_vmin, vehicle_vmax, vehicle_num_atoms),
    ])

    # Get total number of action dimensions from action spec.
    edge_num_dimensions = np.prod(edge_action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    edge_observation_network = tf2_utils.batch_concat

    # Create the policy network.
    edge_policy_network = snt.Sequential([
        networks.LayerNormMLP(edge_policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(edge_num_dimensions),
        networks.TanhToSpec(edge_action_spec),
    ])

    # Create the critic network.
    edge_critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(edge_critic_layer_sizes, activate_final=True),
        networks.DiscreteValuedHead(edge_vmin, edge_vmax, edge_num_atoms),
    ])

    return D3PGNetworks(
        vehicle_observation_network=vehicle_observation_network,
        vehicle_policy_network=vehicle_policy_network,
        vehicle_critic_network=vehicle_critic_network,
        edge_observation_network=edge_observation_network,
        edge_policy_network=edge_policy_network,
        edge_critic_network=edge_critic_network,
    )