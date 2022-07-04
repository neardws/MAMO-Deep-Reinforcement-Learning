"""Shared helpers for different experiment flavours."""

import imp
from typing import Sequence, Optional
from acme import types
from acme.tf import networks
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt
from Agents.MAMOD3PG.dueling import DuellingMLP


def make_policy_network(
        action_spec,
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


def make_default_D3PGNetworks(
    vehicle_action_spec: Optional[None] = None,
    vehicle_policy_layer_sizes: Sequence[int] = (256, 256, 128),
    vehicle_critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vehicle_vmin: float = -150.,
    vehicle_vmax: float = 150.,
    vehicle_num_atoms: int = 51,
    vehicle_action_number: int = 31,
            
    edge_action_spec: Optional[None] = None,
    edge_policy_layer_sizes: Sequence[int] = (256, 256, 128),
    edge_critic_layer_sizes: Sequence[int] = (512, 512, 256),
    edge_vmin: float = -150.,
    edge_vmax: float = 150.,
    edge_num_atoms: int = 51,
    edge_action_number: int = 10,
    
    batch_size: int = 256,
):
    from Agents.MAMOD3PG.agent import D3PGNetworks

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
        DuellingMLP(
            hidden_sizes=vehicle_critic_layer_sizes, 
            action_number=vehicle_action_number,
            batch_size=batch_size,
        ),
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
        DuellingMLP(
            hidden_sizes=edge_critic_layer_sizes,
            action_number=edge_action_number,
            batch_size=batch_size,
        ),
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