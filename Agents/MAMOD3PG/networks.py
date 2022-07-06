"""Shared helpers for different experiment flavours."""

from typing import Sequence, Optional
from acme import types
from acme.tf import networks
from Agents.MAMOD3PG.multiplexers import CriticMultiplexer
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt
from Agents.MAMOD3PG.dueling import DuellingMLP
from typing import Tuple
import dataclasses
from acme.tf import networks as network_utils
from acme.tf import utils

@dataclasses.dataclass
class MAMOD3PGNetworks:
    """Structure containing the networks for D3PG."""

    vehicle_policy_network: types.TensorTransformation
    vehicle_critic_network: types.TensorTransformation
    vehicle_observation_network: types.TensorTransformation

    edge_policy_network: types.TensorTransformation
    edge_critic_network: types.TensorTransformation
    edge_observation_network: types.TensorTransformation

    def __init__(
        self,
        vehicle_policy_network: types.TensorTransformation,
        vehicle_critic_network: types.TensorTransformation,
        vehicle_observation_network: types.TensorTransformation,

        edge_policy_network: types.TensorTransformation,
        edge_critic_network: types.TensorTransformation,
        edge_observation_network: types.TensorTransformation,
    ):
        # This method is implemented (rather than added by the dataclass decorator)
        # in order to allow observation network to be passed as an arbitrary tensor
        # transformation rather than as a snt Module.
        self.vehicle_policy_network = vehicle_policy_network
        self.vehicle_critic_network = vehicle_critic_network
        self.vehicle_observation_network = utils.to_sonnet_module(vehicle_observation_network)

        self.edge_policy_network = edge_policy_network
        self.edge_critic_network = edge_critic_network
        self.edge_observation_network = utils.to_sonnet_module(edge_observation_network)

    def init(
        self, 
        environment_spec,
    ):
        """Initialize the networks given an environment spec."""
        # Get observation and action specs.
        vehicle_observation_spec = environment_spec.vehicle_observations
        critic_vehicle_action_spec = environment_spec.critic_vehicle_actions
        edge_observation_spec = environment_spec.edge_observations
        critic_edge_action_spec = environment_spec.critic_edge_actions

        # Create variables for the observation net and, as a side-effect, get a
        # spec describing the embedding space.
        vehicle_emb_spec = utils.create_variables(self.vehicle_observation_network, [vehicle_observation_spec])
        edge_emb_spec = utils.create_variables(self.edge_observation_network, [edge_observation_spec])

        # Create variables for the policy and critic nets.
        _ = utils.create_variables(self.vehicle_policy_network, [vehicle_emb_spec])
        # TODO:TypeError: __call__() missing 1 required positional argument: 'action'
        _ = utils.create_variables(self.vehicle_critic_network, [vehicle_emb_spec, critic_vehicle_action_spec])

        _ = utils.create_variables(self.edge_policy_network, [edge_emb_spec])
        _ = utils.create_variables(self.edge_critic_network, [edge_emb_spec, critic_edge_action_spec])

    def make_policy(
        self,
        environment_spec,
        sigma: float = 0.0,
    ) -> Tuple[snt.Module, snt.Module]:
        """Create a single network which evaluates the policy."""
        # Stack the observation and policy networks.
        vehicle_stack = [
            self.vehicle_observation_network,
            self.vehicle_policy_network,
        ]

        edge_stack = [
            self.edge_observation_network,
            self.edge_policy_network,
        ]

        # If a stochastic/non-greedy policy is requested, add Gaussian noise on
        # top to enable a simple form of exploration.
        # TODO: Refactor this to remove it from the class.
        if sigma > 0.0:
            vehicle_stack += [
                network_utils.ClippedGaussian(sigma),
                network_utils.ClipToSpec(environment_spec.vehicle_actions),   # Clip to action spec.
            ]
            edge_stack += [
                network_utils.ClippedGaussian(sigma),
                network_utils.ClipToSpec(environment_spec.edge_actions),    # Clip to action spec.
            ]

        # Return a network which sequentially evaluates everything in the stack.
        return snt.Sequential(vehicle_stack), snt.Sequential(edge_stack)



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


def make_default_MAMOD3PGNetworks(
    vehicle_action_spec: Optional[None] = None,
    vehicle_policy_layer_sizes: Sequence[int] = (256, 256, 128),
    vehicle_critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vehicle_vmin: float = -150.,
    vehicle_vmax: float = 150.,
    vehicle_num_atoms: int = 51,
            
    edge_action_spec: Optional[None] = None,
    edge_policy_layer_sizes: Sequence[int] = (256, 256, 128),
    edge_critic_layer_sizes: Sequence[int] = (512, 512, 256),
    edge_vmin: float = -150.,
    edge_vmax: float = 150.,
    edge_num_atoms: int = 51,
    
    vehicle_number: Optional[int] = None,
    vehicle_action_number: Optional[int] = None,
    vehicle_observation_size: Optional[int] = None,
    edge_observation_size: Optional[int] = None,
    edge_action_number: Optional[int] = None,
):
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
        # The multiplexer concatenates the observations/actions/weights.
        CriticMultiplexer(),
        DuellingMLP(
            hidden_sizes=vehicle_critic_layer_sizes, 
            action_number=vehicle_action_number,
            random_action_size=10,
            observation_size=vehicle_observation_size,
            other_action_size=vehicle_action_number*(vehicle_number-1),
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
        # The multiplexer concatenates the observations/actions/weights.
        CriticMultiplexer(),
        DuellingMLP(
            hidden_sizes=edge_critic_layer_sizes,
            action_number=edge_action_number,
            random_action_size=10,
            observation_size=edge_observation_size,
            other_action_size=vehicle_action_number*vehicle_number,
        ),
        networks.DiscreteValuedHead(edge_vmin, edge_vmax, edge_num_atoms),
    ])

    return MAMOD3PGNetworks(
        vehicle_observation_network=vehicle_observation_network,
        vehicle_policy_network=vehicle_policy_network,
        vehicle_critic_network=vehicle_critic_network,
        edge_observation_network=edge_observation_network,
        edge_policy_network=edge_policy_network,
        edge_critic_network=edge_critic_network,
    )