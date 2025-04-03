from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from gymnasium import spaces   
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import StraightLane, CircularLane
from highway_env.vehicle.controller import ControlledVehicle


Observation = np.ndarray

class V2VVehicle(Vehicle):
    """Vehicle with basic V2V communication capabilities."""
    
    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.network = {}  # Dictionary to store received messages

    def send_message(self, message, receivers):
        """Broadcast message to other vehicles."""
        for vehicle in receivers:
            if isinstance(vehicle, V2VVehicle):
                vehicle.receive_message(message)

    def receive_message(self, message):
        """Receive and store messages."""
        self.network[message["sender"]] = message["data"]
    
    def get_network_info(self):
        """Return the latest received messages."""
        return self.network


class VanetHighEnv(AbstractEnv):
    #@classmethod
    #def default_config(cls) -> dict:
    #    config = super().default_config()
    #    config.update(
    #        {
    #            "observation": {"type": "Kinematics"},
    #            "action": {"type": "DiscreteMetaAction"},
    #            "duration": 40,  # Simulation duration in steps
    #        }
    #    )
    #    return config

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

#    def define_spaces(self):
#        """Define action and observation spaces."""
#        super().define_spaces()
        
#    def define_spaces(self):
#        """Define action and observation spaces."""
#        super().define_spaces()  # Call parent class method if needed

#        # Define action space (Example: Discrete or Continuous)
#        self.action_space = spaces.Discrete(5)  # Example: 5 discrete actions (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)
#
#        # Define observation space (Example: Vehicle position, speed, lane index, etc.)
#        self.observation_space = spaces.Box(
#            low=np.array([0, 0, -np.inf]),  # Example lower bound values
#            high=np.array([1, 1, np.inf]),  # Example upper bound values
#            dtype=np.float32
#        )


    def _create_road(self) -> None:
        #"""Create a road composed of straight and curved lanes."""
        net = RoadNetwork()

        #Straight section
        net.add_lane("A", "B", StraightLane([0, 0], [100, 0], width=4))
        net.add_lane("A", "B", StraightLane([0, 4], [100, 4], width=4))  # Second lane

        # Curved section (for an intersection)

        net.add_lane("B", "C", CircularLane([100, 50], radius=50, start_phase=-np.pi/2, end_phase=0))

         #Highway merge
        net.add_lane("D", "B", StraightLane([50, -10], [100, 0], width=4))

        #self.road = net
        
        
        self.road = Road(
            network=net,            
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some vehicles with V2V communication capabilities."""
        # Controlled vehicle (ego)
        ego_vehicle = V2VVehicle(self.road, position=[10, 0], speed=20)
        self.vehicle = ego_vehicle
        self.road.vehicles.append(ego_vehicle)

        # Other vehicles
        for i in range(5):  # Add 5 V2V-enabled vehicles
            other_vehicle = V2VVehicle(self.road, position=[i * 20, 4], speed=18)
            self.road.vehicles.append(other_vehicle)
        
        # Enable V2V communication
        for vehicle in self.road.vehicles:
            vehicle.send_message(
                {"sender": vehicle, "data": {"position": vehicle.position, "speed": vehicle.speed}}, 
                self.road.vehicles
            )

    def _reset(self) -> None:
        """Reset the environment's state (roads and vehicles)."""
        self._create_road()  # Create the road network
        self._create_vehicles()  # Create vehicles

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self._reset()  # Call the custom reset
        return self.observation_type.observe(), {}

    def step(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Log V2V messages for debugging purposes
        for vehicle in self.road.vehicles:
            if isinstance(vehicle, V2VVehicle):
                print(f"Vehicle at {vehicle.position} received: {vehicle.get_network_info()}")

        return obs, reward, terminated, truncated, info

    #def _reward(self, action) -> float:
    #    """Calculate reward based on the vehicle's status and behavior."""
    #    reward = 0
    #    for vehicle in self.road.vehicles:
    #        if isinstance(vehicle, V2VVehicle):
    #            # Example: Reward vehicles that are receiving useful information
    #            if vehicle.get_network_info():
    #                reward += 0.1  # Arbitrary reward for receiving a message
    #            
    #            # Reward for speed (just as in HighwayEnv)
    #            reward += np.clip(vehicle.speed / 30, 0, 1)
    #    
    #    return reward
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }


    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]