import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
import glob
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import queue

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

from encoder_init import EncodeState
from networks.on_policy.PPO.agent import PPOAgent
from networks.off_policy.DQN.agent import DQNAgent

# Placeholder classes for compatibility - these will need proper implementation
class ClientConnection:
    def __init__(self, town, host='localhost', port=2000):
        self.town = town
        self.host = host
        self.port = port
        
    def setup(self):
        client = carla.Client(self.host, self.port)
        client.set_timeout(30.0)  # Increased timeout to 30 seconds
        
        print(f"Attempting to connect to CARLA at {self.host}:{self.port}")
        print("Waiting for CARLA server to respond...")
        
        # Get the world
        world = client.get_world()
        print("Successfully connected to CARLA server!")
        
        # Check if we need to load the town
        current_map = world.get_map()
        print(f"Current map: {current_map.name}")
        print(f"Requested town: {self.town}")
        
        # Get available maps
        available_maps = client.get_available_maps()
        print(f"Available maps: {available_maps}")
        
        # If the current map doesn't match the requested town, try to load it
        if current_map.name != self.town:
            # Check if the requested town is available
            town_available = self.town in available_maps
            if town_available:
                print(f"Loading town: {self.town}")
                try:
                    world = client.load_world(self.town)
                    print(f"Successfully loaded {self.town}")
                except Exception as e:
                    print(f"Failed to load {self.town}: {e}")
                    print(f"Using current map: {current_map.name}")
            else:
                print(f"Town {self.town} not available. Using current map: {current_map.name}")
        else:
            print(f"Town {self.town} is already loaded")
            
        return client, world

class CarlaEnvironment:
    def __init__(self, client, world, town, checkpoint_frequency=None):
        self.client = client
        self.world = world
        self.town = town
        self.checkpoint_frequency = checkpoint_frequency
        
        # Set up the world
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        
        # Get the map and spawn points
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        
        # Spawn the vehicle
        self.vehicle = None
        self.spawn_vehicle()
        
        # Set up sensors (camera for observation)
        self.camera = None
        self.collision_sensor = None
        self.setup_camera()
        self.setup_collision_sensor()
        
        # Episode tracking
        self.episode_step = 0
        self.total_distance = 0
        self.last_location = None
        self.collision_detected = False
        
    def spawn_vehicle(self):
        """Spawn a vehicle at a random spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'hero')
        
        spawn_points = self.map.get_spawn_points()
        print(f"Found {len(spawn_points)} spawn points")
        
        # Try to spawn at different spawn points if one fails
        for attempt in range(5):
            spawn_point = random.choice(spawn_points)
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                print(f"Vehicle spawned successfully at spawn point {attempt}")
                break
            except Exception as e:
                print(f"Failed to spawn vehicle at attempt {attempt}: {e}")
                if attempt == 4:  # Last attempt
                    raise Exception("Could not spawn vehicle after 5 attempts")
        
        self.vehicle.set_simulate_physics(True)
        
        # Set initial location for distance tracking
        self.last_location = self.vehicle.get_location()
        print(f"Vehicle location: {self.last_location}")
        
    def setup_camera(self):
        """Set up a camera sensor for observation."""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '160')
        camera_bp.set_attribute('image_size_y', '80')
        camera_bp.set_attribute('fov', '90')
        
        # Attach camera to vehicle
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        try:
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            print("Camera spawned successfully")
        except Exception as e:
            print(f"Failed to spawn camera: {e}")
            self.camera = None
            return
        
        # Set up image queue
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        print("Camera sensor listening for images")
        
    def setup_collision_sensor(self):
        """Set up a collision sensor to detect collisions."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_bp.set_attribute('role_name', 'collision_sensor')
        
        collision_transform = carla.Transform(carla.Location(x=0.0, z=1.0)) # Attach to the vehicle's top
        try:
            self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)
            print("Collision sensor spawned successfully")
        except Exception as e:
            print(f"Failed to spawn collision sensor: {e}")
            self.collision_sensor = None
            return
        
        # Set up collision callback
        self.collision_detected = False
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        print("Collision sensor listening for collisions")

    def _on_collision(self, event):
        """Callback for collision events."""
        self.collision_detected = True
        print(f"Collision detected! Actor: {event.other_actor.type_id}, Name: {event.other_actor.name}")

    def reset(self):
        """Reset the environment for a new episode."""
        # Destroy current vehicle and camera
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
            
        # Spawn new vehicle and camera
        self.spawn_vehicle()
        self.setup_camera()
        self.setup_collision_sensor()
        
        # Reset episode tracking
        self.episode_step = 0
        self.total_distance = 0
        self.last_location = self.vehicle.get_location()
        self.collision_detected = False
        
        # Get initial observation
        observation = self.get_observation()
        return observation
        
    def get_observation(self):
        """Get current observation from sensors."""
        # Get camera image
        try:
            image = self.image_queue.get(timeout=1.0)
            # Convert image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            array = array / 255.0  # Normalize to [0, 1]
        except queue.Empty:
            # If no image available, return zeros
            array = np.zeros((80, 160, 3))
            
        # Get vehicle state for navigation info
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_rotation = vehicle_transform.rotation
        vehicle_velocity = self.vehicle.get_velocity()
        
        # Create navigation observation (simplified)
        nav_obs = np.array([
            vehicle_location.x / 1000.0,  # Normalize coordinates
            vehicle_location.y / 1000.0,
            vehicle_rotation.yaw / 360.0,  # Normalize angle
            vehicle_velocity.x / 20.0,     # Normalize velocity
            vehicle_velocity.y / 20.0
        ])
        
        return [array, nav_obs]
        
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        # Parse action (steer, throttle, brake)
        steer, throttle, brake = action
        
        # Apply control
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        
        self.vehicle.apply_control(control)
        
        # Step the simulation
        self.world.tick()
        
        # Get new observation
        observation = self.get_observation()
        
        # Calculate reward and done
        reward, done, info = self.calculate_reward_and_done()
        
        self.episode_step += 1
        
        return observation, reward, done, info
        
    def calculate_reward_and_done(self):
        """Calculate reward and determine if episode is done."""
        # Get current vehicle state
        vehicle_location = self.vehicle.get_location()
        vehicle_velocity = self.vehicle.get_velocity()
        
        # Calculate distance traveled
        if self.last_location:
            distance = vehicle_location.distance(self.last_location)
            self.total_distance += distance
        self.last_location = vehicle_location
        
        # Calculate speed
        speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
        
        # Simple reward function
        reward = speed * 0.1  # Reward for moving forward
        
        # Check for collisions
        if self.collision_detected:
            reward -= 100  # Penalty for collision
            done = True
        else:
            done = False
            
        # Episode timeout
        if self.episode_step >= 1000:  # 1000 steps max
            done = True
            
        # Check if vehicle is stuck (not moving)
        if speed < 0.1 and self.episode_step > 100:
            reward -= 10
            done = True
            
        info = {
            'distance': self.total_distance,
            'speed': speed,
            'episode_step': self.episode_step
        }
        
        return reward, done, info
        
    def cleanup(self):
        """Clean up actors."""
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()

from parameters import *


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', '--exp-name', type=str, default='ppo', choices=['ppo', 'dqn'], help='RL algorithm to use (ppo or dqn)')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town10HD_Opt", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    parser.add_argument('--carla-host', type=str, default='localhost', help='CARLA server host (default: localhost)')
    parser.add_argument('--carla-port', type=int, default=2000, help='CARLA server port (default: 2000)')
    parser.add_argument('--test-env', action='store_true', help='Test environment setup only')
    args = parser.parse_args()
    
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    algo = args.algo
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init
    carla_host = args.carla_host
    carla_port = args.carla_port

    ALGO_MAP = {'ppo': PPOAgent, 'dqn': DQNAgent}
    run_name = algo.upper()

    if train == True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    client = None
    world = None
    try:
        client, world = ClientConnection(town, carla_host, carla_port).setup()
        logging.info("Connection has been setup successfully.")
    except Exception as e:
        logging.error(f"Connection has been refused by the server: {e}")
        print(f"ERROR: Could not connect to CARLA server at {carla_host}:{carla_port}")
        print("Please make sure CARLA is running on the specified host and port")
        print("You can start CARLA with: ./CarlaUE4.exe -opengl")
        print(f"Or specify a different host/port with: --carla-host HOST --carla-port PORT")
        sys.exit(1)
        
    if train:
        env = CarlaEnvironment(client, world,town)
    else:
        env = CarlaEnvironment(client, world,town, checkpoint_frequency=None)
    encode = EncodeState(LATENT_DIM)

    # Test environment if requested
    if args.test_env:
        print("Testing environment setup...")
        try:
            obs = env.reset()
            print(f"Environment reset successful. Observation shape: {[o.shape if hasattr(o, 'shape') else len(o) for o in obs]}")
            
            # Test a few steps
            for i in range(5):
                action = (0.0, 0.5, 0.0)  # Straight forward
                obs, reward, done, info = env.step(action)
                print(f"Step {i+1}: Reward={reward:.2f}, Done={done}, Speed={info.get('speed', 0):.2f}")
                if done:
                    break
                    
            env.cleanup()
            print("Environment test completed successfully!")
            sys.exit(0)
        except Exception as e:
            print(f"Environment test failed: {e}")
            env.cleanup()
            sys.exit(1)


    #========================================================================
    #                           ALGORITHM
    #========================================================================
    try:
        time.sleep(0.5)
        
        AgentClass = ALGO_MAP[algo]
        if checkpoint_load:
            # NOTE: Checkpoint logic will need to be specialized for DQN later
            if algo == 'ppo':
                chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
                chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                with open(chkpt_file, 'rb') as f:
                    data = pickle.load(f)
                    episode = data['episode']
                    timestep = data['timestep']
                    cumulative_score = data['cumulative_score']
                    action_std_init = data['action_std_init']
                agent = AgentClass(town, action_std_init)
                agent.load()
            elif algo == 'dqn':
                # DQN checkpoint loading logic
                agent = AgentClass(town)
                agent.load()  # Load latest checkpoint if available
        else:
            if train == False:
                agent = AgentClass(town, action_std_init) if algo == 'ppo' else AgentClass(town)
                agent.load()
                if algo == 'ppo':
                    for params in agent.old_policy.actor.parameters():
                        params.requires_grad = False
            else:
                agent = AgentClass(town, action_std_init) if algo == 'ppo' else AgentClass(town)
        
        # Print algorithm startup information
        print(f"\n{'='*60}")
        print(f"Starting {algo.upper()} Training")
        print(f"{'='*60}")
        print(f"Algorithm: {algo.upper()}")
        print(f"Town: {town}")
        print(f"Total Timesteps: {total_timesteps}")
        print(f"Episode Length: {args.episode_length}")
        if algo == 'ppo':
            print(f"Initial Action Std: {action_std_init}")
            print(f"Learning Rate: {args.learning_rate}")
        elif algo == 'dqn':
            print(f"Initial Epsilon: {agent.epsilon}")
            print(f"Learning Rate: {agent.lr}")
            print(f"Buffer Size: {agent.memory.capacity}")
        print(f"{'='*60}\n")
        
        if train:
            #Training
            while timestep < total_timesteps:
            
                observation = env.reset()
                observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                
                    # select action with policy
                    action_result = agent.get_action(observation, train=True)
                    
                    # Handle different action formats
                    if algo == 'dqn':
                        action, action_idx = action_result
                    else:
                        action = action_result

                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    
                    # Algorithm-specific memory handling
                    if algo == 'ppo':
                        agent.memory.rewards.append(reward)
                        agent.memory.dones.append(done)
                    elif algo == 'dqn':
                        # Store experience in replay buffer
                        agent.memory.add(observation, action_idx, reward, observation, done)
                        # Perform learning step every few steps
                        if timestep % 4 == 0:  # Learn every 4 steps
                            agent.learn()
                    
                    timestep +=1
                    current_ep_reward += reward
                    
                    # PPO-specific action std decay
                    if algo == 'ppo' and timestep % action_std_decay_freq == 0:
                        action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep == total_timesteps -1:
                        if algo == 'ppo':
                            agent.chkpt_save()
                        else:
                            agent.save()

                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                
                deviation_from_center += info.get('deviation_from_center', 0)
                distance_covered += info.get('distance', 0)
                
                scores.append(current_ep_reward)
                
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)

                # Log algorithm-specific metrics
                if algo == 'dqn':
                    epsilon_log = agent.epsilon
                    action_std_log = 0.0
                else:  # PPO
                    epsilon_log = 0.0
                    action_std_log = agent.action_std

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score), f', Epsilon: {epsilon_log:.3f}' if algo == 'dqn' else f', Action_Std: {action_std_log:.3f}')
                
                # Debug: Show action being taken
                if episode % 5 == 0:
                    print(f"Debug - Action: {action}, Speed: {info.get('speed', 0):.2f}, Distance: {info.get('distance', 0):.2f}")
                    if algo == 'ppo':
                        print(f"PPO Debug - Action Std: {agent.action_std:.3f}, Memory Size: {len(agent.memory.rewards)}")
                    elif algo == 'dqn':
                        print(f"DQN Debug - Epsilon: {agent.epsilon:.3f}, Buffer Size: {len(agent.memory)}")
                
                if episode % 10 == 0:
                    if algo == 'ppo':
                        agent.learn()
                        agent.chkpt_save()
                        chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                        if chkt_file_nums != 0:
                            chkt_file_nums -=1
                        chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                        data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                        with open(chkpt_file, 'wb') as handle:
                            pickle.dump(data_obj, handle)
                    elif algo == 'dqn':
                        agent.save()
                    
                
                if episode % 5 == 0:

                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/5, timestep)
                    
                    # DQN-specific logging
                    if algo == 'dqn':
                        writer.add_scalar("DQN/Epsilon", agent.epsilon, episode)
                    # PPO-specific logging
                    elif algo == 'ppo':
                        writer.add_scalar("PPO/Action_Std", agent.action_std, episode)
                        writer.add_scalar("PPO/Memory_Size", len(agent.memory.rewards), episode)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:
                    
                    if algo == 'ppo':
                        agent.save()
                        chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                        chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                        data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                        with open(chkpt_file, 'wb') as handle:
                            pickle.dump(data_obj, handle)
                    elif algo == 'dqn':
                        agent.save()
                        
            print("Terminating the run.")
            env.cleanup()
            sys.exit()
        else:
            #Testing
            while timestep < args.test_timesteps:
                observation = env.reset()
                observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    # select action with policy
                    action_result = agent.get_action(observation, train=False)
                    
                    # Handle different action formats
                    if algo == 'dqn':
                        action, _ = action_result
                    else:
                        action = action_result
                        
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info.get('deviation_from_center', 0)
                distance_covered += info.get('distance', 0)
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                
                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            env.cleanup()
            sys.exit()

    finally:
        sys.exit()


if __name__ == "__main__":
    import sys
    if '--test-baseline-imports' in sys.argv:
        from networks.common.base_agent import BaseAgent
        from networks.off_policy.DQN.agent import DQNAgent
        from networks.on_policy.PPO.agent import PPOAgent
        print("BaseAgent, DQNAgent, PPOAgent import and instantiate test...")
        try:
            class DummyAgent(BaseAgent):
                def get_action(self, state, train): return 0
                def learn(self): pass
                def save(self, path): pass
                def load(self, path): pass
            dummy = DummyAgent()
            dqn = DQNAgent('TestTown')
            ppo = PPOAgent('TestTown')
            print("SUCCESS: All agents imported and instantiated.")
        except Exception as e:
            print(f"FAIL: {e}")
        sys.exit(0)
    runner()
