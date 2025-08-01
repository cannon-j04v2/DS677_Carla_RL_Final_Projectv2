import os
import sys
import time
import random
import numpy as np
import logging
import pickle
import torch
import glob
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

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
from networks.off_policy.DQN_FINALTEST.agent import DQNAgent as DQNAgentFinalTest
from rewards import RewardFactory
from get_args import get_args

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
        
        # Set to synchronous mode with a fixed time step for stability
        settings = world.get_settings()
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        # print("Enabled synchronous mode with a fixed time step of 0.05s.")

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
    def __init__(self, client, world, town, checkpoint_frequency=None, reward_type='simple', reward_config=None, driver_view=False):
        self.client = client
        self.world = world
        self.town = town
        self.checkpoint_frequency = checkpoint_frequency
        self.driver_view = driver_view
        
        # Set up the world
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        
        # Get the map and spawn points
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        
        # Spawn the vehicle
        self.vehicle = None
        self.spawn_vehicle()
        
        # Set up sensors
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.imu_sensor = None
        self.lidar_sensor = None
        self.setup_sensors()
        
        # Set up driver view if enabled
        self.driver_camera = None
        self.display = None
        if self.driver_view:
            self.setup_driver_view()
        
        # Episode tracking
        self.episode_step = 0
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.num_collisions_this_episode = 0
        
        # Set up reward function
        self.reward_function = RewardFactory.create_reward(reward_type, reward_config)
        
    def spawn_vehicle(self):
        """Spawn a vehicle at a random spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'hero')
        
        spawn_points = self.map.get_spawn_points()
        # print(f"Found {len(spawn_points)} spawn points")  # Commented out
        
        # Try to spawn at different spawn points if one fails
        for attempt in range(5):
            spawn_point = random.choice(spawn_points)
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                # print(f"Vehicle spawned successfully at spawn point {attempt}")  # Commented out
                break
            except Exception as e:
                # print(f"Failed to spawn vehicle at attempt {attempt}: {e}")  # Commented out
                if attempt == 4:  # Last attempt
                    print(f"ERROR: Could not spawn vehicle after 5 attempts: {e}")
                    raise Exception("Could not spawn vehicle after 5 attempts")
        
        self.vehicle.set_simulate_physics(True)
        
        # Set initial location for distance tracking
        self.last_location = self.vehicle.get_location()
        # print(f"Vehicle location: {self.last_location}")  # Commented out
        
    def setup_sensors(self):
        """Set up all sensors for the vehicle."""
        self.setup_camera()
        self.setup_collision_sensor()
        self.setup_lane_invasion_sensor()
        self.setup_imu_sensor()
        self.setup_lidar_sensor()

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
            # print("Camera spawned successfully")  # Commented out
        except Exception as e:
            print(f"ERROR: Failed to spawn camera: {e}")
            self.camera = None
            return
        
        # Set up image queue
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        # print("Camera sensor listening for images")  # Commented out
        
    def setup_collision_sensor(self):
        """Set up a collision sensor to detect collisions."""
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_bp.set_attribute('role_name', 'collision_sensor')
        
        collision_transform = carla.Transform(carla.Location(x=0.0, z=1.0)) # Attach to the vehicle's top
        try:
            self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)
            # print("Collision sensor spawned successfully")  # Commented out
        except Exception as e:
            print(f"ERROR: Failed to spawn collision sensor: {e}")
            self.collision_sensor = None
            return
        
        # Set up collision callback
        self.collision_detected = False
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        # print("Collision sensor listening for collisions")  # Commented out

    def setup_lane_invasion_sensor(self):
        """Set up a lane invasion sensor."""
        blueprint_library = self.world.get_blueprint_library()
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_detected = False
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    def setup_imu_sensor(self):
        """Set up an IMU sensor."""
        blueprint_library = self.world.get_blueprint_library()
        imu_bp = blueprint_library.find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        self.imu_data = {'accelerometer': carla.Vector3D(), 'gyroscope': carla.Vector3D()}
        self.imu_sensor.listen(lambda data: self._on_imu_update(data))

    def setup_lidar_sensor(self):
        """Set up a LiDAR sensor for obstacle detection."""
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '112000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-30')
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        try:
            self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        except Exception as e:
            print(f"ERROR: Failed to spawn LiDAR sensor: {e}")
            self.lidar_sensor = None
            return

        self.lidar_queue = queue.Queue()
        self.lidar_sensor.listen(self.lidar_queue.put)

    def setup_driver_view(self):
        """Set up a driver view camera for visualization."""
        try:
            import pygame
            pygame.init()
            self.display = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Driver View")
            
            # Create driver camera blueprint
            blueprint_library = self.world.get_blueprint_library()
            driver_camera_bp = blueprint_library.find('sensor.camera.rgb')
            driver_camera_bp.set_attribute('image_size_x', '640')
            driver_camera_bp.set_attribute('image_size_y', '480')
            driver_camera_bp.set_attribute('fov', '90')
            
            # Attach camera to vehicle (driver position)
            driver_camera_transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))
            self.driver_camera = self.world.spawn_actor(driver_camera_bp, driver_camera_transform, attach_to=self.vehicle)
            
            # Set up image callback for driver view
            self.driver_image_data = {'array': None}
            self.driver_camera.listen(lambda image: self._driver_image_callback(image))
            
            print("[INFO] Driver view camera attached")
            
        except ImportError:
            print("[WARNING] Pygame not available. Driver view disabled.")
            self.driver_view = False
        except Exception as e:
            print(f"[ERROR] Failed to setup driver view: {e}")
            self.driver_view = False

    def _driver_image_callback(self, image):
        """Callback for driver view camera images."""
        try:
            import pygame
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            self.driver_image_data['array'] = array[:, :, :3]
            
            if self.display:
                surface = pygame.surfarray.make_surface(self.driver_image_data['array'].swapaxes(0, 1))
                self.display.blit(surface, (0, 0))
                pygame.display.flip()
                
        except Exception as e:
            print(f"[ERROR] Driver image callback failed: {e}")

    def _on_collision(self, event):
        """Callback for collision events."""
        self.collision_detected = True
        self.num_collisions_this_episode += 1
        try:
            actor_name = event.other_actor.name if hasattr(event.other_actor, 'name') else 'Unknown'
            print("\n###### COLLISION OCCURRED #######")
            print(f"Actor: {event.other_actor.type_id}, Name: {actor_name}")
            print("##################################\n")
        except Exception as e:
            print("\n###### COLLISION OCCURRED #######")
            print(f"Actor: {event.other_actor.type_id}, Error getting name: {e}")
            print("##################################\n")

    def _on_lane_invasion(self, event):
        """Callback for lane invasion events."""
        self.lane_invasion_detected = True

    def _on_imu_update(self, data):
        """Callback for IMU updates."""
        self.imu_data['accelerometer'] = data.accelerometer
        self.imu_data['gyroscope'] = data.gyroscope

    def _process_lidar(self, raw_lidar_data):
        """Process LiDAR point cloud into a fixed-size representation."""
        points = np.frombuffer(raw_lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Get x, y, z coordinates
        lidar_points = points[:, :3]
        
        # 8 sectors, 45 degrees each
        num_sectors = 8
        sectors = [[] for _ in range(num_sectors)]
        
        for point in lidar_points:
            x, y, z = point
            if x == 0 and y == 0:
                continue

            # Get angle and distance
            angle = np.degrees(np.arctan2(y, x))
            if angle < 0:
                angle += 360
            
            dist = np.sqrt(x**2 + y**2)

            # Assign to sector
            sector_index = int(angle // (360 / num_sectors))
            if 0 <= sector_index < num_sectors:
                sectors[sector_index].append(dist)

        # Get minimum distance in each sector
        min_dists = []
        for sector_dists in sectors:
            if sector_dists:
                min_dists.append(min(sector_dists))
            else:
                min_dists.append(50) # Max range
        
        return np.array(min_dists) / 50.0 # Normalize

    def reset(self):
        """Reset the environment for a new episode."""
        # Destroy current vehicle and sensors
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.destroy()
        if self.driver_camera:
            self.driver_camera.destroy()
            
        # Spawn new vehicle and sensors
        self.spawn_vehicle()
        self.setup_sensors()
        if self.driver_view:
            self.setup_driver_view()
        
        # Reset episode tracking
        self.episode_step = 0
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.num_collisions_this_episode = 0
        
        # Reset reward function
        self.reward_function.reset()
        
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
            # print("Warning: No camera image received, using zeros")  # Commented out
        
        # Get LiDAR data
        try:
            lidar_data = self.lidar_queue.get(timeout=1.0)
            lidar_features = self._process_lidar(lidar_data)
        except queue.Empty:
            lidar_features = np.zeros(8)
        
        # Get traffic light state
        traffic_light_state = [0, 0, 0, 0] # [None, Green, Yellow, Red]
        if self.vehicle.is_at_traffic_light():
            light = self.vehicle.get_traffic_light()
            if light is not None:
                if light.state == carla.TrafficLightState.Green:
                    traffic_light_state = [0, 1, 0, 0]
                elif light.state == carla.TrafficLightState.Yellow:
                    traffic_light_state = [0, 0, 1, 0]
                elif light.state == carla.TrafficLightState.Red:
                    traffic_light_state = [0, 0, 0, 1]
        else:
            traffic_light_state = [1, 0, 0, 0]

        # Get vehicle state for navigation info
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_rotation = vehicle_transform.rotation
        vehicle_velocity = self.vehicle.get_velocity()
        
        # Create navigation observation with additional IMU data
        nav_obs = np.array([
            vehicle_location.x / 1000.0,
            vehicle_location.y / 1000.0,
            vehicle_rotation.yaw / 360.0,
            vehicle_velocity.x / 20.0,
            vehicle_velocity.y / 20.0,
            self.imu_data['accelerometer'].x / 10.0,
            self.imu_data['accelerometer'].y / 10.0,
            self.imu_data['gyroscope'].z / 10.0
        ])
        
        # Combine all navigation and sensor features
        combined_nav = np.concatenate([nav_obs, lidar_features, np.array(traffic_light_state)])

        return [array, combined_nav]
        
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        # Parse action (steer, throttle, brake)
        steer, throttle, brake = action
        
        # Apply control
        control = carla.VehicleControl()
        control.steer = float(steer)
        # Ensure throttle and brake are in valid ranges [0, 1]
        control.throttle = float(max(0.0, throttle))
        control.brake = float(max(0.0, brake))
        
        self.vehicle.apply_control(control)
        
        # Step the simulation
        self.world.tick()
        
        # Get new observation
        observation = self.get_observation()
        
        # Calculate reward and done
        reward, done, info = self.calculate_reward_and_done()
        
        self.episode_step += 1
        
        # Remove noisy step debug prints in step()
        # if self.episode_step <= 2:
        #     print(f"Step {self.episode_step}: reward={reward:.3f}, speed={info.get('speed', 0):.2f}, done={done}")
        
        return observation, reward, done, info
        
    def calculate_reward_and_done(self):
        """Calculate reward and determine if episode is done using the reward function."""
        # Use the reward function to calculate reward and done
        reward, done, info = self.reward_function.calculate_reward(
            vehicle=self.vehicle,
            episode_step=self.episode_step,
            collision_detected=self.collision_detected,
            lane_invasion_detected=self.lane_invasion_detected,
            world=self.world  # Pass world reference for waypoint access
        )
        
        info['num_collisions'] = self.num_collisions_this_episode
        
        return reward, done, info
        
    def cleanup(self):
        """Clean up actors."""
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lidar_sensor:
            self.lidar_sensor.destroy()
        if self.driver_camera:
            self.driver_camera.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.display:
            try:
                import pygame
                pygame.quit()
            except:
                pass

from parameters import *


def print_summary_statistics(run_mode, timestep, total_collisions, total_speed_for_stats, total_distance_for_stats):
    """Prints a formatted summary of the run."""
    print(f"\n========== {run_mode.upper()} SUMMARY STATISTICS ==========")
    if timestep > 0:
        collision_rate_pct = (total_collisions / timestep) * 100
        avg_speed = total_speed_for_stats / timestep
        print(f"  - Collision Rate: {collision_rate_pct:.2f}% (collisions per 100 timesteps)")
        print(f"  - Average Speed: {avg_speed:.2f} m/s")
    else:
        print("  - No timesteps were run.")

    if total_collisions > 0:
        avg_dist_per_collision = total_distance_for_stats / total_collisions
        print(f"  - Average Distance per Collision: {avg_dist_per_collision:.2f} m")
    else:
        print("  - No collisions occurred.")
    print("===================================================\n")


def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = get_args()
    algo = args.algo
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init
    carla_host = args.carla_host
    carla_port = args.carla_port
    reward_type = args.reward_type
    driver_view = args.driver_view

    ALGO_MAP = {'ppo': PPOAgent, 'dqn': DQNAgent, 'dqn_finaltest': DQNAgentFinalTest}
    run_name = algo.upper()

    if train == True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    # Handle list rewards argument
    if args.list_rewards:
        print("Available reward functions:")
        available_rewards = RewardFactory.get_available_rewards()
        for reward_name in available_rewards:
            reward_info = RewardFactory.get_reward_info(reward_name)
            print(f"  {reward_name}: {reward_info['description']}")
            print(f"    Default config: {reward_info['default_config']}")
        sys.exit(0)

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
    total_collisions = 0
    total_distance_for_stats = 0
    total_speed_for_stats = 0
    losses = []
    best_avg_reward = -np.inf
    degrade_counter = 0

    # define variables after reading args
    learn_starts = args.learn_starts
    learn_freq = args.learn_freq

    # Set up real-time plotting
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Top plot for rewards
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'Training Progress - {algo.upper()} with {reward_type} reward\nTown: {town}, LR: {args.learning_rate}, Std: {action_std_init}, Seed: {args.seed}')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot for MSE loss
    ax2.set_xlabel('Total Timesteps')
    ax2.set_ylabel('MSE Loss')
    ax2.grid(True, alpha=0.3)
    
    # Data for plotting
    plot_timesteps = []
    plot_rewards = []
    plot_losses = []
    reward_line, = ax1.plot([], [], 'b-', linewidth=2, label='Cumulative Reward')
    loss_line, = ax2.plot([], [], 'r-', linewidth=2, label='MSE Loss')
    ax1.legend()
    ax2.legend()
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

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
        env = CarlaEnvironment(client, world, town, reward_type=reward_type, driver_view=driver_view)
    else:
        env = CarlaEnvironment(client, world, town, checkpoint_frequency=None, reward_type=reward_type, driver_view=driver_view)
    encode = EncodeState(LATENT_DIM)

    # Test environment if requested
    # Adjust test timesteps if user did not specify and provided total_timesteps
    if not train and (args.test_timesteps == TEST_TIMESTEPS):
        args.test_timesteps = total_timesteps

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
                agent = AgentClass(town, use_double_dqn=args.double_dqn,
                                  target_update_method=args.target_update_method,
                                  soft_update_tau=args.soft_update_tau)
                agent.load()  # Load latest checkpoint if available
            elif algo == 'dqn_finaltest':
                # DQN_FINALTEST checkpoint loading logic with custom parameters
                agent = AgentClass(town, 
                                   target_update_method=args.target_update_method,
                                   soft_update_tau=args.soft_update_tau,
                                   use_double_dqn=args.double_dqn)
                agent.load()  # Load latest checkpoint if available
        else:
            if train == False:
                if algo == 'ppo':
                    agent = AgentClass(town, action_std_init)
                elif algo == 'dqn':
                    agent = AgentClass(town, use_double_dqn=args.double_dqn,
                                      target_update_method=args.target_update_method,
                                      soft_update_tau=args.soft_update_tau)
                elif algo == 'dqn_finaltest':
                    agent = AgentClass(town, 
                                       target_update_method=args.target_update_method,
                                       soft_update_tau=args.soft_update_tau,
                                       use_double_dqn=args.double_dqn)
                agent.load()
                if args.random_test and (algo == 'dqn' or algo == 'dqn_finaltest'):
                    agent.epsilon = 1.0  # force random actions during evaluation
                if algo == 'ppo':
                    for params in agent.old_policy.actor.parameters():
                        params.requires_grad = False
            else:
                if algo == 'ppo':
                    agent = AgentClass(town, action_std_init)
                elif algo == 'dqn':
                    agent = AgentClass(town, use_double_dqn=args.double_dqn,
                                      target_update_method=args.target_update_method,
                                      soft_update_tau=args.soft_update_tau)
                elif algo == 'dqn_finaltest':
                    agent = AgentClass(town, 
                                       target_update_method=args.target_update_method,
                                       soft_update_tau=args.soft_update_tau,
                                       use_double_dqn=args.double_dqn)
        
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
        elif algo == 'dqn_finaltest':
            print(f"Initial Epsilon: {agent.epsilon}")
            print(f"Learning Rate: {agent.lr}")
            print(f"Buffer Size: {agent.memory.capacity}")
            print(f"Network Hidden Layers: {agent.num_hidden}")
            print(f"Hidden Layer Dimension: {agent.hidden_dim}")
            print(f"Target Update Method: {agent.target_update_method}")
            if agent.target_update_method == 'soft':
                print(f"Soft Update Tau: {agent.soft_update_tau}")
            else:
                print(f"Hard Update Frequency: {agent.target_update_freq}")
        print(f"{'='*60}\n")
        
        if train:
            #Training
            print(f"Starting training loop with {total_timesteps} total timesteps...")
            while timestep < total_timesteps:
                
                    print(f"Starting episode {episode + 1}... (timestep: {timestep}/{total_timesteps})")
                    observation = env.reset()
                    observation = encode.process(observation)

                    current_ep_reward = 0
                    t1 = datetime.now()

                    for t in range(args.episode_length):
                    
                        # Handle pygame events for driver view
                        if driver_view:
                            try:
                                import pygame
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        print("Driver view window closed. Continuing training...")
                                        driver_view = False
                                        env.driver_view = False
                                        if env.display:
                                            pygame.quit()
                                            env.display = None
                            except:
                                pass
                    
                        # select action with policy
                        action_result = agent.get_action(observation, train=True)
                        
                        # Handle different action formats
                        if algo == 'dqn' or algo == 'dqn_finaltest':
                            action, action_idx = action_result
                        else:
                            action = action_result

                        observation, reward, done, info = env.step(action)
                        total_speed_for_stats += info.get('speed', 0)
                        if observation is None:
                            break
                        observation = encode.process(observation)
                        
                        # Algorithm-specific memory handling
                        if algo == 'ppo':
                            agent.memory.rewards.append(reward)
                            agent.memory.dones.append(done)
                        elif algo == 'dqn' or algo == 'dqn_finaltest':
                            # Store experience in replay buffer
                            agent.memory.add(observation, action_idx, reward, observation, done)
                            # Perform learning step every few steps
                            if timestep >= learn_starts and timestep % learn_freq == 0:
                                loss = agent.learn()
                                if loss:
                                    losses.append(loss)
                        
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
                            t2 = datetime.now()
                            t3 = t2-t1
                            
                            episodic_length.append(abs(t3.total_seconds()))
                            break
                    
                    # Increment episode counter after the loop (unconditional)
                    episode += 1
                    
                    total_collisions += info.get('num_collisions', 0)
                    total_distance_for_stats += info.get('distance', 0)

                    deviation_from_center += info.get('deviation_from_center', 0)
                    distance_covered += info.get('distance', 0)
                    
                    scores.append(current_ep_reward)
                    
                    if checkpoint_load:
                        cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                    else:
                        cumulative_score = np.mean(scores)

                    # Early stopping check
                    if args.early_stop and episode >= args.early_stop_patience:
                        if cumulative_score > best_avg_reward + 1e-3:
                            best_avg_reward = cumulative_score
                            degrade_counter = 0
                        elif cumulative_score < best_avg_reward - args.early_stop_tolerance:
                            degrade_counter += 1
                            if degrade_counter >= args.early_stop_patience:
                                print("Early stopping triggered: performance degraded. Saving model and stopping.")
                                agent.save()
                                break

                    # Print episode analytics summary
                    print("\n========== EPISODE SUMMARY ==========")
                    print(f"Episode: {episode}")
                    print(f"Total Reward: {current_ep_reward:.2f}")
                    print(f"Average Reward: {cumulative_score:.2f}")
                    print(f"Distance Covered: {info.get('distance', 0):.2f} m")
                    print(f"Deviation from Center: {info.get('deviation_from_center', 0):.2f}")
                    print(f"Speed: {info.get('speed', 0):.2f} m/s")
                    print(f"Collision Detected: {info.get('collision_detected', False)}")
                    print("====================================\n")

                    # Only print every 5 episodes for analytics
                    if episode > 0 and episode % 5 == 0:
                        print(f"[Analytics] Last 5 episodes avg reward: {np.mean(scores[-min(len(scores),5):]):.2f}")
                        print(f"[Analytics] Last 5 episodes avg distance: {distance_covered/min(len(scores),5):.2f}")

                    # Update real-time plot
                    plot_timesteps.append(timestep)
                    plot_rewards.append(np.sum(scores))
                    
                    # Update the plot data
                    reward_line.set_data(plot_timesteps, plot_rewards)
                    
                    # Update loss plot if there is data
                    if len(losses) > 0:
                        loss_timesteps = np.linspace(0, timestep, len(losses))
                        plot_losses = losses
                        loss_line.set_data(loss_timesteps, plot_losses)
                        ax2.set_xlim(0, max(plot_timesteps))
                        ax2.set_ylim(0, max(plot_losses) * 1.1)

                    # Adjust plot limits
                    if len(plot_timesteps) > 1:
                        ax1.set_xlim(0, max(plot_timesteps))
                        ax1.set_ylim(min(plot_rewards) * 0.9 if min(plot_rewards) < 0 else 0, max(plot_rewards) * 1.1)
                    
                    # Redraw the plot
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    # Save plot every 10 episodes
                    if episode > 0 and episode % 10 == 0:
                        plot_filename = os.path.join(plots_dir, f'{algo.upper()}_{reward_type}_ep{episode}_ts{timestep}_lr{args.learning_rate}_std{action_std_init}_seed{args.seed}.png')
                        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        print(f"Plot saved: {plot_filename}")
                    
                    # Debug: Show action being taken
                    if episode > 0 and episode % 5 == 0:
                        # print(f"Debug - Action: {action}, Speed: {info.get('speed', 0):.2f}, Distance: {info.get('distance', 0):.2f}") # Commented out
                        if algo == 'ppo':
                            # print(f"PPO Debug - Action Std: {agent.action_std:.3f}, Memory Size: {len(agent.memory.rewards)}") # Commented out
                            pass # Removed debug print
                        elif algo == 'dqn' or algo == 'dqn_finaltest':
                            # print(f"DQN Debug - Epsilon: {agent.epsilon:.3f}, Buffer Size: {len(agent.memory)}") # Commented out
                            pass # Removed debug print
                    
                    if episode > 0 and episode % 10 == 0:
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
                        elif algo == 'dqn' or algo == 'dqn_finaltest':
                            agent.save()
                    
                    if episode > 0 and episode % 5 == 0:
                        # Calculate dynamic divisor for recent episodes
                        n_recent = min(len(scores), 5)

                        writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                        writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                        writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                        writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-n_recent:]), episode)
                        writer.add_scalar("Average Reward/(t)", np.mean(scores[-n_recent:]), timestep)
                        writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                        writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                        writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/n_recent, episode)
                        writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/n_recent, timestep)
                        writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/n_recent, episode)
                        writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/n_recent, timestep)
                        if len(losses) > 0:
                            writer.add_scalar("Loss/episode", np.mean(losses[-n_recent*10:]), episode) # Approx
                            writer.add_scalar("Loss/(t)", np.mean(losses[-n_recent*10:]), timestep)
                        
                        # DQN-specific logging
                        if algo == 'dqn' or algo == 'dqn_finaltest':
                            writer.add_scalar("DQN/Epsilon", agent.epsilon, episode)
                        # PPO-specific logging
                        elif algo == 'ppo':
                            writer.add_scalar("PPO/Action_Std", agent.action_std, episode)
                            writer.add_scalar("PPO/Memory_Size", len(agent.memory.rewards), episode)

                        episodic_length = list()
                        deviation_from_center = 0
                        distance_covered = 0

                    # Debug: Check loop condition
                    # print(f"Debug: After episode {episode}, timestep={timestep}, total_timesteps={total_timesteps}, continue={timestep < total_timesteps}") # Commented out

                    if episode > 0 and episode % 100 == 0:
                        
                        if algo == 'ppo':
                            agent.save()
                            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                            data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                            with open(chkpt_file, 'wb') as handle:
                                pickle.dump(data_obj, handle)
                        elif algo == 'dqn' or algo == 'dqn_finaltest':
                            agent.save()
                        
            print("Terminating the run.")
            
            # Save final plot
            final_plot_filename = os.path.join(plots_dir, f'{algo.upper()}_{reward_type}_FINAL_ep{episode}_ts{timestep}_lr{args.learning_rate}_std{action_std_init}_seed{args.seed}.png')
            fig.savefig(final_plot_filename, dpi=150, bbox_inches='tight')
            print(f"Final plot saved: {final_plot_filename}")

            print("\n========== TRAINING SUMMARY STATISTICS ==========")
            if timestep > 0:
                collision_rate = total_collisions / timestep
                avg_speed = total_speed_for_stats / timestep
                print(f"Collision Rate: {collision_rate:.4f} (collisions/timestep)")
                print(f"Average Speed: {avg_speed:.2f} m/s")
            else:
                print("No timesteps were run.")

            if total_collisions > 0:
                avg_dist_per_collision = total_distance_for_stats / total_collisions
                print(f"Average Distance per Collision: {avg_dist_per_collision:.2f} m")
            else:
                print("No collisions occurred.")
            if len(losses)>0:
                print(f"Average Loss: {np.mean(losses):.4f}")
            print("==============================================\n")
            
            # Close the plot
            plt.close(fig)
            
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
                    # Handle pygame events for driver view
                    if driver_view:
                        try:
                            import pygame
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    print("Driver view window closed. Continuing testing...")
                                    driver_view = False
                                    env.driver_view = False
                                    if env.display:
                                        pygame.quit()
                                        env.display = None
                        except:
                            pass
                            
                    # select action with policy
                    action_result = agent.get_action(observation, train=False)
                    
                    # Handle different action formats
                    if algo == 'dqn' or algo == 'dqn_finaltest':
                        action, _ = action_result
                    else:
                        action = action_result
                        
                    observation, reward, done, info = env.step(action)
                    total_speed_for_stats += info.get('speed', 0)
                    if observation is None:
                        break
                    observation = encode.process(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                # Increment episode counter after the loop (unconditional)
                episode += 1
                total_collisions += info.get('num_collisions', 0)
                total_distance_for_stats += info.get('distance', 0)
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

            # Save final plot (only during training)
            if train:
                final_plot_filename = os.path.join(plots_dir, f'{algo.upper()}_{reward_type}_FINAL_ep{episode}_ts{timestep}_lr{args.learning_rate}_std{action_std_init}_seed{args.seed}.png')
                fig.savefig(final_plot_filename, dpi=150, bbox_inches='tight')
                print(f"Final plot saved: {final_plot_filename}")
            
            # Close the plot
            plt.close(fig)

            # Print summary statistics
            run_mode = "training" if train else "testing"
            print_summary_statistics(run_mode, timestep, total_collisions, total_speed_for_stats, total_distance_for_stats)

            env.cleanup()
            sys.exit()

    finally:
        # Comment out blanket exit to expose real errors during debugging
        # sys.exit()
        pass


if __name__ == "__main__":
    import sys
    if '--test-baseline-imports' in sys.argv:
        from networks.common.base_agent import BaseAgent
        from networks.off_policy.DQN.agent import DQNAgent
        from networks.off_policy.DQN_FINALTEST.agent import DQNAgent as DQNAgentFinalTest
        from networks.on_policy.PPO.agent import PPOAgent
        print("BaseAgent, DQNAgent, DQNAgentFinalTest, PPOAgent import and instantiate test...")
        try:
            class DummyAgent(BaseAgent):
                def get_action(self, state, train): return 0
                def learn(self): pass
                def save(self, path): pass
                def load(self, path): pass
            dummy = DummyAgent()
            dqn = DQNAgent('TestTown')
            dqn_finaltest = DQNAgentFinalTest('TestTown')
            ppo = PPOAgent('TestTown')
            print("SUCCESS: All agents imported and instantiated.")
        except Exception as e:
            print(f"FAIL: {e}")
        sys.exit(0)
    runner()
