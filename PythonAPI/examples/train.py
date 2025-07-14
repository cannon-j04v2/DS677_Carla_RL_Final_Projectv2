# train.py
import carla
import numpy as np
import cv2
import pygame
import time
import random
import torch
from dqn_agent import DQNAgent

IMG_HEIGHT, IMG_WIDTH = 84, 84
ACTIONS = [
    (0.5, 0.0, 0.0),   # go straight
    (0.5, -0.2, 0.0),  # turn left
    (0.5, 0.2, 0.0),   # turn right
    (0.0, 0.0, 0.5)    # brake
]

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    norm = resized.astype(np.float32) / 255.0
    return np.expand_dims(norm, axis=0)

def get_camera_blueprint(world):
    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    return camera_bp

def get_collision_sensor(world, vehicle, callback):
    bp = world.get_blueprint_library().find('sensor.other.collision')
    transform = carla.Transform()
    sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
    sensor.listen(callback)
    return sensor

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town10HD')
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("[INFO] Vehicle spawned")

    # Create external camera (spectator-like)
    camera_bp = get_camera_blueprint(world)
    camera_transform = carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print("[INFO] External camera attached")

    pygame.init()
    display = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("External View")

    image_data = {'array': None}
    def image_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        image_data['array'] = array[:, :, :3]
        surface = pygame.surfarray.make_surface(image_data['array'].swapaxes(0, 1))
        display.blit(surface, (0, 0))
        pygame.display.flip()

    camera.listen(image_callback)

    collided = {'flag': False}
    def collision_callback(event):
        collided['flag'] = True

    collision_sensor = get_collision_sensor(world, vehicle, collision_callback)
    print("[INFO] Collision sensor attached")

    agent = DQNAgent(num_actions=len(ACTIONS))
    episodes = 100
    max_steps = 500

    for episode in range(episodes):
        vehicle.set_transform(random.choice(world.get_map().get_spawn_points()))
        collided['flag'] = False
        total_reward = 0
        done = False

        while image_data['array'] is None:
            world.tick()
            time.sleep(0.1)

        state = preprocess_image(image_data['array'])

        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action_idx = agent.act(state)
            throttle, steer, brake = ACTIONS[action_idx]
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            world.tick()
            time.sleep(0.05)

            if image_data['array'] is None:
                continue

            next_state = preprocess_image(image_data['array'])

            speed = vehicle.get_velocity()
            speed_magnitude = np.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

            reward = speed_magnitude * 0.1
            if collided['flag']:
                reward -= 100
                done = True

            agent.remember(state, action_idx, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_target_network()
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    camera.stop()
    collision_sensor.stop()
    vehicle.destroy()
    pygame.quit()
    print("Training complete.")

if __name__ == '__main__':
    main()
