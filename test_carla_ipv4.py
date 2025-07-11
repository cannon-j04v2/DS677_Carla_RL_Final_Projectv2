#!/usr/bin/env python3

import carla
import time
import sys
import socket

def test_carla_connection():
    """Test CARLA connection with explicit IPv4 addressing"""
    
    # Test different connection methods
    connection_methods = [
        ('127.0.0.1', 2000),
        ('localhost', 2000),
        ('0.0.0.0', 2000)
    ]
    
    for host, port in connection_methods:
        print(f"\nTesting connection to {host}:{port}...")
        
        try:
            # Create client with longer timeout
            client = carla.Client(host, port)
            client.set_timeout(30.0)  # 30 second timeout
            
            # Get world
            world = client.get_world()
            print(f"✓ Successfully connected to CARLA via {host}:{port}!")
            print(f"  World: {world}")
            print(f"  Map: {world.get_map().name}")
            
            # Get available maps
            maps = client.get_available_maps()
            print(f"  Available maps: {len(maps)} maps")
            
            # Test spawning a vehicle
            print("\nTesting vehicle spawning...")
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            
            if vehicle_bp:
                spawn_points = world.get_map().get_spawn_points()
                if spawn_points:
                    spawn_point = spawn_points[0]
                    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                    print(f"✓ Successfully spawned Tesla Model 3 at {spawn_point.location}")
                    
                    # Clean up
                    vehicle.destroy()
                    print("✓ Vehicle destroyed successfully")
                else:
                    print("✗ No spawn points available")
            else:
                print("✗ Tesla Model 3 blueprint not found")
            
            return True, host, port
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            continue
    
    return False, None, None

def test_socket_connection():
    """Test basic socket connection to CARLA port"""
    print("\nTesting basic socket connection...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 2000))
        sock.close()
        
        if result == 0:
            print("✓ Socket connection successful")
            return True
        else:
            print(f"✗ Socket connection failed with error code: {result}")
            return False
    except Exception as e:
        print(f"✗ Socket test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== CARLA Connection Test ===")
    
    # Test basic socket connection first
    socket_ok = test_socket_connection()
    
    # Test CARLA API connection
    success, host, port = test_carla_connection()
    
    if success:
        print(f"\n🎉 CARLA connection test PASSED!")
        print(f"Working connection: {host}:{port}")
        print("You can now run your RL training script.")
    else:
        print("\n❌ CARLA connection test FAILED!")
        print("Troubleshooting tips:")
        print("1. Make sure CARLA is running (CarlaUE4.exe)")
        print("2. Check if firewall is blocking the connection")
        print("3. Try restarting CARLA")
        print("4. Check if multiple CARLA instances are running")
    
    sys.exit(0 if success else 1) 