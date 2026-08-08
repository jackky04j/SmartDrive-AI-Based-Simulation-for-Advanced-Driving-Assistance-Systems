import carla
import time

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Load your two custom cones
    orange_cone_bp = bp_lib.find("static.prop.orange_cone")
    blue_cone_bp   = bp_lib.find("static.prop.blue_cone")

    spawn_points = world.get_map().get_spawn_points()
    base_point = spawn_points[0]   # start from first spawn point
    base_loc = base_point.location

    left_cones, right_cones = [], []

    # Spawn a row of cones
    for i in range(10):
        # Distance between cones (forward direction)
        x_offset = i * 3.0  

        # Left cone (orange)
        left_loc = carla.Location(x=base_loc.x + x_offset,
                                  y=base_loc.y - 2.0,
                                  z=base_loc.z)
        left_tf = carla.Transform(left_loc)
        left_cones.append(world.spawn_actor(orange_cone_bp, left_tf))

        # Right cone (blue)
        right_loc = carla.Location(x=base_loc.x + x_offset,
                                   y=base_loc.y + 2.0,
                                   z=base_loc.z)
        right_tf = carla.Transform(right_loc)
        right_cones.append(world.spawn_actor(blue_cone_bp, right_tf))

    print("Spawned orange cones on left and blue cones on right.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Cleaning up...")
    finally:
        for c in left_cones + right_cones:
            c.destroy()

if __name__ == "__main__":
    main()
