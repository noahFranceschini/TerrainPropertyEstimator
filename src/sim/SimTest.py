import open3d.visualization
from TerrainEstimatorSim import TerrainEstimatorSim
import time
import numpy as np
from klampt import vis
from klampt.math import se3, so3
import klampt
from collections import deque
from contextlib import nullcontext
from threading import Lock
from klampt.vis.glinterface import GLPluginInterface
import open3d
class EventLoggerGLPlugin(GLPluginInterface):
    def __init__(self, event_log: deque, lock: Lock=None):
        super().__init__()
        self.event_log = event_log
        self.lock = lock if lock is not None else nullcontext()

    def keyboardfunc(self, c, x, y):
        with self.lock:
            self.event_log.append(("keyboard", c, x, y))
        return True
    
    def keyboardupfunc(self, c, x, y):
        with self.lock:
            self.event_log.append(("keyboardup", c, x, y))
        return True
def clamp(val, maxi, mini):
    r"""
    Returns the value clamped between the max and min
    """
    return min(max(val,mini),maxi)   
    # def mousefunc(self, button, state, x, y):
    #     with self.lock:
    #         self.event_log.append(("mouse", button, state, x, y))
    #     return True

# class ObjectMover(vis.GLPluginInterface):
#     def __init__(self, obj: klampt.Geometry3D, step_size=0.01):
#         super().__init__()
#         self.obj = obj
#         self.step_size = step_size

#     def keyboardfunc(self, key, x=None, y=None):
#         key = key.lower()  # Normalize input
#         print("Echo: ", key)
#         rotation,translation = self.obj.getCurrentTransform()
#         if key == 'q':  # Move up
#             translation[2] += self.step_size
#         elif key == 'e':  # Move down
#             translation[2] -= self.step_size
#         elif key == 'a':  # Move left
#             translation[0] -= self.step_size
#         elif key == 'd':  # Move right
#             translation[0] += self.step_size
#         elif key == 'w':  # Move forward
#             translation[1] += self.step_size
#         elif key == 's':  # Move backward
#             translation[1] -= self.step_size
#         elif key == "up":
#             rotation = so3.mul(rotation,so3.from_moment([0,-np.pi/6,0]))
#         elif key == "down":
#             rotation = so3.mul(rotation,so3.from_moment([0,np.pi/6,0]))
#         elif key == "left":
#             rotation = so3.mul(rotation,so3.from_moment([-np.pi/6,0,0]))
#         elif key == "right":
#             rotation = so3.mul(rotation,so3.from_moment([np.pi/6,0,0]))
#         elif key == "[":
#             rotation = so3.mul(rotation,so3.from_moment([0,0,-np.pi/6]))
#         elif key == "]":
#             rotation = so3.mul(rotation,so3.from_moment([0,0,np.pi/6]))
#         # Update object transform
#         self.obj.setCurrentTransform(rotation, translation)
#         return True
def compute_arrow_transform(arrow_start: list, arrow_end:list):
    t = arrow_start
    R = so3.align([0,0,1],arrow_end - arrow_start)
    return (R,t)
    
def move_body(key,obj,step_size=0.01):
        key = key.lower()  # Normalize input
        rotation,translation = obj.getTransform()
        if key == 'q':  # Move up
            translation[2] += step_size
        elif key == 'e':  # Move down
            translation[2] -= step_size
        elif key == 'a':  # Move left
            translation[0] -= step_size
        elif key == 'd':  # Move right
            translation[0] += step_size
        elif key == 'w':  # Move forward
            translation[1] += step_size
        elif key == 's':  # Move backward
            translation[1] -= step_size
        elif key == "up":
            rotation = so3.mul(rotation,so3.from_moment([0,-np.pi/12,0]))
        elif key == "down":
            rotation = so3.mul(rotation,so3.from_moment([0,np.pi/12,0]))
        elif key == "left":
            rotation = so3.mul(rotation,so3.from_moment([-np.pi/12,0,0]))
        elif key == "right":
            rotation = so3.mul(rotation,so3.from_moment([np.pi/12,0,0]))
        elif key == "[":
            rotation = so3.mul(rotation,so3.from_moment([0,0,-np.pi/12]))
        elif key == "]":
            rotation = so3.mul(rotation,so3.from_moment([0,0,np.pi/12]))
        # Update object transform
        return (rotation,translation)
if __name__ == "__main__":
    arrow = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.002, cylinder_height=0.01, cone_height=0.002, resolution=20, cylinder_split=4, cone_split=1)
    arrow.compute_vertex_normals()
    open3d.io.write_triangle_mesh("C:/Users/Noah/Desktop/2025Research/TerrainPropertyEstimator/src/meshes/Geometries/arrow.stl",arrow)
    arrow = open3d.io.read_triangle_mesh("C:/Users/Noah/Desktop/2025Research/TerrainPropertyEstimator/src/meshes/Geometries/arrow.stl")

    #load arrow mesh
    arrow = klampt.Geometry3D()
    arrow.loadFile("C:/Users/Noah/Desktop/2025Research/TerrainPropertyEstimator/src/meshes/Geometries/arrow.stl")
    arrow.convert("TriangleMesh")

    # scoop_mesh_fn = "C:/Users/Noah/Desktop/2025Research/TerrainPropertyEstimator/src/meshes/subdivided_mesh.STL"
    scoop_mesh_fn = "C:/Users/Noah/Desktop/2025Research/TerrainPropertyEstimator/src/meshes/scoop_no_probe_assem.STL"
    default_scoop_pos = (so3.from_moment([0,np.pi/2,0]),[0,0,0.1])
    # Used to test out the simulator
    heightmap_raw = np.load('C:/Users/Noah/Desktop/2025Research/TerrainPropertyEstimator/src/HeightMapEstimation/data/height_maps/gravel_height_test_0.npy')
    #flip for correct klampt view
    heightmap_raw = np.flip(heightmap_raw, axis=1)
    #TODO: create klampt simulation showing forces
    zetas = np.ones(heightmap_raw.shape) * 0.12*1e6
    terrain_properties = {}
    terrain_properties["zeta"] = zetas
    terrain_properties["grid_resolution"] = 0.005
    terrain_properties["grid_minimum"] = (-1,-1)

    height = klampt.Heightmap()
    height.setHeights(heightmap_raw)
    height.setSize(2,2)
    geo = klampt.Geometry3D()
    geo.setHeightmap(height)
    
    geo_occupancy = klampt.Geometry3D()
    geo_occupancy.loadFile(scoop_mesh_fn)
    geo_occupancy = geo_occupancy.convert("OccupancyGrid")
    scoop = klampt.Geometry3D()
    scoop.loadFile(scoop_mesh_fn)
  

    world_model = klampt.WorldModel()
    # Setup TerrainEstimatorSim
    terrain_sim = TerrainEstimatorSim(world_model)
    terrain_sim.add_geometry(geo, "height")
    terrain_sim.add_geometry(scoop, "scoop")
    # terrain_sim.add_geometry(geo_occupancy,"occupied")
    terrain_sim.set_object_config("scoop", default_scoop_pos)
    # terrain_sim.set_object_config("occupied", default_scoop_pos)
    scoop_model = world_model.rigidObject("scoop")
    terrain_model = world_model.rigidObject("height")
    # setup inputs and visualization
    logging = deque()
    keyboard_hook = EventLoggerGLPlugin(logging)
    vis.pushPlugin(keyboard_hook)
    vis.add("za warudo",world_model)
    vis.show()

    while vis.shown():

        if len(logging) != 0:
            event = logging.popleft()
        else:
            event = "NULL"
            key = "null"
            vis.update()
            time.sleep(1.0/30.0)

            continue
        if event[0] == "keyboard":
            key = event[1]
            try:
                for name in arrow_names:
                    vis.remove(name)
            except:
                pass
            arrow_names = []
        else:

            continue
        vis.lock()
        prev_transform = terrain_sim.elements["scoop"].getTransform()
        new_transform = move_body(key,scoop_model)
        terrain_sim.update_configuration("scoop",new_transform)
        # terrain_sim.update_configuration("occupied",new_transform)
        diff_t = se3.error(new_transform,prev_transform)
        diff_t = (so3.from_moment(diff_t[:3]),diff_t[3:])
        map_update, force_points, forces, _ = terrain_sim.sim_interaction(scoop_model, diff_t, terrain_model, terrain_properties)
        # geo = terrain_model.geometry()
        # height.setHeights(height.getHeights() + map_update)
        # geo.setHeightmap(height)
        if len(force_points) != 0:
            pc = klampt.PointCloud()
            pc.setPoints(force_points)
            colors = [(1,0,0) for i in range(len(force_points))]
            pc.setColors(colors,color_format=('r','g','b'))
            vis.add("collisions",pc)

            # for i, force in enumerate(forces):
            #     arrow_cpy = arrow.copy()
            #     arrow_start = np.array(force_points[i])
            #     arrow_end = np.array(-force) + arrow_start
            #     arrow_T  = compute_arrow_transform(arrow_start, arrow_end)
            #     arrow_cpy.setCurrentTransform(arrow_T[0], arrow_T[1])
            #     vis.add(f"arrow_{i}",arrow_cpy)
            #     vis.setAttribute(f"arrow_{i}","color",(clamp(np.linalg.norm(force),1,0),0,0))
            #     vis.hideLabel(f"arrow_{i}")
            #     arrow_names.append(f"arrow_{i}")
        vis.update()
        vis.unlock()
        time.sleep(1.0/30.0)
