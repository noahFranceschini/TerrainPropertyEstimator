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

def compute_arrow_transform(arrow_start: list, arrow_end:list):
    t = arrow_start
    R = so3.align([0,0,1],arrow_end - arrow_start)
    return (R,t)

def move_body(key,obj,step_size=0.0025):
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
        elif key == "up": # Various rotations
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
        return (rotation,translation)

if __name__ == "__main__":

    arrow_fn = "D:/2025Research/TerrainPropertyEstimator/src/meshes"

    #load arrow mesh
    arrow = klampt.Geometry3D()
    arrow.loadFile(arrow_fn + "/arrow.stl")
    arrow.convert("TriangleMesh")

    scoop_mesh_fn = "D:/2025Research/TerrainPropertyEstimator/src/meshes/scoop_eih.STL" #"TerrainPropertyEstimator/src/meshes/scoop_eih.STL" 
    default_scoop_pos = (so3.from_moment([-1.13906602279157, -1.1888265740976542, 1.2987508365563492]),[0,0.0,0.17])
    
    heightmap_raw = np.zeros((400,400))

    zetas = np.ones(heightmap_raw.shape) * 2.23*1e6
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
    terrain_sim.set_object_config("scoop", default_scoop_pos)
    scoop_model = world_model.rigidObject("scoop")
    terrain_model = world_model.rigidObject("height")

    # setup inputs and visualization
    logging = deque()
    keyboard_hook = EventLoggerGLPlugin(logging)
    vis.pushPlugin(keyboard_hook)
    vis.add("the world",world_model)
    vis.show()
    arrow_names = []

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
        diff_t = se3.error(new_transform,prev_transform)
        diff_t = (so3.from_moment(diff_t[:3]),diff_t[3:])

        map_update, force_points, forces, _ = terrain_sim.sim_interaction(scoop_model, diff_t, terrain_model, terrain_properties)
        geo = terrain_model.geometry()
        height.setHeights(map_update)
        geo.setHeightmap(height)

        if len(force_points) != 0:
            pc = klampt.PointCloud()
            pc.setPoints(force_points)
            colors = [(1,0,0) for i in range(len(force_points))]
            pc.setColors(colors,color_format=('r','g','b'))
            vis.add("collisions",pc)
            vis.hideLabel("collisions")
            for i, force in enumerate(forces):
                arrow_cpy = arrow.copy()
                arrow_start = np.array(force_points[i])
                arrow_end = np.array(-force) + arrow_start
                arrow_T  = compute_arrow_transform(arrow_start, arrow_end)
                arrow_cpy.setCurrentTransform(arrow_T[0], arrow_T[1])
                vis.add(f"arrow_{i}",arrow_cpy)
                vis.setAttribute(f"arrow_{i}","color",(clamp(np.linalg.norm(force),1,0),0,0))
                vis.hideLabel(f"arrow_{i}")
                arrow_names.append(f"arrow_{i}")
        vis.update()
        vis.unlock()
        time.sleep(1.0/30.0)
