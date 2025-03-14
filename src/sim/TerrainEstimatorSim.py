import klampt
import numpy as np
import pdb
from klampt import model, vis
from klampt.math import se3, so3
from klampt.math import vectorops as vo
#from ..HeightMapEstimation.DynamicHeightMap import DynamicHeightMap

# 3rd degree lookup table for f_i
c_table = np.array([
    [ 0.00212,  -0.06796,  -0.02634],   # k = 1
    [-0.02320,  -0.10941,  -0.03436],   # k = 2
    [-0.20890,   0.04725,   0.45256],   # k = 3
    [-0.43083,  -0.06914,   0.00835],   # k = 4
    [-0.00259,  -0.05835,   0.02553],   # k = 5
    [ 0.48872,  -0.65880,  -1.31290],   # k = 6
    [-0.00415,  -0.11985,  -0.05532],   # k = 7
    [ 0.07204,  -0.25739,   0.06790],   # k = 8
    [-0.02750,  -0.26834,  -0.16404],   # k = 9
    [-0.08772,   0.02692,   0.02287],   # k = 10
    [ 0.01992,  -0.00736,   0.02927],   # k = 11
    [-0.45961,   0.63758,   0.95406],   # k = 12
    [ 0.40799,   0.08997,  -0.00131],   # k = 13
    [-0.10107,   0.21069,  -0.11028],   # k = 14
    [-0.06576,   0.04748,   0.01487],   # k = 15
    [ 0.05664,   0.20406,  -0.02730],   # k = 16
    [-0.09269,   0.18519,   0.10911],   # k = 17
    [ 0.01892,   0.04934,  -0.04097],   # k = 18
    [ 0.01033,   0.13527,   0.07881],   # k = 19
    [ 0.15120,  -0.33207,  -0.27519],   # k = 20
])
powers = np.array([
    [ 0,  0,  0],   # k = 1
    [ 1,  0,  0],   # k = 2
    [ 0,  1,  0],   # k = 3
    [ 0,  0,  1],   # k = 4
    [ 2,  0,  0],   # k = 5
    [ 0,  2,  0],   # k = 6
    [ 0,  0,  2],   # k = 7
    [ 1,  1,  0],   # k = 8
    [ 0,  1,  1],   # k = 9
    [ 1,  0,  1],   # k = 10
    [ 3,  0,  0],   # k = 11
    [ 0,  3,  0],   # k = 12
    [ 0,  0,  3],   # k = 13
    [ 1,  2,  0],   # k = 14
    [ 2,  1,  0],   # k = 15
    [ 0,  1,  2],   # k = 16
    [ 0,  2,  1],   # k = 17
    [ 2,  0,  1],   # k = 18
    [ 1,  0,  2],   # k = 19
    [ 1,  1,  1],   # k = 20
],dtype=int)
# Access example: c[i, j] corresponds to c^{i}_{j}
def is_valid_object(object: klampt.Geometry3D) -> bool:
    r"""
    Helper function to check if a loaded robot is valid.
    Returns false if loaded robot has zero links
    """
    return not object.empty()

def is_valid_robot(robot: klampt.RobotModel) -> bool:
    r"""
    Helper function to check if a loaded robot is valid.
    Returns false if loaded robot has zero links
    """
    return True if robot.numLinks() > 0 else False

def calculate_triangle_area(vertices: list) -> float:
    r"""
    Helper function to compute the area of a triangle
    from vertices
    """
    vertices_copy = np.array(vertices.copy())
    ab = vertices_copy[2] - vertices_copy[0]
    ac = vertices_copy[1] - vertices_copy[0]
    return (0.5) * np.linalg.norm(np.cross(ab,ac))

def calculate_convex_polygon_area(vertices: list) -> float:
    r"""
    Calculates the area of a convex polygon given its vertices
    in clockwise order
    """
    assert len(vertices) > 2
    centroid = np.average(vertices,axis=1)
    total_area = 0
    for i in range(len(vertices)-1):
        used_vertices = [centroid,vertices[i],vertices[i+1]]
        total_area += calculate_triangle_area(used_vertices)
    return total_area
def find_intersection(line1: list[list], line2: list[list]) -> list:
    r"""
    Finds the point of intsersection between two lines in a 2D plane
    """
    x1,y1,z1 = line1[0]
    x2,y2,z2 = line1[1]
    x3,y3,z3 = line2[0]
    x4,y4,z4 = line2[1]

    x_intersection = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)
    y_intersection = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)

    # find z intersection based off of x,y intersection
    if x4 != x3:
        grad = (z4-z3)/(x4-x3)
        z_intersection = z3 + grad * (x_intersection - x3)
    elif y4 != y3:
        grad = (z4-z3)/(y4-y3)
        z_intersection = z3 + grad * (y_intersection - y3)
    else:
        z_intersection = z4
    
    return [x_intersection, y_intersection, z_intersection]


def clip_edge(clipper_segment: list[list], clippee_segment: list[list], clip_height: float) -> list[list]:
    r"""
    Clips polygon line segment against the clipper line segment

    :return new_vertices: new polygon vertices
    """
    x1,y1,z1 = clipper_segment[0]
    x2,y2,z2 = clipper_segment[1]
    x3,y3,z3 = clippee_segment[0]
    x4,y4,z4 = clippee_segment[1]

    #clip z's
    z3 = min(clip_height, z3)
    z4 = min(clip_height, z4)

    # Calculate whether clipee points are within or outside the clipper polygon
    P_3 = (x2 - x1)*(y3 - y1) - (y2 - y1) * (x3 - x1)
    P_4 = (x2 - x1)*(y4 - y1) - (y2 - y1) * (x4 - x1)

    # Case 1: Both points are inside clipper polygon
    if P_3 < 0 and P_4 < 0:
        return [[x4,y4,z4]]
    
    # Case 2: First vertex outside, second inside
    if P_3 >= 0 and P_4 < 0:
        intersection = find_intersection(clipper_segment, clippee_segment)
        return [intersection,[x4,y4,z4]]
    
    # Case 3: First vertex inside, second outside
    if P_3 < 0 and P_4 >= 0:
        intersection = find_intersection(clipper_segment, clippee_segment)
        return [intersection]

    # Case 4: Both outside
    return []

def clip_polygons(bounding_vertices: list[list], intruding_vertices: list[list], clip_height: float):
    r"""
    Given two convex polygons, will clip the vertices of the intruding
    polygon such that it fets within the bounds of the bounding
    polygon

    Uses the Sutherland-Hodgman Algorithm for clipping polygons

    :param bounding_vertices: list of vertices that make up the
                              bounding polygon.
                              NOTE: Vertices MUST be given in clockwise
                              order
    :param intruding_vertices: list of vertices that make up the
                               intruding polygon
                               NOTE: Vertices MUST be given in clockwise
                               order
    :param clip_height: used to clip the z value of the intruding polygon

    :return clipped_vertices: list of clipped vertices of the intruding
                              polygon
    """
    num_bounding_vert = len(bounding_vertices)
    curr_intruding_vert = intruding_vertices
    # For the current edge we are clipping on
    for bound_i in range(num_bounding_vert):

        clipper_start = bounding_vertices[bound_i]
        clipper_end = bounding_vertices[(bound_i+1)%num_bounding_vert]
        clipper_segment = [clipper_start, clipper_end]

        # For the current polygon edge being clipped
        num_clippee_vert = len(curr_intruding_vert)
        new_vert_list = []
        for clippee_i in range(num_clippee_vert):

            clippee_start = curr_intruding_vert[clippee_i]
            clippee_end = curr_intruding_vert[(clippee_i+1)%num_clippee_vert]
            clippee_segment = [clippee_start, clippee_end]

            new_vert_list += clip_edge(clipper_segment, clippee_segment, clip_height)
        curr_intruding_vert = new_vert_list

    return curr_intruding_vert

def triangle_bounding_box(triangle_vertices: list[list]) -> list:
    r"""
    Returns two points that comprise of the bounding box for
    a triangle
    """
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf

    for vertex in triangle_vertices:
        x,y,z= vertex
        min_x = min(min_x,x)
        min_y = min(min_y,y)

        max_x = max(max_x,x)
        max_y = max(max_y, y)

    return [[min_x, min_y,0], [max_x,max_y,0]]

def index_to_point(i,j,k,bbox,map_shape,pose):
    x = bbox[0][0] + (bbox[1][0] - bbox[0][0]) * i / (map_shape[0]-1)
    y = bbox[0][1] + (bbox[1][1] - bbox[0][1]) * j / (map_shape[1]-1)
    z = bbox[0][2] + (bbox[1][2] - bbox[0][2]) * k / (map_shape[2]-1)
    return se3.apply(pose,(x,y,z))

class TerrainEstimatorSim():
    r"""
    Forward simulator for terrain property estimation.
    """
    def __init__(self,world:klampt.WorldModel=None, sim_params:dict=None):
        self.world = world
        self.sim_params = sim_params

        self.elements = {}
        self.previous_configuration = {} # mapping from element_name -> previous configuration (list)

        self.collision_surfaces = []
        self.resistive_heightmap_names = []

        if self.world is not None:
            self._load_elements_from_world()
        if self.world is not None and sim_params is not None:
            self._load_important_names()

    def get_local_tf(self, local_frame:str , other_frame:str):
        r"""
        Computes other frame transform in local frame transform

        :return other_to_world: RigidTransform  of other frame in
                                local frame coordinates
        """
        local_to_world = self.elements[local_frame].getTransform()
        other_to_world = self.elements[other_frame].getTransform()
    
        world_to_local = se3.inv(local_to_world)
        
        other_to_world = se3.apply(world_to_local, other_to_world)

        return other_to_world 
    
    def add_geometry(self, *args):
        r"""
        Add a geometry to the WorldModel

        add_geometry(Geometry3D, name)
        add_geometry(fn, name)
        """
        arg1 = args[0]
        name = args[1]

        # load geometry
        if isinstance(arg1,str):
            geom = klampt.Geometry3D()
            loaded = geom.loadFile(arg1)
            if not loaded:
                return -1
        elif isinstance(arg1,klampt.Geometry3D):
            geom = arg1
        else:
            return -1
        
        # create new RigidObjectModel and set its geometry
        rigid_model = self.world.makeRigidObject(name)
        rigid_model.geometry().set(geom)
        self.elements[name] = rigid_model

    def add_robot(self, fn: str, name: str):
        r"""
        Add a robot to the WorldModel

        :param fn: file path to .rob or .urdf file
        :param name: name of the robot
        """

        robot_model = self.world.loadRobot(fn)

        # check for empty robot model
        if not is_valid_robot(robot_model):
            return False 
        
        robot_model.setName(name)
        self.elements[name] = robot_model

    def set_object_config(self, object_name:str, configuration: tuple) -> bool:
        r"""
        Sets the configuration of a rigid body in the WorldModel. Does not
        save previous configuration
        """
        object_model = self.world.rigidObject(object_name)
        if not is_valid_object(object_model.geometry()):
            return False
        
        object_model.setTransform(configuration[0],configuration[1])
        return True

    def update_object_config(self, object_name:str, configuration: list[float]) -> bool:
        r"""
        Updates the configuration of the named object and updates
        its previous configuration. Returns False if configuration
        could not be set
        """
        object_model = self.world.rigidObject(object_name)
        if not is_valid_object(object_model.geometry()):
            return False
        
        # update previous config and set new config
        prev_config = object_model.getTransform()
        self.previous_configuration[object_model] = prev_config

        object_model.setTransform(configuration[0],configuration[1])
        return True    

    def set_robot_config(self, robot_name:str, joint_angles:list[float]) -> bool:
        r"""
        Sets the configuration of a robot in the WorldModel. Does not save
        previous configuration.
        """
        robot_model = self.world.robot(robot_name)
        if not is_valid_robot(robot_model):
            return False

        robot_model.setConfig(joint_angles)
        return True

    def update_robot_config(self, robot_name:str, configuration: list[float]) -> bool:
        r"""
        Updates the configuration of the named robot and updates
        its previous configuration. Returns False if configuration
        could not be set
        """
        robot_model = self.world.robot(robot_name)
        if not is_valid_robot(robot_model):
            return False
        
        # update previous config and set new config
        prev_config = robot_model.getConfig()
        self.previous_configuration[robot_name] = prev_config

        robot_model.setConfig(configuration)
        return True

    def update_configuration(self, element_name: str, configuration) -> bool:
        r"""
        Updates the configuration for a given entity. If
        the entity is a robot, configuration should be a
        list of joint angles with shape (1xN) where N
        is the number of links. If the element is a rigid object,
        the configuration should be a rigid transform (R,t).

        Important note, updating the configuration also updates the
        elements previous position in the simulator which is used
        for tracking motions.

        :param element_name: name of object to be modified
        :param configuration: new configuration
        """
        curr_element = self.elements[element_name]

        # update configuration of a RobotModel
        if isinstance(curr_element, klampt.RobotModel):
            if len(configuration) != curr_element.numLinks():
                print(f"TerrainEstimatorSim: Element {element_name} is of type klampt.Robotmodel, but new configuration \
                        only has {len(configuration)}.")
                return self.update_robot_config(element_name, configuration)
        elif len(configuration) == 2:
            return self.update_object_config(element_name, configuration)
        else:
            return False

    def sim_interaction(self, intruder_rigid_object: klampt.RigidObjectModel, delta_pose: tuple, terrain: klampt.RigidObjectModel, 
                        terrain_material_params: dict, gradiants=None):
        r"""
        Computes the expected forces of each triangle center

        :param intruder_rigid_object: TriangleMesh of the scooping surface
        :param delta_pose: change in pose (R,t) of the intruder from time t-1 to t
        :param terrain: Geometry3D of the terrain to be interacted with
        :param terrain_material_params: parameters of the deformable terrain 

        :return new_terrain_state: Geometry3D model with the updated state of the model
        :return force_pts: (Nx3) list of point-force locations on the mesh in the mesh
                           frame
        :return forces: (Nx3) list of force vectors for each force point
        :return gradients: TODO: Figure that out
        """

        # All code adapted from 3D-RFT implementation found at: TODO: fill in paper link
        intruder_rigid_object_mesh = intruder_rigid_object.geometry().getTriangleMesh()
        curr_object_pose = intruder_rigid_object.getTransform()

        mesh_vertices = np.array(intruder_rigid_object_mesh.getVertices()[intruder_rigid_object_mesh.getIndices().copy()])
        triangle_normals = np.array(intruder_rigid_object_mesh.triangleNoormals().copy())

        terrain_coefficients = terrain_material_params["zeta"]
        #surface_friction = terrain_material_params["u_surf"]
        resolution = terrain_material_params["grid_resolution"]
        mins = terrain_material_params["grid_minimum"]

        # Compute and transform triangle centroids and normals
        def convert_centroid_frame(centroids,frame):
            world_frame_centroids = np.array([se3.apply(frame,centroid) for centroid in centroids])
            return world_frame_centroids
        def rotate_normals(normals,R):
            world_frame_normals = np.array([so3.apply(R,normal) for normal in normals])
            return world_frame_normals

        triangle_centroids = np.average(mesh_vertices.copy(),axis=1)
        triangle_centroids_cpy = triangle_centroids.copy()

        # convert centroids and normals into the world frame
        triangle_centroids = convert_centroid_frame(triangle_centroids_cpy,curr_object_pose)
        mesh_vertices = convert_centroid_frame(mesh_vertices.reshape((-1,3)),curr_object_pose).reshape((-1,3,3))
        triangle_normals = rotate_normals(triangle_normals.copy(), curr_object_pose[0])

        # Compute depths by taking difference between heightmap heights and colliding ray
        heightmap_pose = terrain.getTransform()

        def point_to_index(mins: tuple,grid_pose:tuple ,resolution: float,point: tuple, shape: np.ndarray):
            world_to_grid = se3.inv(grid_pose)
            x_min,y_min = mins
            x,y = point
            coords = [x,y,0]
            coords = se3.apply(world_to_grid,coords)
            i = int((coords[0] - x_min)/resolution)
            j = shape[1] - int((coords[1] - y_min)/resolution)
            return [i,j]
        
        heightmap = np.array(terrain.geometry().getHeightmap().heights)
        grid_z = heightmap_pose[1][2]

        # get indices from points
        index_points = np.array([point_to_index(mins,heightmap_pose,resolution,(pt[0],pt[1]),heightmap.shape) for pt in triangle_centroids])
        valid = []
        x_max, y_max = heightmap.shape
        for index in index_points:
            x,y = index
            if x >= 0 and x < x_max and y >= 0 and y < y_max:
                valid += [True]
            else:
                valid += [False]

        # Compute centroid velocities from EE delta_pose
        to_old = se3.inv(delta_pose)
        prev_pose = (so3.mul(to_old[0],curr_object_pose[0]),vo.add(to_old[1],curr_object_pose[1]))
        triangle_centroids_prev = convert_centroid_frame(triangle_centroids_cpy,prev_pose)
        centroid_velocities = triangle_centroids - triangle_centroids_prev

        # Prune centroids that hit outside of valid range
        index_points = index_points[valid]
        triangle_centroids = triangle_centroids[valid]
        centroid_velocities = centroid_velocities[valid]/np.linalg.norm(centroid_velocities[valid],axis=1).reshape((len(centroid_velocities),1))
        triangle_normals = triangle_normals[valid]
        mesh_vertices = mesh_vertices[valid]

        depths = np.array([heightmap[index_points[i][0], index_points[i][1]]+grid_z-triangle_centroids[i][2]  for i in range(len(triangle_centroids))])

        # Prune centroids that are not part of the leading edge
        leading_edge = np.where(np.sum(np.multiply(triangle_normals, centroid_velocities),axis=1) > 0, True, False).flatten()
        depths = depths[leading_edge]
        triangle_centroids = triangle_centroids[leading_edge]
        centroid_velocities = centroid_velocities[leading_edge]
        triangle_normals = triangle_normals[leading_edge]
        mesh_vertices = mesh_vertices[leading_edge]

        # # break leading edge surfaces into cell-specific faces
        scoop_bbox = intruder_rigid_object.geometry().getBB()
        for i,face in enumerate(triangle_normals):
            
            # find bounding box for triangle support geometry
            support_vertices = mesh_vertices[i]
            bbox_triangle = triangle_bounding_box(support_vertices)
            # bbox_indices = [point_to_index(mins, heightmap_pose, resolution, point, heightmap.shape) for point in bbox_triangle]
            # bbox_shape = (bbox_indices[1][0] - bbox_indices[0][0], bbox_indices[1][1] - bbox_indices[0][1])
            # print(bbox_indices)
            # # for each cell in the bounding box of the support geometry,
            # # find the geometry in that cell
            # for i in range(bbox_indices[0][0], bbox_indices[1][0]+1):
            #     for j in range(bbox_indices[1][0],bbox_indices[1][1]+1):

            #         cell_center = index_to_point(i,j,0,scoop_bbox,bbox_shape,curr_object_pose)

        # Prune centroids that are not under the surface
        valid_depths = np.where(depths > 0, True, False)
        depths = depths[valid_depths]
        triangle_centroids = triangle_centroids[valid_depths]
        centroid_velocities = centroid_velocities[valid_depths]
        triangle_normals = triangle_normals[valid_depths]
        mesh_vertices = mesh_vertices[valid_depths]

        def find_beta_angle(n,r,z):
            nz_dot = np.dot(n,z)
            nr_dot = np.dot(n,r)
            if nr_dot >= 0 and nz_dot >= 0:
                return -np.arccos(nz_dot)
            elif nr_dot >= 0 and nz_dot < 0:
                return np.pi - np.arccos(nz_dot)
            elif nr_dot < 0 and nz_dot >= 0:
                return np.arccos(nz_dot)
            else:
                return -np.pi + np.arccos(nz_dot)

        force_points = []
        force_cells = []
        forces = []

        # Iterate through each centroid and find forces on body
        for i, centroid in enumerate(triangle_centroids):
            #breakpoint()
            # Step 4: Find local coordinate frame
            z = np.array([0,0,1.0])
            n = triangle_normals[i]/np.linalg.norm(triangle_normals[i])
            
            # normalize vectors
            v = np.array(centroid_velocities[i]) # velocity
            v = v/np.linalg.norm(v)

            v_on_z = v - np.dot(v,z)*z
            if np.linalg.norm(v_on_z) <= 1e-6:
                r = (n - np.dot(n,z)*z)/(np.linalg.norm(n - np.dot(n,z)*z))
            else:
                r = (v_on_z)/(np.linalg.norm(v_on_z))
            theta = np.cross(z,r)

            # Step 5: Find RFT characteristic angles
            vr = np.clip(np.dot(v,r),-1,1) # clamp values to stay within domain
            gamma = np.arccos(vr) if np.dot(v,z) <= 0 else -np.arccos(vr) 

            # Find psi
            nro_mag = np.linalg.norm(n - np.dot(n,z)*z)
            if nro_mag <= 1e-6:
                psi = 0
            else:
                nro = (n - np.dot(n,z)*z)/nro_mag
                psi = np.arctan2(np.dot(nro,theta),np.dot(nro,r)) # Might be a typo in the directions since it says nro*r/nro*r. Figured they meant theta instead
            
            if psi == 0 and abs(abs(gamma) - np.pi/2) <= 1e-6:
                r = np.array([1.0,0,0])
            # Step 6: calculate f1, f2, f3
            beta = find_beta_angle(n,r,z) # Note: do this after potentially resetting r!

            sin_g = np.sin(gamma)
            cos_g = np.cos(gamma)
            sin_b = np.sin(beta)
            cos_b = np.cos(beta)
            sin_p = np.sin(psi)
            cos_p = np.cos(psi)

            gv = sin_g
            gn = cos_b
            nv = cos_p * cos_g * sin_b + sin_g * cos_b

            p_i = np.array([gv, gn, nv])
            T_all = np.prod(np.power(p_i,powers), axis=1).reshape((20,1))
            f_i = T_all.T @ c_table
            f_i = f_i[0]

            # Step 7: calculate alpha_r, alpha_theta, alpha_z
            alpha_r = f_i[0] * sin_b * cos_p + f_i[1] * cos_g
            alpha_theta = f_i[0] * sin_b * sin_p
            alpha_z = -f_i[0] * cos_b - f_i[1] * sin_g - f_i[2]

            # Step 8 - 10: Extract cell coefficient and calculate alpha
            # Note: step 9 and 10 might be switched in the paper implementation?
            x,y = point_to_index(mins,heightmap_pose,resolution,(centroid[0],centroid[1]),heightmap.shape)

            zeta = terrain_coefficients[x][y]
            
            alpha_gen = alpha_r * r + alpha_theta * theta + alpha_z * z

            # Step 11: multply by depth, zeta, and area to obtain triangle-specific forces
            triangle_area = calculate_triangle_area(mesh_vertices[i])
            force_points.append(centroid)
            forces.append(alpha_gen * zeta * triangle_area * depths[i])
            force_cells.append((x,y))
        
        # Use forces to update map
        map_update = np.zeros(heightmap.shape) # how much each cell's height changes
        cell_force = np.zeros((heightmap.shape[0],heightmap.shape[1],3)) # total force experienced on a cell

        # fill out force field
        for i, force_point in enumerate(force_points):
            x,y = force_cells[i]
            cell_force[x][y] += forces[i]

        intruder_occupancy = intruder_rigid_object.geometry().convert("OccupancyGrid",0.005)

        intruder_occupancy.setCurrentTransform(curr_object_pose[0],curr_object_pose[1])
        def index_to_point_occupancy(i,j,k,bbox,map_shape,pose):
            x = bbox[0][0] + (bbox[1][0] - bbox[0][0]) * i / (map_shape[0]-1)
            y = bbox[0][1] + (bbox[1][1] - bbox[0][1]) * j / (map_shape[1]-1)
            z = bbox[0][2] + (bbox[1][2] - bbox[0][2]) * k / (map_shape[2]-1)
            return se3.apply(pose,(x,y,z))
        
        # Step 1: Caclulate collisions and move soil based on overlapping
        # volumes
        occupancy_grid = intruder_occupancy.getOccupancyGrid().getValues()
        intruder_occupancy.setCurrentTransform(so3.identity(),[0,0,0])

        #TODO: Fix index to point conversion
        pointed = []
        i_shape, j_shape, k_shape = occupancy_grid.shape
        bbox = intruder_occupancy.getBBTight()
        for i in range(i_shape):
            for j in range(j_shape):
                for k in range(k_shape):
                    occupied = occupancy_grid[i][j][k]

                    # if cell is occupied, check if it collides with
                    # terrain
                    if occupied:
                        
                        # find occupancy index to point
                        point = index_to_point_occupancy(i,j,k,bbox,occupancy_grid.shape,curr_object_pose)

                        #find that point's index w.r.t. heightmap index
                        x,y = point_to_index(mins,heightmap_pose,resolution,(point[0],point[1]),heightmap.shape)

                        # check if subsurface
                        if point[2] < heightmap[x][y]:

                            pointed.append(point)
                            force = -cell_force[x][y]
                            force_mag = np.linalg.norm(force) 

                            if force_mag > 0:
                                force = force/force_mag

                            x_dir = np.rint(force[0])
                            y_dir = np.rint(force[1])

                            new_x = int(np.clip(x+x_dir,0,heightmap.shape[0]-1))
                            new_y = int(np.clip(y+y_dir,0,heightmap.shape[1]-1))

                            volume_moved = resolution
                            map_update[x][y] -= volume_moved
                            map_update[new_x][new_y] += volume_moved
                            #print(x,y,force,force_mag,map_update[x][y],map_update[new_x][new_y],new_x,new_y)
                    else:
                        continue
        
        # Step 2: Settle
        # i_shape, j_shape = heightmap.shape
        # for i in range(i_shape):
        #     for j in range(j_shape):
        #mesh_vertices.reshape((-1,3))
        return map_update, bbox_triangle, forces, None

    def _load_elements_from_world(self):
        r"""
        Creates a dictionary mapping all element names
        to their associated elements (robots, rigid objects, terrains).
        Used mainly for ease of access
        """
        world = self.world
        if world is not None:
            self.elements.update(world.getRobotsDict())
            self.elements.update(world.getRigidObjectsDict())
            self.elements.update(world.getTerrainsDict())
        
    
    def _load_important_names(self):
        r"""
        Extracts the names of surfaces used in the collision
        between the terrain estimation height map and the
        end effector from sim_params
        """
        if self.sim_params is None:
            raise("sim_params file not specified")
        
        self.collision_surfaces = self.sim_params["tool_names"]
        self.resistive_heightmap_names = self.sim_params["dynamic_heightmap_names"]#


    