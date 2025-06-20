# TerrainPropertyEstimator

## Summary
This here is the code used for the Terrain Property Estimator that was my second project. The two main files that you will be using will be
`TerrainEstimatorSim.py` and `SimTest.py`. The `TerrainEstimatorSim contains the `TerrainEstimatorSim` class which computes all of the Resistive Force Theory
calculations given an intruding object's geometry, velocity, and the current height map of the terrain. The SimTest script was my way of testing this
simulator using the scoop end-effector mesh and some flat plate meshes.

### How To Use The Simulator Class
The main method that you will use is the `sim_interaction()` function within the `TerrainEstimatorSim` class. There are several other methods that I implemented
in an attempt to replicate the VSF simulator that came from Shaoxiong's project, however, the `sim_interaction()` method is the only real one that will be needed.

`sim_interaction()` takes in 4 arguments: `intruder_rigid_object`, `delta_pose`, `terrain`, and `terrain_material_params`. The `intruder_rigid_object` is simply a `klampt.RigidObjectModel` that
contains a `TriangleMesh` of the surface we wish to perform force calculations on. `delta_pose` is the difference in pose between `t` and `t + 1` of the rigid object in (R,t) format as used in
Klampt. `terrain` is another `klampt.RigidObjectModel` containing a `HeighMap` of our terrain. Lastly is the `terrain_material_params` which is a dictionary mapping strings to different terrain
properties. The 3 necessary parameters for this dictionary are:

- `zeta`: a 2D grid with the same dimensions as the heightmap. Each cell contains the cell's current resistivity coefficient as described in the 3D-RFT paper
- `grid_resolution`: Heightmap resolution
- `grid_minimum`: the bottom corner of the heightmap in world coordinates.

Given these arguments, the simulator will then return the force points, force vectors, and terrain deltas for the heightmap.

### Quirks
Naturally, there are some quirks and bugs in this code. Below is a non-comprehensive list of the things that will probably need to be addressed in the final product:

- Terrain updates reload the entire mesh in the SimTest which is a surprisingly costly thing to do
- The "Settle" step scans over the entire heightmap which for larger heightmaps is computationally inefficient
- Terrain updates favor one direction for some reason and will sometimes lead to piling around the scoop to be greater on one side than the other
- Terrain updates do not account for material that would be pushed under a cell occupied by the scoop, meaning material can "phase" through the bottom of the scoop upwards
- The direction of pushed materials are determined by the normalized aggregated forces calculated within the cell of moved material. While not inherently an issue, this method
  is sensitive to noise and cells with small force magnitudes
- If a flat intruding surface is pressed directly downwards into the terrain there will be no terrain updates since there will be no forces in the XY-plane


