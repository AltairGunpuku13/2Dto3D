import numpy as np
import laspy
import h5py
# Visualization
import open3d as o3d

pcd_o3d = o3d.io.read_point_cloud("./pointCloudDeepLearning1.ply")
print(pcd_o3d)
print('PLY file loaded')
print(dir(pcd_o3d))
cl, ind = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
pcd_o3d = pcd_o3d.select_by_index(ind)
pcd_o3d.estimate_normals()
pcd_o3d.orient_normals_to_align_with_direction()
#o3d.visualization.draw_geometries([pcd_o3d])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_o3d, depth=5, width=0, scale=1.1, linear_fit=False)[0]

#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_o3d, depth=10, width=0, scale=1.1, linear_fit=False)[0]
bbox = pcd_o3d.get_axis_aligned_bounding_box()
p_mesh_crop = mesh.crop(bbox)
rotation = p_mesh_crop.get_rotation_matrix_from_xyz((np.pi, 0, 0))
p_mesh_crop.rotate(rotation, center=(0, 100, 0))
# Calculate the normals of the vertex
p_mesh_crop.compute_vertex_normals()
# Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
#p_mesh_crop.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))

# save the mesh
o3d.io.write_triangle_mesh(f'./mesh.obj', p_mesh_crop)

# visualize the mesh
o3d.visualization.draw_geometries([p_mesh_crop], mesh_show_back_face=True)