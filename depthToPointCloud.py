import cv2
import torch
import time
import numpy as np
import open3d as o3d

# only read the last frame of the video, do not know how to combine point cloud

# Q matrix - Camera parameters - Can also be found using stereoRectify
Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)


# Load a MiDas model for depth estimation
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)

#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)

#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)


midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Open up the video capture from a webcam
cap = cv2.VideoCapture("3.mp4") # read the video with openCV
TotalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #find the total number of frames of the video
print(TotalFrame)
numframe = 10
FrameNext = round(TotalFrame/numframe)
AngleNext =round(90*100/numframe)/100
print(FrameNext)
print(AngleNext)
ret, frame = cap.read()
framecur = 0
anglecur = 0
#stereo = cv2.StereoBM_create(numDisparities=4096, blockSize=5)
cal = 0
vis = o3d.visualization.Visualizer()
vis.create_window()
pcdfinal = []
trajectory = o3d.io.read_pinhole_camera_trajectory
while (1):
    
    print(str(framecur)+"/"+str(TotalFrame))
    
    if (framecur > TotalFrame-1):
        break
    cap.set(TotalFrame, 1)
    success, img = cap.read()
    if cal == framecur:
        framecur = framecur + FrameNext
        anglecur = anglecur + AngleNext
        start = time.time()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #disparity = stereo.compute(img,disparity)
        # Apply input transforms
        input_batch = transform(img).to(device)

        # Prediction and resize to original resolution
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=True,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

        #Get rid of points with value 0 (i.e no depth)
        mask_map = depth_map > 0.4

        #Mask colors and points. 
        output_points = points_3D[mask_map]
        output_colors = img[mask_map]
        im = o3d.geometry.RGBDImage.create_from_color_and_depth(output_colors,output_points,1000,5,False)
        intrinsic = trajectory.parameters[framecur].intrinsic

        #print(output_points)
        

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
        
        
        
        pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(output_points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(np.round(output_colors))
        #vis.add_geometry(pcd_o3d)
        #vis.poll_events()
        #vis.update_renderer()
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map)
        o3d.visualization.draw_geometries([pcd_o3d])
    
    
    
    
    if cv2.waitKey(33) == ord('a'):
        print("pressed a")
        break
    if framecur >= TotalFrame:
        break
        #if cv2.waitKey(5) & 0xFF == 27:
        #    break
    print(cal)
    cal += 1



# --------------------- Create The Point Clouds ----------------------------------------

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')
 

output_file = 'pointCloudDeepLearning.ply'
#Generate point cloud 
create_output(x1, x2, output_file)

cap.release()
cv2.destroyAllWindows()

