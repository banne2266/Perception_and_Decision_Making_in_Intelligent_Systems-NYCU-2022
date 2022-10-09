import cv2
import open3d as o3d
import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
from sklearn.neighbors import NearestNeighbors


USE_O3D_ICP = 0
ICP_MAX_ITER = 30
ICP_SAMPLE_NUM = 2000


test_scene = "./apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0  # sensor pitch (x rotation in rads)
}

voxel_size = 8e-6
reconstruct_point_cloud = o3d.geometry.PointCloud()
pre_frame_pcd = None
pre_frame_pcd_fpfh = None
last_transform = np.identity(4)

estimated_path = []
ground_truth_path = []
ground_truth_init_loc = 0
lines = []
path_cnt = 0

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE


    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 1.0, 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


def depth_image_to_point_cloud(rgb:np.array, depth:np.array):
    f = 512 / np.tan(np.deg2rad(90) / 2) / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(512, 512, f, f, 255, 255)
    rgb = np.asarray(rgb, order="C")
    rgb = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def nearest_neighbor(source, target):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(target)
    distances, indices = neigh.kneighbors(source, return_distance=True)
    return distances.ravel(), indices.ravel()

def reject_outlier(distances, indices, voxel_size):
    index = np.where(distances < voxel_size * 5)
    return index[0], indices[index]

def optimal_transform(source, target):
    source_mu = np.mean(source, axis=0)
    target_mu = np.mean(target, axis=0)

    source_p = source - source_mu
    target_p = target - target_mu

    W = np.dot(source_p.T, target_p)
    U, S, Vt = np.linalg.svd(W)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
    

    t = target_mu.T - np.dot(R, source_mu.T)
    transform = np.identity(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform


def my_local_ICP(source, target, distance_threshold, trans_init):
    transform = np.array(trans_init)
    source = o3d.geometry.PointCloud(source)
    cur_iter = 0
    mean_distance = 9999
    source_pts = np.array(source.points)
    target_pts = np.array(target.points)

    source_iter = source.transform(transform)
    source_iter_pts = np.array(source_iter.points)

    while (cur_iter < ICP_MAX_ITER) & (mean_distance > distance_threshold):
        sampled_src_pts = source_iter_pts[np.random.choice(source_iter_pts.shape[0], \
            min(ICP_SAMPLE_NUM, source_iter_pts.shape[0]), replace=False), :]#random points in source

        distances, indices = nearest_neighbor(sampled_src_pts, target.points)
        indices_src, indices_target = reject_outlier(distances, indices, voxel_size)    
        transform = optimal_transform(sampled_src_pts[indices_src, :], target_pts[indices_target, :])

        source_iter = source_iter.transform(transform)
        source_iter_pts = np.array(source_iter.points)
        mean_distance = np.mean(distances)
        cur_iter += 1
    
    transform = optimal_transform(source_pts, source_iter_pts)
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = transform
    return result


def calculate_transform(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5

    trans_init = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(#global registration
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    result = o3d.pipelines.registration.registration_icp(#local registration (ICP)
        source_down, target_down, distance_threshold, trans_init.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()) if USE_O3D_ICP \
        else my_local_ICP(source_down, target_down, distance_threshold, trans_init.transformation)

    return result

def navigateAndSee(action=""):
    global pre_frame_pcd, pre_frame_pcd_fpfh, reconstruct_point_cloud, last_transform, path_cnt, ground_truth_init_loc
    if action in action_names:
        observations = sim.step(action)
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
        rgb = transform_rgb_bgr(transform_rgb_bgr(observations["color_sensor"]))
        depth = transform_depth(observations["depth_sensor"])
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        cur_frame_pcd = depth_image_to_point_cloud(rgb, depth)
        cur_frame_pcd, cur_frame_pcd_fpfh = preprocess_point_cloud(cur_frame_pcd, voxel_size)

        if pre_frame_pcd == None: #for the first frame
            reconstruct_point_cloud += cur_frame_pcd
            ground_truth_init_loc = sensor_state.position
            ground_truth_path.append([0, 0, 0])
            estimated_path.append([0, 0, 0])
            
        else:
            result = calculate_transform(cur_frame_pcd, pre_frame_pcd,
                                            cur_frame_pcd_fpfh, pre_frame_pcd_fpfh,
                                            voxel_size)
            last_transform = np.dot(last_transform, result.transformation, )
            to_add_pcd =  o3d.geometry.PointCloud(cur_frame_pcd)
            to_add_pcd.transform(last_transform)
            print(last_transform)

            reconstruct_point_cloud += to_add_pcd
            reconstruct_point_cloud.voxel_down_sample(voxel_size)

            if action == "move_forward":
                temp = (sensor_state.position - ground_truth_init_loc) / 255 / 255 / 1.5 * 10
                temp = np.array([temp[0], temp[1], -temp[2]])
                ground_truth_path.append(temp)

                estimated_path.append([last_transform[0,3], last_transform[1,3], last_transform[2,3]])
                lines.append([path_cnt, path_cnt+1])
                path_cnt += 1

        
        pre_frame_pcd = cur_frame_pcd
        pre_frame_pcd_fpfh = cur_frame_pcd_fpfh


def get_lineset(path, lines, color):
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(path),
        lines = o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set



def main():
    global reconstruct_point_cloud
    FORWARD_KEY="w"
    LEFT_KEY="a"
    RIGHT_KEY="d"
    FINISH="f"
    print("#############################")
    print("use keyboard to control the agent")
    print(" w for go forward  ")
    print(" a for turn left  ")
    print(" d for trun right  ")
    print(" f for finish and quit the program")
    print(" g for reconstruct the indoor model")
    print("#############################")

    action = "move_forward"
    navigateAndSee(action)

    while True:
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            navigateAndSee(action)
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            navigateAndSee(action)
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            navigateAndSee(action)
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        elif keystroke == ord("g"):
            print("action: RECONSTRUCT MODEL")

            points = np.asarray(reconstruct_point_cloud.points)
            reconstruct_point_cloud = reconstruct_point_cloud.select_by_index(np.where(points[:,1] > -0.00005)[0]) #remove ceiling

            estimated_path_line = get_lineset(estimated_path, lines, (1,0,0))
            ground_truth_path_line = get_lineset(ground_truth_path, lines, (0,0,0))

            dist = np.mean(np.linalg.norm(np.array(estimated_path) - np.array(ground_truth_path), axis=1))
            print(dist)

            o3d.visualization.draw_geometries([reconstruct_point_cloud, estimated_path_line, ground_truth_path_line])
            print("action: FINISH")
            break
        else:
            print("INVALID KEY")
            continue
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()