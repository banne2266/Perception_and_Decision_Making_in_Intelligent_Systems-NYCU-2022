import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math
import json

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "../replica/apartment_0/habitat/mesh_semantic.ply"
path = "../replica/apartment_0/habitat/info_semantic.json"
path_data = np.load("path_info.npy")
homography = np.load("homography.npy")
with open("setting.json", "r") as f:
    target_str = json.load(f)


temp = np.ones((path_data.shape[0], 3))
temp[:, 0:2] = path_data
path_on_habitat = np.dot(homography, temp.T).T[:, 0:2] / 100
path_on_habitat = np.flip(path_on_habitat, axis=0)

target_str = target_str["target"]
target_ids = {"refrigerator": 67, "rack": 66, "cushion": 29, "lamp": 47, "cooktop": 32}
target_id = target_ids[target_str]

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

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

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.03) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([path_on_habitat[0][0], 0.0, path_on_habitat[0][1]])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


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
print("#############################")


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)
        img = transform_rgb_bgr(observations["color_sensor"])
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("semantic", transform_semantic(id_to_label[observations["semantic_sensor"]]))

        temp = id_to_label[observations["semantic_sensor"]]
        mask = temp == target_id
        mask = np.array(mask, dtype=np.uint8)
        co = np.zeros((512, 512, 3), dtype=np.uint8)
        co[:, :, 2] = 150
        redMask = cv2.bitwise_and(co, co, mask=mask)

        img = cv2.addWeighted(redMask, 1, img, 1, 0)
        cv2.imshow("aaa", img)

        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        return img



action = "move_forward"
navigateAndSee(action)



cur_node = 1

def l2_distance(p1, p2):
    return math.sqrt(np.sum((p1 - p2) ** 2))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("./results/" + target_str + ".mp4", fourcc, 60.0, (512,  512))

while True:
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    cur_pos = np.array([sensor_state.position[0], sensor_state.position[2]])
    w, x, y, z = sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z

    face_angle = np.rad2deg(np.arccos(w)) * 2
    if y < 0:
        face_angle = -face_angle
    if face_angle < 0:
        face_angle += 360
    face_angle -= 180
    print("face", face_angle)

    next_target = path_on_habitat[cur_node]
    diff = next_target - cur_pos
    target_angle = np.rad2deg(np.arctan2(diff[0], diff[1]))
    print("target", target_angle)

    diff_angle = face_angle - target_angle
    print("diff", diff_angle)

    if diff_angle > 180:
        diff_angle -= 360
    if diff_angle < -180:
        diff_angle += 360
    print("new: diff", diff_angle)

    if diff_angle > 2:
        action = "turn_right"
        frame = navigateAndSee(action)
        print("action: RIGHT")
        out.write(frame)
    elif diff_angle < -2:
        action = "turn_left"
        frame = navigateAndSee(action)
        print("action: LEFT")
        out.write(frame)
    else:
        action = "move_forward"
        frame = navigateAndSee(action)
        print("action: FORWARD")
        out.write(frame)
    
    if l2_distance(cur_pos, next_target) < 0.4:
        cur_node += 1
        if cur_node == len(path_on_habitat):
            break
    keystroke = cv2.waitKey(16)

    #keystroke = cv2.waitKey(0)
    if keystroke == ord(FINISH):
        print("action: FINISH")
        break

out.release()
cv2.destroyAllWindows()