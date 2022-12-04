import numpy as np
import cv2
import math
import json

USE_RRT_STAR = 1
start_point = (1360, 670)
#target_colors = {"refrigerator": (255, 0, 0), "rack": (0, 255, 133), "cushion": (255, 9, 92), "lamp": (160, 150, 20), "cooktop": (7, 255, 224)}
target_coors = {"refrigerator": (776, 559), "rack": (1190, 412), "cushion": (1584, 719), "lamp": (1370, 948), "cooktop": (637, 736)}

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global start_point
        print(x, ' ', y)
        start_point = (x, y)
        temp = img.copy()
        cv2.circle(temp, (x, y), 10, (0, 0, 255), -1)
        cv2.imshow("map", temp)
    return

def l2_distance(p1, p2):
    return math.sqrt(np.sum((p1 - p2) ** 2))


def RRT(map_img, start_point, target, step_size = 100, iter_num = 1000, goal_rate = 0.2, RRT_star = False, RRT_star_iter_num = 100):
    target_coordinate  = np.array(target_coors[target])
    start_point = np.array(start_point)
    found = 0

    node_coordinates = [start_point]
    node_parents = [0]
    RRT_path = []

    for _ in range(iter_num):
        x_rand = random_sample(goal_rate, target_coordinate)

        x_nearest_idx = nearest(node_coordinates, x_rand)
        x_nearest = node_coordinates[x_nearest_idx]

        x_new = steer(x_nearest, x_rand, step_size)

        if not check_collision_line(map_img, x_new, x_nearest):
            node_coordinates.append(x_new)
            node_parents.append(x_nearest_idx)

            if l2_distance(x_new, target_coordinate) < 10:
                found = 1
                cur_node = len(node_coordinates) - 1
                while cur_node != 0:
                    RRT_path.append(node_coordinates[cur_node])
                    cur_node = node_parents[cur_node]
                RRT_path.append(start_point)
                break

    # RRT*
    if RRT_star and found:
        for _ in range(RRT_star_iter_num):
            path_nodes = len(RRT_path)
            a = np.random.choice(path_nodes, size=2)
            if not check_collision_line(map_img, RRT_path[a[0]], RRT_path[a[1]]):
                del RRT_path[a[0]+1 : a[1]]
    
    img = draw_RRT_result(map_img, node_coordinates, node_parents, found, RRT_path)
    return RRT_path, img


def random_sample(goal_rate, target_coordinate):
    p = np.random.rand()
    if p <= goal_rate:
        return target_coordinate
    else:
        return np.random.randint(low = (330, 230), high = (1700, 1000), size=2)


def nearest(points, target):
    points = np.array(points)
    distances = np.sum((points - target) ** 2, axis=1)
    idx = np.argmin(distances)
    return idx


def steer(x_nearest, x_rand, step_size):
    distance = l2_distance(x_nearest, x_rand)
    if distance <= step_size:
        return x_rand
    diff = x_rand - x_nearest
    diff = diff / distance * step_size
    x_new = x_nearest + diff
    x_new = np.array(x_new, dtype=int)
    return x_new


def check_collision_line(map_img, p1, p2):
    map_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2GRAY)
    orig_white_pixel = np.sum(map_img == 255)
    cv2.line(map_img, p1, p2, (255), 5)
    new_white_pixel = np.sum(map_img == 255)

    if new_white_pixel > orig_white_pixel:
        return True
    else:
        return False

def draw_RRT_result(map_img, node_coordinates, node_parents, found, RRT_path):
    cv2.circle(map_img, node_coordinates[0], 15, (0, 255, 0), -1)
    if(found):
        cv2.circle(map_img, RRT_path[0], 15, (255, 255, 0), -1)

    for idx, node in enumerate(node_coordinates):
        cv2.line(map_img, node, node_coordinates[node_parents[idx]], (0, 0, 0), 2)
    for node in node_coordinates:
        cv2.circle(map_img, node, 5, (255, 0, 0), -1)

    if(found):
        for idx in range(len(RRT_path) - 1):
            if idx != 0:
                cv2.circle(map_img, RRT_path[idx], 10, (255, 0, 255), -1)
            cv2.line(map_img, RRT_path[idx], RRT_path[idx+1], (0, 0, 255), 3)

    return map_img



if __name__ == "__main__":

    img_origin = cv2.imread("map.png")
    img = img_origin.copy()

    print("Please enter the category name you want to go (refrigerator, rack, cushion, lamp, and cooktop):")
    target_str = "cooktop"#input()

    cv2.putText(img, "Please click the starting position, then press any key to start RRT.", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,0), thickness = 5)
    cv2.imshow("map", img)
    cv2.setMouseCallback('map', click_event)
    cv2.waitKey(0)

    print("Starting point: ", start_point)

    RRT_path, RRT_img = RRT(img_origin, start_point, target_str, RRT_star = USE_RRT_STAR)
    cv2.imshow("map", RRT_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("RRT_path.png", RRT_img)

    RRT_path = np.array(RRT_path)
    np.save("path_info.npy", RRT_path)

    dict = {"target" : target_str}
    with open("setting.json", "w") as f:
        json.dump(dict, f)
