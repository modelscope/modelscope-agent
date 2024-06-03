# flake8: noqa

import base64
import math
import os

import cv2
import numpy as np
import torch
from modelscope_agent.utils.retry import retry
from PIL import Image, ImageDraw


def crop_image(img, position):

    def distance(x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    position = position.tolist()
    for i in range(4):
        for j in range(i + 1, 4):
            if (position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4, 2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2,
                         (y2 + y3) / 2)
    img_height = distance((x1 + x2) / 2, (y1 + y2) / 2, (x4 + x3) / 2,
                          (y4 + y3) / 2)

    corners_trans = np.zeros((4, 2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform,
                              (int(img_width), int(img_height)))
    return dst


def calculate_size(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea
    iou = interArea / unionArea

    return iou


def in_box(box, target):
    if (box[0] > target[0]) and (box[1] > target[1]) and (
            box[2] < target[2]) and (box[3] < target[3]):
        return True
    else:
        return False


def crop_for_clip(image, box, i, position):
    image = Image.open(image)
    w, h = image.size
    if position == 'left':
        bound = [0, 0, w / 2, h]
    elif position == 'right':
        bound = [w / 2, 0, w, h]
    elif position == 'top':
        bound = [0, 0, w, h / 2]
    elif position == 'bottom':
        bound = [0, h / 2, w, h]
    elif position == 'top left':
        bound = [0, 0, w / 2, h / 2]
    elif position == 'top right':
        bound = [w / 2, 0, w, h / 2]
    elif position == 'bottom left':
        bound = [0, h / 2, w / 2, h]
    elif position == 'bottom right':
        bound = [w / 2, h / 2, w, h]
    else:
        bound = [0, 0, w, h]

    if in_box(box, bound):
        cropped_image = image.crop(box)
        cropped_image.save(f'./temp/{i}.jpg')
        return True
    else:
        return False


def clip_for_icon(clip_model, clip_preprocess, images, prompt):
    image_features = []
    for image_file in images:
        image = clip_preprocess(Image.open(image_file)).unsqueeze(0).to(
            next(clip_model.parameters()).device)
        image_feature = clip_model.encode_image(image)
        image_features.append(image_feature)
    image_features = torch.cat(image_features)

    import clip
    text = clip.tokenize([prompt]).to(next(clip_model.parameters()).device)
    text_features = clip_model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features
                  @ text_features.T).softmax(dim=0).squeeze(0)
    _, max_pos = torch.max(similarity, dim=0)
    pos = max_pos.item()

    return pos


def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points


def longest_common_substring_length(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def ocr(image_path, ocr_detection, ocr_recognition):
    text_data = []
    coordinate = []

    image_full = cv2.imread(image_path)
    det_result = ocr_detection(image_full)
    det_result = det_result['polygons']
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i])
        image_crop = crop_image(image_full, pts)

        try:
            result = ocr_recognition(image_crop)['text'][0]
        except Exception as e:
            print(e)
            continue

        box = [int(e) for e in list(pts.reshape(-1))]
        box = [box[0], box[1], box[4], box[5]]

        text_data.append(result)
        coordinate.append(box)

    else:
        return text_data, coordinate


def remove_boxes(boxes_filt, size, iou_threshold=0.5):
    boxes_to_remove = set()

    for i in range(len(boxes_filt)):
        if calculate_size(boxes_filt[i]) > 0.05 * size[0] * size[1]:
            boxes_to_remove.add(i)
        for j in range(len(boxes_filt)):
            if calculate_size(boxes_filt[j]) > 0.05 * size[0] * size[1]:
                boxes_to_remove.add(j)
            if i == j:
                continue
            if i in boxes_to_remove or j in boxes_to_remove:
                continue
            iou = calculate_iou(boxes_filt[i], boxes_filt[j])
            if iou >= iou_threshold:
                boxes_to_remove.add(j)

    boxes_filt = [
        box for idx, box in enumerate(boxes_filt) if idx not in boxes_to_remove
    ]

    return boxes_filt


def det(input_image_path,
        caption,
        groundingdino_model,
        box_threshold=0.07,
        text_threshold=0.5):
    image = Image.open(input_image_path)
    size = image.size

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    inputs = {
        'IMAGE_PATH': input_image_path,
        'TEXT_PROMPT': caption,
        'BOX_TRESHOLD': box_threshold,
        'TEXT_TRESHOLD': text_threshold
    }

    result = groundingdino_model(inputs)
    boxes_filt = result['boxes']

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu().int().tolist()
    filtered_boxes = remove_boxes(boxes_filt, size)  # [:9]
    coordinates = []
    for box in filtered_boxes:
        coordinates.append([box[0], box[1], box[2], box[3]])

    return coordinates


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size,
                      coord[0] + point_size, coord[1] + point_size),
                     fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path


def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2 - 10 or y1 >= y2 - 10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f'./temp/{i}.jpg')


@retry(max_retries=5, delay_seconds=0.5)
def generate(image_file, query):

    from dashscope import MultiModalConversation

    local_file_path = f'file://{image_file}'
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': local_file_path
            },
            {
                'text': query
            },
        ]
    }]
    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages,
        api_key=os.getenv('DASHSCOPE_API_KEY'))

    return response.output.choices[0].message.content[0]['text']


async def agenerate(image_path, prompt):
    import asyncio
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, generate, image_path, prompt)
    return result


def merge_text_blocks(text_list, coordinates_list):
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(
        range(len(coordinates_list)),
        key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue

        anchor = i

        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i + 1, num_blocks):
            if merge[j]:
                continue

            if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
            sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
            abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = '\n'.join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

    return merged_text_blocks, merged_coordinates


def merge_text_blocks(text_list, coordinates_list):
    merged_text_blocks = []
    merged_coordinates = []

    sorted_indices = sorted(
        range(len(coordinates_list)),
        key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]

    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks

    for i in range(num_blocks):
        if merge[i]:
            continue

        anchor = i

        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]

        for j in range(i + 1, num_blocks):
            if merge[j]:
                continue

            if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
            sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
            abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
                merge[anchor] = True

        merged_text = '\n'.join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]

        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

    return merged_text_blocks, merged_coordinates


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    img = f'data:image/jpeg;base64,{base64_image}'
    return img
