import asyncio
import os
import shutil
import subprocess
import time
from typing import List

from modelscope_agent.utils.logger import agent_logger as logger
from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from .utils import (agenerate, crop, det, draw_coordinates_on_image,
                    encode_image, get_all_files_in_folder, merge_text_blocks,
                    ocr)


class ADBEnvironment:

    def __init__(self, adb_path: str) -> None:
        self.adb_path = adb_path

        # ocr pipeline
        self.ocr_detection = pipeline(
            Tasks.ocr_detection,
            model='damo/cv_resnet18_ocr-detection-line-level_damo')
        self.ocr_recognition = pipeline(
            Tasks.ocr_recognition,
            model='damo/cv_convnextTiny_ocr-recognition-document_damo')

        # groundingdino model
        model_dir = snapshot_download(
            'AI-ModelScope/GroundingDINO', revision='v1.0.0')
        self.groundingdino = pipeline('grounding-dino-task', model=model_dir)

        self.temp_dir = 'temp'

        self.screenshot_dir = 'screenshot'
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        if not os.path.exists(self.screenshot_dir):
            os.mkdir(self.screenshot_dir)

        self.screenshot_file = os.path.join(self.screenshot_dir,
                                            'screenshot.jpg')
        self.last_screenshot_file = os.path.join(self.screenshot_dir,
                                                 'last_screenshot.jpg')

    def observe(self):
        perception_infos, width, height, keyboard = asyncio.run(
            self.get_perception_infos(self.screenshot_file))
        screenshot_file = encode_image(self.screenshot_file)
        return perception_infos, width, height, keyboard, screenshot_file

    def act(self, action):
        if 'Open app' in action:
            app_name = action.split('(')[-1].split(')')[0]
            text, coordinate = ocr(self.screenshot_file, self.ocr_detection,
                                   self.ocr_recognition)
            # tap_coordinate = [0, 0]
            for ti in range(len(text)):
                if app_name == text[ti]:
                    name_coordinate = [
                        int((coordinate[ti][0] + coordinate[ti][2]) / 2),
                        int((coordinate[ti][1] + coordinate[ti][3]) / 2)
                    ]
                    self.tap(name_coordinate[0], name_coordinate[1]
                             - int(coordinate[ti][3] - coordinate[ti][1]))  #

        elif 'Tap' in action:
            coordinate = action.split('(')[-1].split(')')[0].split(', ')
            x, y = int(coordinate[0]), int(coordinate[1])
            self.tap(x, y)

        elif 'Swipe' in action:
            coordinate1 = action.split('Swipe (')[-1].split('), (')[0].split(
                ', ')
            coordinate2 = action.split('), (')[-1].split(')')[0].split(', ')
            x1, y1 = int(coordinate1[0]), int(coordinate1[1])
            x2, y2 = int(coordinate2[0]), int(coordinate2[1])
            self.slide(x1, y1, x2, y2)

        elif 'Type' in action:
            if '(text)' not in action:
                text = action.split('(')[-1].split(')')[0]
            else:
                text = action.split(" \"")[-1].split("\"")[0]
            self.type(text)

        elif 'Back' in action:
            self.back()

        elif 'Home' in action:
            self.home()

        elif 'Stop' in action:
            return True
        time.sleep(5)
        if os.path.exists(self.last_screenshot_file):
            os.remove(self.last_screenshot_file)
        os.rename(self.screenshot_file, self.last_screenshot_file)
        return False

    async def get_perception_infos(self, screenshot_file):

        logger.info('Start getting perception infos')
        self.get_screenshot()

        width, height = Image.open(screenshot_file).size
        logger.info('Start use OCR get text and coordinates')
        text, coordinates = ocr(screenshot_file, self.ocr_detection,
                                self.ocr_recognition)
        text, coordinates = merge_text_blocks(text, coordinates)
        logger.info('End use OCR get text and coordinates')

        center_list = [[(coordinate[0] + coordinate[2]) / 2,
                        (coordinate[1] + coordinate[3]) / 2]
                       for coordinate in coordinates]
        draw_coordinates_on_image(screenshot_file, center_list)

        perception_infos = []
        for i in range(len(coordinates)):
            perception_info = {
                'text': 'text: ' + text[i],
                'coordinates': coordinates[i]
            }
            perception_infos.append(perception_info)

        logger.info('Start use groundino to detect icons')
        coordinates = det(screenshot_file, 'icon', self.groundingdino)
        logger.info('End use groundino to detect icons')

        for i in range(len(coordinates)):
            perception_info = {'text': 'icon', 'coordinates': coordinates[i]}
            perception_infos.append(perception_info)

        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            if perception_infos[i]['text'] == 'icon':
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)

        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i])

        images = get_all_files_in_folder(self.temp_dir)

        logger.info('Start use qwen-vl to describe icons')
        if len(images) > 0:
            images = sorted(
                images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [
                int(image.split('/')[-1].split('.')[0]) for image in images
            ]
            icon_map = {}
            # Please describe this icon.
            tasks = []
            idx_arr = []
            prompt = 'This image is an icon from a phone screen. Please describe the color and shape of this icon.'
            for i in range(len(images)):
                image_path = os.path.join(self.temp_dir, images[i])
                icon_width, icon_height = Image.open(image_path).size
                if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                    des = 'None'
                    icon_map[i + 1] = des
                else:
                    task = agenerate(image_path, prompt)
                    idx_arr.append(i)
                    tasks.append(task)

            descriptions = await asyncio.gather(*tasks)
            for i, j in zip(idx_arr, range(len(descriptions))):
                icon_map[i + 1] = descriptions[j]

            for i, j in zip(image_id, range(1, len(image_id) + 1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] = 'icon: ' + icon_map[j]

        logger.info('End use qwen-vl to describe icons')
        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [
                int((perception_infos[i]['coordinates'][0]
                     + perception_infos[i]['coordinates'][2]) / 2),
                int((perception_infos[i]['coordinates'][1]
                     + perception_infos[i]['coordinates'][3]) / 2)
            ]

        shutil.rmtree(self.temp_dir)
        os.mkdir(self.temp_dir)

        keyboard = False
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < 0.95 * height:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break
        logger.info('Finish getting perception infos')
        return perception_infos, width, height, keyboard

    # ADB related functions
    def get_size(self):
        command = self.adb_path + ' shell wm size'
        result = subprocess.run(
            command, capture_output=True, text=True, shell=True)
        resolution_line = result.stdout.strip().split('\n')[-1]
        width, height = map(int, resolution_line.split(' ')[-1].split('x'))
        return width, height

    def get_xml(self):
        adb_path = self.adb_path
        process = subprocess.Popen([adb_path, 'shell', 'uiautomator', 'dump'],
                                   stdout=subprocess.PIPE)
        process.communicate()
        subprocess.run([
            adb_path, 'pull', '/sdcard/window_dump.xml',
            './xml/window_dump.xml'
        ])

    def take_screenshots(self, num_screenshots, output_folder, crop_y_start,
                         crop_y_end, slide_y_start, slide_y_end):
        adb_path = self.adb_path
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in range(num_screenshots):
            command = adb_path + f' shell rm /sdcard/screenshot{i}.png'
            subprocess.run(command, capture_output=True, text=True, shell=True)
            command = adb_path + f' shell screencap -p /sdcard/screenshot{i}.png'
            subprocess.run(command, capture_output=True, text=True, shell=True)
            command = adb_path + f' pull /sdcard/screenshot{i}.png {output_folder}'
            subprocess.run(command, capture_output=True, text=True, shell=True)
            image = Image.open(f'{output_folder}/screenshot{i}.png')
            cropped_image = image.crop(
                (0, crop_y_start, image.width, crop_y_end))
            cropped_image.save(f'{output_folder}/screenshot{i}.png')
            subprocess.run([
                adb_path, 'shell', 'input', 'swipe', '500',
                str(slide_y_start), '500',
                str(slide_y_end)
            ])

    def get_screenshot(self):
        adb_path = self.adb_path
        command = adb_path + ' shell rm /sdcard/screenshot.png'
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = adb_path + ' shell screencap -p /sdcard/screenshot.png'
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = adb_path + ' pull /sdcard/screenshot.png ./screenshot'
        subprocess.run(command, capture_output=True, text=True, shell=True)
        image_path = './screenshot/screenshot.png'
        save_path = './screenshot/screenshot.jpg'
        image = Image.open(image_path)
        image.convert('RGB').save(save_path, 'JPEG')
        os.remove(image_path)

    def get_keyboard(self):
        adb_path = self.adb_path
        command = adb_path + ' shell dumpsys input_method'
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            encoding='utf-8')
        output = process.stdout.strip()
        for line in output.split('\n'):
            if 'mInputShown' in line:
                if 'mInputShown=true' in line:

                    for line in output.split('\n'):
                        if 'hintText' in line:
                            hintText = line.split('hintText=')[-1].split(
                                ' label')[0]
                            break

                    return True, hintText
                elif 'mInputShown=false' in line:
                    return False, None

    def tap(self, x, y):
        command = self.adb_path + f' shell input tap {x} {y}'
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def type(self, text):
        adb_path = self.adb_path
        text = text.replace('\\n', '_').replace('\n', '_')
        for char in text:
            if char == ' ':
                command = adb_path + ' shell input text %s'
            elif char == '_':
                command = adb_path + ' shell input keyevent 66'
            elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
                command = adb_path + f' shell input text {char}'
            elif char in '-.,!?@\'°/:;()':
                command = adb_path + f" shell input text \"{char}\""
            else:
                command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)

    def slide(self, x1, y1, x2, y2):
        command = self.adb_path + f' shell input swipe {x1} {y1} {x2} {y2} 500'
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def back(self):
        command = self.adb_path + ' shell input keyevent 4'
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def home(self):
        command = self.adb_path + ' shell am start -a android.intent.action.MAIN -c android.intent.category.HOME'
        subprocess.run(command, capture_output=True, text=True, shell=True)
