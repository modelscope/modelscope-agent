from __future__ import annotations
import os
import re
import traceback
import uuid

import gradio as gr
import json


def generate_story(user_input, num_scene, max_scene, agent):
    """
    产生story
    """
    # chatbot.append((user_input, '处理中'))
    # user_input += f'分成{num_scene}幕生成。请在每一幕的开始用1,2,3...标记'
    user_input += f'分成{num_scene}段写出'


    def get_response(prompt):

        response = ''

        for frame in agent.stream_run(prompt, remote=False):
            is_final = frame.get("frame_is_final")
            llm_result = frame.get("llm_text", "")
            exec_result = frame.get('exec_result', '') 

            if len(exec_result) != 0:
                # llm_result
                frame_text = f'\n\n<|startofexec|>{exec_result}<|endofexec|>\n'
            else:
                # action_exec_result
                frame_text = llm_result
            response = f'{response}{frame_text}'
        return response

    story = get_response(user_input) 
    print('---------------------')
    print(story)

    scene_prompt = get_scene_prompt(story)

    print('---------------------')
    print(scene_prompt)

    story_response = ''
    imgs = [None] * max_scene
    texts = [None] * max_scene
    for i, scene_i in enumerate(scene_prompt):

        if i >= max_scene:
            break
        i_prompt = f'请生成第{i+1}段的图片'
        
        img_i = get_response(i_prompt)

        # 匹配图片的markdown
        match_image = re.search(r'!\[IMAGEGEN\]\((.*?)\)', img_i)

        if match_image:
            img_i = match_image.group(1)
        else:
            img_i = None

        img_i_md = f'![IMAGEGEN]("/file={img_i}")'
        story_response += f'\n\n{scene_i}\n{img_i_md}\n'

        imgs[i] = img_i
        texts[i] = scene_i

        # chatbot.append((None, scene_i))
        # chatbot.append((None, img_i_md))

        print(story_response)
        # yield chatbot, *imgs, *texts
        yield *imgs, *texts


    # chatbot[-1] = (chatbot[-1][0], story_response)

    yield *imgs, *texts
    

def get_scene_prompt(story_response):
    """
    获取场景的prompt
    """

    scene_prompt = []

    lines = story_response.strip().split('\n')  # Split the string into lines

    current_scene = None

    for line in lines:
        line = line.strip()
        match_scene = re.match(r'^\d+\.(.+)', line)
        if match_scene:
            if current_scene is not None:
                scene_prompt.append(current_scene)
            current_scene = match_scene.group(1)
        elif current_scene is not None:
            current_scene += line

    if current_scene is not None:
        scene_prompt.append(current_scene)


    # TODO:转写

    return scene_prompt
