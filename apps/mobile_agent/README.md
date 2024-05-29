# Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration

## 🔧Getting Started

### Installation
```
pip install -r requirements.txt
```

### Preparation for Connecting Mobile Device
1. Download the [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en).
2. Turn on the ADB debugging switch on your Android phone, it needs to be turned on in the developer options first.
3. Connect your phone to the computer with a data cable and select "Transfer files".
4. Test your ADB environment as follow: ```/path/to/adb devices```. If the connected devices are displayed, the preparation is complete.
5. If you are using a MAC or Linux system, make sure to turn on adb permissions as follow: ```sudo chmod +x /path/to/adb```
6. If you are using Windows system, the path will be ```xx/xx/adb.exe```

### Preparation for Visual Perception Tools

* Download the icon detection model [Grounding DION](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth). Currently you should place it under your   `groundingdino_dir`.

* Download `bert-base-uncased` from [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased/summary) or [Huggingface](https://huggingface.co/google-bert/bert-base-uncased).
* Specify the `bert-base-uncased` model path in `text_encoder_type` of `groundingdino/config/GroundingDINO_SwinT_OGC.py`.


### Run

The related args to run demo include:
* `--adb_path`: The path to debug with your adb.
* `--openai_api_key`: The OpenAI key to call llm.
* `--dashscope_api_key`: The Dashscope key to call qwen-vl.
* `--groundingdino_dir`: The groundingdino path.
* `--instruction`: Your instruction.
