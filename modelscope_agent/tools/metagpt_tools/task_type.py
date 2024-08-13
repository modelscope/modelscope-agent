# this code is originally from https://github.com/geekan/MetaGPT and modified by modelscope-agent
from enum import Enum

from pydantic import BaseModel

# Prompt for taking on "eda" tasks
EDA_PROMPT = """
The current task is about exploratory data analysis, please note the following:
- Distinguish column types with `select_dtypes` for tailored analysis and visualization.
- Remember to `import numpy as np` before using Numpy functions.
- don't plot any columns
- don't calculate correlations
- Avoid showing all data, always use df.head() to show the first 5 rows.

"""

# Prompt for taking on "data_preprocess" tasks
DATA_PREPROCESS_PROMPT = """
The current task is about data preprocessing, please note the following:
- make sure train and test data MUST have the same columns except for the label column.
- Remove ID columns if exist.
- Handle missing values with suitable strategies.
- Avoid any change to label column, such as standardization, etc.
- Each step do data preprocessing to train, must do same for test separately at the same time.
- Always copy the DataFrame before processing it and use the copy to process.
- Avoid writing processed data to files.
"""

# Prompt for taking on "feature_engineering" tasks
FEATURE_ENGINEERING_PROMPT = """
The current task is about feature engineering. when performing it, please adhere to the following principles:
- Avoid creating redundant or excessively numerous features in one step.
- Each feature engineering operation performed on the train set must also applies to the \
test separately at the same time.
- Use the data from previous task result if exist, do not mock or reload data yourself.
- Always copy the DataFrame before processing it and use the copy to process.
- Use Label Encoding for non-numeric columns, avoid One-Hot Encoding.
"""

# Prompt for taking on "model_train" tasks
MODEL_TRAIN_PROMPT = """
The current task is about training a model, please ensure high performance:
- If non-numeric columns exist, perform label encode together with all steps.
- Try to use the best model such as RandomForest, XGBoost, if these models casued timeout error or \
other issues, please try other models, don't stick to the same model.
- If the model caused timeout error, please don't use this model again.
- Use the data from previous task result directly, do not mock or reload data yourself.
- Never save the model to a file.
"""

# Prompt for taking on "model_evaluate" tasks
MODEL_EVALUATE_PROMPT = """
The current task is about evaluating a model, please note the following:
- Ensure that the evaluated data is same processed as the training data.
- Use trained model from previous task result directly, do not mock or reload model yourself.
"""
OCR_PROMPT = """
The current task is about OCR, please note the following:
- you can follow the following code to get the OCR result:
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr('/path/to/the/pic', cls=True) # please replace the path with the real path
"""


class TaskTypeDef(BaseModel):
    name: str
    desc: str = ''
    guidance: str = ''


class TaskType(Enum):
    """By identifying specific types of tasks, we can inject human priors (guidance) to help task solving"""

    EDA = TaskTypeDef(
        name='eda',
        desc='For performing exploratory data analysis',
        guidance=EDA_PROMPT,
    )
    DATA_PREPROCESS = TaskTypeDef(
        name='data preprocessing',
        desc=
        'For preprocessing dataset in a data analysis or machine learning task ONLY,'
        "general data operation doesn't fall into this type",
        guidance=DATA_PREPROCESS_PROMPT,
    )
    FEATURE_ENGINEERING = TaskTypeDef(
        name='feature engineering',
        desc='Only for creating new columns for input data.',
        guidance=FEATURE_ENGINEERING_PROMPT,
    )
    MODEL_TRAIN = TaskTypeDef(
        name='model train',
        desc='Only for training model.',
        guidance=MODEL_TRAIN_PROMPT,
    )
    MODEL_EVALUATE = TaskTypeDef(
        name='model evaluate',
        desc='Only for evaluating model.',
        guidance=MODEL_EVALUATE_PROMPT,
    )
    OCR = TaskTypeDef(
        name='ocr', desc='For performing OCR tasks', guidance=OCR_PROMPT)
    OTHER = TaskTypeDef(
        name='other', desc='Any tasks not in the defined categories')

    @property
    def type_name(self):
        return self.value.name

    @classmethod
    def get_type(cls, type_name):
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None
