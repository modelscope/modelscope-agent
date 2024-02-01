import os
import zipfile

import nltk

current_dir_abs_path = os.path.dirname(os.path.abspath(__file__))


def install_nltk_data():
    user_current_working_dir = os.getcwd()
    nltk_working_dir = os.path.join(user_current_working_dir, 'tmp',
                                    'nltk_data')
    nltk.data.path.append(nltk_working_dir)

    # get nltk artifacts
    nltk_artifacts_dir = os.path.join(current_dir_abs_path, 'nltk')
    punkt_zip_file = os.path.join(nltk_artifacts_dir, 'punkt.zip')
    averaged_perceptron_tagger_zip_file = os.path.join(
        nltk_artifacts_dir, 'averaged_perceptron_tagger.zip')

    # get target dir
    punkt_target_dir = os.path.join(nltk_working_dir, 'tokenizers')

    averaged_target_dir = os.path.join(nltk_working_dir, 'taggers')

    if not os.path.exists(os.path.join(punkt_target_dir, 'punkt')):
        os.makedirs(os.path.join(punkt_target_dir, 'punkt'), exist_ok=True)
        with zipfile.ZipFile(punkt_zip_file, 'r') as zip_ref:
            zip_ref.extractall(punkt_target_dir)

    if not os.path.exists(
            os.path.join(averaged_target_dir, 'averaged_perceptron_tagger')):
        os.makedirs(
            os.path.join(averaged_target_dir, 'averaged_perceptron_tagger'),
            exist_ok=True)
        with zipfile.ZipFile(averaged_perceptron_tagger_zip_file,
                             'r') as zip_ref:
            zip_ref.extractall(averaged_target_dir)
