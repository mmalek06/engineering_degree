import os
import shutil

import pandas as pd

from constants.categories import Categories


def create_dir_if_not_exists(check_path: str):
    if not os.path.exists(check_path):
        try:
            os.makedirs(check_path)
        except OSError as error:
            print(f'Creation of directory {check_path} failed. Error: {error}')


def create_category_folders(type_path: str) -> None:
    for category in Categories:
        name = category.value
        category_path = os.path.join(type_path, name)

        create_dir_if_not_exists(category_path)


def categorize_files(
        frame: pd.DataFrame,
        type_path: str,
        images_path: str,
        category_col_name: str,
        image_col_name: str,
        use_file_extension: bool = True) -> None:
    for _, row in frame.iterrows():
        category = row[category_col_name]
        image_name = row[image_col_name]
        category_path = os.path.join(type_path, category)
        img = f'{image_name}.jpg' if use_file_extension else image_name
        orig_image_path = os.path.join(images_path, img)
        new_image_path = os.path.join(category_path, img)

        create_dir_if_not_exists(category_path)
        shutil.copy(orig_image_path, new_image_path)

    # make sure all category folders have been created
    create_category_folders(type_path)
