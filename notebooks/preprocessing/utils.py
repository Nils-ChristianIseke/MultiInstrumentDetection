import random
import datetime
import os
import json
import torch
import numpy as np
from PIL import Image
from natsort import natsorted
import yaml


def get_sub_dirs(path, sort=True, paths=True):
    """ Returns all the sub-directories in the given path as a list
    If paths flag is set, returns the whole path, else returns just the names
    """
    sub_things = os.listdir(path)  # thing can be a folder or a file
    if sort: sub_things = natsorted(sub_things)
    sub_paths = [os.path.join(path, thing) for thing in sub_things]

    sub_dir_paths = [sub_path for sub_path in sub_paths if os.path.isdir(sub_path)]  # choose only sub-dirs
    sub_dir_names = [os.path.basename(sub_dir_path) for sub_dir_path in sub_dir_paths]
    return sub_dir_paths if paths else sub_dir_names


def print_elements_of_list(list):
    """Prints each element of list in a newline"""
    [print(element) for element in list]


def check_and_create_folder(path):
    """Check if a folder in given path exists, if not then create it"""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else: return False


def write_list_to_text_file(save_path, text_list, verbose=True):
    """
    Function to write a list to a text file.
    Each element of the list is written to a new line.
    Note: Existing text in the file will be overwritten!
    :param save_path: Path to save-should be complete with .txt extension)
    :param text_list: List of text-each elem of list written in new line)
    :param verbose: If true, prints success message to console
    :return: No return, writes file to disk and prints a success message
    """
    with open(save_path, 'w+') as write_file:
        for text in text_list:
            if isinstance(text, str):
                write_file.write(text + '\n')
            else:
                write_file.write(str(text) + '\n')
        write_file.close()
    if verbose: print("Text file successfully written to disk at {}".format(save_path))


def read_lines_from_text_file(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

