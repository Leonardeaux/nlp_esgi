import pandas as pd
import re

def word_in_array(word, array):
    for w in array:
        if word in w:
            return True

    return False


def get_index(title, target):
    splited_title = re.split(r"[ ']", title)
    index_array = [0] * len(splited_title)

    for i, word in enumerate(splited_title):
        if word_in_array(word, target):
            index_array[i] = 1
    
    return index_array