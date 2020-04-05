import os
import re
import json
import pickle

LINK_PATTERN = r"https?:\/\/(www\.)?([-\w@:%._\+~#=]+\.[\w()]+)\b([-\w()@:%_\+.~#?&\/\/=]*)"

def slice_url(url):
    match = re.search(LINK_PATTERN, url)
    return match.group(1), match.group(2), match.group(3)

def save_json_file(base, name, file):
    if not os.path.exists(base):
        os.makedirs(base)

    open(os.path.join(base, name + ".json"), "w+").write(json.dumps(file))

def save_pickle(path, document):
    with open(path, 'wb') as file:
        pickle.dump(document, file, pickle.HIGHEST_PROTOCOL)
