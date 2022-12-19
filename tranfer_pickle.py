
import os, glob
from shutil import copy
def tranfer():
    basedir = r'C:\Users\lucasdevries\surfdrive\Projects\ppinn\wandb'
    folders = os.listdir(basedir)
    newdir = r'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\sppinn_results'
    runids = [
        '05omku2s',
        'rxc5bxn8',
        'kiwvbqhs',
        '5zaj8d7d',
        'w8ceho0t',
        'leelcl3r',
        'at88dr6j',
        'rfmllymw',
        '8zjjlppc',
        'gl03eybi',
        'oibtb7tr',
        '99fisvx1',
        '1ldk3ynd',
        'taxd8z5d',
        'syz4keng',
        'htbqc6ri',
        'zonaq6se',
        'u0618ilx',
        'y6gmuthf',
        'kovzigfx',
        'vdjf9y1a',
        '4ui5zus4',
        'fceanbd1',
        '45i9o7qi',
        'jtr1n7h1',
        'wqc3o9dh',
        '0vmzbjdr',
        'd78jtj8j',
        'p41cmi0s',
        '47fh5w5u',
        "l2lnqf8r", "hifxpzb8", "a5hy1dmt", "seyrbgyu", "9nmj6srq"
    ]

    for id in runids:
        folder = [fol for fol in folders if id in fol][0]
        file = os.path.join(basedir, folder, 'files/*.pickle')
        file = glob.glob(file)[0]
        name = file.split(os.sep)[-1]
        copy(file, os.path.join(newdir, name))

if __name__ == '__main__':
    tranfer()