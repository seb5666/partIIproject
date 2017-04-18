import sys
from os import listdir
from string import Template
from shutil import copyfile

assert(len(sys.argv) == 3)

directory = sys.argv[1]
sub = sys.argv[2]
print("Dir", directory, 'Sub number', sub)

template = Template('VSD.spb61-sub${sub}.${scan}.${num}.mha')
for file in listdir(directory):
    if "processed" in file:
        id = file[:4]
        scan = int(file[2:4])
        num = 17572 + (scan - 1) * 4
        dest_file = template.substitute(sub = sub, num = num, scan= id)
        source = directory + '/' + file
        dest = directory + '/' + dest_file
        print(source, '->', dest)
        copyfile(source, dest)
