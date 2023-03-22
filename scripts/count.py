import re



f = open("../build/partials.cpp")
txt = f.read()

#import pdb; pdb.set_trace()
temp = re.findall("v\[[0-9]+\]", txt)
big_set = set(temp)
print(len(big_set))
