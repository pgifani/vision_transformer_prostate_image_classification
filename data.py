

import os

path = "./"
data_path= "J:/gifani/BreastCancer/"
f = open(path + '/data/data.txt', 'w') 
    


path_org = os.listdir(data_path + 'all/')
print(path_org)
all_lst = []
for d in path_org:
    
    
    lst1 = os.listdir(data_path +  'all/' + d)
    for j in lst1:
        all_lst.append(data_path + 'all/' +  d + '/' + j)
        f.write(data_path + 'all/' +  d + '/' + j +'\n')

f.close()

