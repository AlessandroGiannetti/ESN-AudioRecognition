import os

def extractFiles(dir):
    l = os.listdir(dir)
    d = {}
    # subd = {}
    l.remove('.DS_Store')
    for elem in l:
        #print(elem)
        sublist = []
        for directory in os.listdir('Utah Audio Data/'+ elem):
            sublist.append(directory)
            #print(sublist)
        if elem == '[New] Concrete Mixer 3':
            sublist.remove('.DS_Store')
        d[elem] = sublist

# see how to do recursive directories scan

    for elem in l:
        for directories in d[elem]:
            print(directories)
            if directories != '.DS_Store':
                h = os.listdir('Utah Audio Data/' + elem + "/" + directories)
                print(h)


    # print(d)
    # for d[elem] in d:
    #     for subdirectories in d[elem]:
    #         helplist = os.listdir('Utah Audio Data/'+ d[elem] + "/" + subdirectories)
    #         print(helplist)
    #         subd[d[elem]] = helplist
    #         print(subd)
    # return l

if __name__ == '__main__':
    list = extractFiles('Utah Audio Data')
    #print(list)
    #print(list[0])

