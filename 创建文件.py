import os

i = 0
for package,_,filenames in os.walk(r"E:\\semantic_segmentation\\VOC\\benchmark_RELEASE"):
    i = i+1
    if ".git" in package or i == 1:
        continue
    # print(package.split("E\\")[1])
    path = os.path.join(r"E:\\semantic_segmentation\\VOC\\new_Data",package.split("E\\")[1])

    # make_new_dir = path.split("\\")[-1]
    # root = "E:"
    # for root_ in path.split("\\")[1:-1]:
    #     root = root+"\\"+ root_

    os.makedirs(path,exist_ok=True)
    if "cls"  in package.split("E\\")[1] or "img"  in package.split("E\\")[1] or "inst"  in package.split("E\\")[1]:
        continue

    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'w') as file:
            file.write("Write in line format here {} Specific serial number picture".format(filename))





