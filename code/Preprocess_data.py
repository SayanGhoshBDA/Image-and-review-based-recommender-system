import gzip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import os




# function for parsing data
def parse(path):
    g = gzip.open(path, 'r')
    list_info = []
    for l in g:
        list_info.append(eval(str(l.decode()).replace(": true",": True").replace(": false",": False")))
    return list_info




# parsing interaction data
interaction_data = parse("./data/AMAZON_FASHION.json.gz")
open(os.getcwd()+"/data/interaction_data.txt","w").write(str(interaction_data))

# parsing metadata
product_metadata = parse("./data/meta_AMAZON_FASHION.json.gz")
open(os.getcwd()+"/data/product_metadata.txt","w").write(str(product_metadata))




# downloading images

product_old_id_product_new_id_map = {}
product_id = 0
i = 1
n = len(product_metadata)


location = os.getcwd()+"/data/product_images"
if os.path.exists(location):
    os.rmdir(location)
os.mkdir(location,0o777)


for product in product_metadata:
    if "image" in product.keys() and len(product["image"])>0:

        count = 0
        for url in product["image"]:
            try:
                response = requests.get(url)
                open(location+f"/P{product_id:08d}_{count:02d}.png","wb").write(response.content)
                count += 1
            except:
                pass

        if count>0:
            product_old_id_product_new_id_map[product["asin"]] = product_id
            product_id += 1

    print("\r",end="")
    print(f"Image downloading... {float(i)/n * 100 : 3.10f}%",end="")
    i += 1


open(os.getcwd()+"/data/product_old_id_product_new_id_map.txt","w").write(str(product_old_id_product_new_id_map))




# creating training and test data

interaction_data = eval(open(os.getcwd()+"/data/interaction_data.txt","r").readlines()[0])
product_old_id_product_new_id_map = eval(open(os.getcwd()+"/data/product_old_id_product_new_id_map.txt","r").readlines()[0])


with open(os.getcwd()+"/data/preprocessed_train_data.txt","w") as train_file, open(os.getcwd()+"/data/preprocessed_test_data.txt","w") as test_file:

    i = 1
    n = len(interaction_data)
    reviewer_old_id_reviewer_new_id_map = {}
    reviewer_id = 0

    def to_string(string):
        return str(string).replace("\n","")

    for interaction in interaction_data:
        if interaction["asin"] in product_old_id_product_new_id_map.keys() and\
        "reviewText" in interaction.keys() and\
        len(interaction["reviewText"])>0:

            if interaction["reviewerID"] in reviewer_old_id_reviewer_new_id_map.keys():
                to_be_written = "\t".join(map(to_string,[reviewer_old_id_reviewer_new_id_map[interaction["reviewerID"]],product_old_id_product_new_id_map[interaction["asin"]],interaction["overall"],interaction["reviewText"]]))+"\n"
                where_to_keep = np.random.choice([0,1],p=[0.70,0.30])
                if where_to_keep==0:
                    train_file.write(to_be_written)
                else:
                    test_file.write(to_be_written)
            else:
                train_file.write("\t".join(map(to_string,[reviewer_id,product_old_id_product_new_id_map[interaction["asin"]],interaction["overall"],interaction["reviewText"]]))+"\n")
                reviewer_old_id_reviewer_new_id_map[interaction["reviewerID"]] = reviewer_id
                reviewer_id += 1

        print("\r",end="")
        print(f"Data file creation... {float(i)/n * 100 : 3.10f}%",end="")
        i += 1
