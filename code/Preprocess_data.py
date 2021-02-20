import gzip
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from skimage import io




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


location = os.getcwd()+"/product_images"
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


open(os.getcwd()+"/product_old_id_product_new_id_map.txt","w").write(str(product_old_id_product_new_id_map))




# creating training and test data

interaction_data = eval(open(os.getcwd()+"/interaction_data.txt","r").readlines()[0])
product_old_id_product_new_id_map = eval(open(os.getcwd()+"/product_old_id_product_new_id_map.txt","r").readlines()[0])


with open(os.getcwd()+"/preprocessed_train_data.txt","w") as train_file, open(os.getcwd()+"/preprocessed_test_data.txt","w") as test_file:

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

        
        
        
main_dir = os.getcwd()+"/product_images"
for filename in os.listdir(main_dir):
    dirname = main_dir+"/"+(filename.split("_")[0])
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    statement = shutil.move(main_dir+"/"+filename,dirname)
    print("\r"+statement.strip(),end="")

print("\rCompleted subdirectorizing")





for ele in os.listdir("product_images"):
    for file in os.listdir("product_images/"+ele):
        if os.path.isfile("product_images/"+ele+"/"+file):
            try:
                a = io.imread("product_images/"+ele+"/"+file)
            except:
                os.remove("product_images/"+ele+"/"+file)
    
    i = 0
    all_f = sorted(os.listdir("product_images/"+ele))
    for file in all_f:
        if os.path.isfile("product_images/"+ele+"/"+file):
            itemid, pic_no = file.split(".")[0].split("_")
            if int(pic_no)!=i:
                os.rename("product_images/"+ele+"/"+file,"product_images/"+ele+"/"+itemid+f"_{i:02d}.png")
            i +=1
            
            
            
            
train_data = pd.read_csv("preprocessed_train_data.txt",delimiter="\t",names=["user_id","item_id","rating","review"])
item_id__image_count__map = {int(item_id.replace("P","")):len(glob.glob1(f"product_images/{item_id}/",f"{item_id}*.png")) for item_id in os.listdir("product_images")}
to_be_eliminated = [key for key,value in item_id__image_count__map.items() if value==0]
print(f"Count of items to be eliminated : {len(to_be_eliminated)}")
print(f"Items to be eliminated {to_be_eliminated}")
print(f"Number of removed data points : {len(train_data[train_data['item_id'].isin(to_be_eliminated)].index)}")
train_data = train_data.drop(index=train_data[train_data["item_id"].isin(to_be_eliminated)].index).reset_index(drop=True)
os.remove("preprocessed_train_data.txt")
train_data.to_csv("preprocessed_train_data.txt",sep="\t",index=False,header=False)
test_data = pd.read_csv("preprocessed_test_data.txt",delimiter="\t",names=["user_id","item_id","rating","review"])
test_data = test_data.drop(index=test_data[test_data["item_id"].isin(to_be_eliminated)].index).reset_index(drop=True)
os.remove("preprocessed_test_data.txt")
test_data.to_csv("preprocessed_test_data.txt",sep="\t",index=False,header=False)
