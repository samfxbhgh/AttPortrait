import json
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from datasets import load_dataset,concatenate_datasets


class Face_Att_datataset(Dataset):
    def __init__(self, file_path, image_dir, embed_dir, default_att_names=None,transform=None,data_type='train'):

        self.file_path = file_path
        self.image_dir = image_dir
        self.embed_dir = embed_dir
        self.transform = transform
        self.default_att_names=default_att_names
        ds=[]
        for path in file_path:
            ds.append(load_dataset('json',data_files=path,split=data_type))

        self.data=concatenate_datasets(ds)
        self.has_attri=False
        if 'attributes' in self.data[0]:
            self.has_attri=True
            self.ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
            'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
            'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
            'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
            'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
            'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
            'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
            'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
            'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
            'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
            'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

            self.ID_ATT = {v: k for k, v in self.ATT_ID.items()}            
            attri_index=[]
            if default_att_names:
                for attri in default_att_names:
                    attri_index.append(self.ATT_ID[attri])#index[]
            self.attri_index = attri_index

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        l = len(self.data)
        image_name = sample["image_name"]
        if self.has_attri:
            edit_all_attributes = (torch.tensor(self.data[l-1-idx]["attributes"])+1)/2
            all_attributes = (torch.tensor(sample['attributes']) + 1) / 2
            if len(self.attri_index)>0:
                attributes=all_attributes[self.attri_index]
                edit_attributes=edit_all_attributes[self.attri_index]


        embed_name=image_name+'.embedding.npy'
        
        # load images
        image_path = os.path.join(self.image_dir, image_name)
        embedding_path = os.path.join(self.embed_dir, embed_name)
        #image = Image.open(image_path).convert("RGB")  # convert to RGB


        try:
            image = Image.open(image_path).convert("RGB")  # Convert to RGB
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Provide a default black image if loading fails
            image = Image.new('RGB', (572, 572), (0, 0, 0))  # Example size and default color

        try:
            embedding = np.load(embedding_path)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            embedding = embedding / torch.norm(embedding)  # Normalize embedding
        except Exception as e:
            print(f"Error loading embedding {embedding_path}: {e}")
            # Provide a default embedding (zero vector)
            embedding = torch.zeros(512)  # Set embedding_dim based on your model's requirement

        #image transform
        if self.transform:
            image = self.transform(image)

        if self.has_attri:
            return {"image": image, "attributes": attributes, "edit_attributes":edit_attributes, "embedding": embedding, 'id':image_name}
        else:
            return {"image": image, "embedding": embedding,'id':image_name}
    
def imshow(imgs, immediate=False, **make_grid_kwargs):
    if isinstance(imgs, list) or imgs.dim() == 4:
        imgs = [img[None, ...] if img.dim() == 2 else img for img in imgs]
        imgs = torchvision.utils.make_grid(imgs, **make_grid_kwargs)
    else:
        if imgs.dim() == 2:
            imgs = imgs.unsqueeze(0).repeat(3, 1, 1)
        elif imgs.dim() == 3 and imgs.size(0) == 1:
            imgs = imgs.repeat(3, 1, 1)
    imgs = F.to_pil_image(imgs.detach())
    plt.imshow(np.asarray(imgs))
    imgs.save('output_center.jpg')
    if immediate:
        plt.show()



def bulid_dataset(data_type,transform,default_att_names,args):
    dataset_name=args.dataset_name
    out_dir=args.output_dir
    assert data_type in ['train','val'], f"Not support {data_type} type" 

    if data_type=='train':
        path=['assets/examples_data/metadata.jsonl']

       
    img_dir=f'assets/examples_data/images'
    embed_dir=f'assets/examples_data/embeddings'
    
    dataset=Face_Att_datataset(file_path=path, 
                            image_dir=img_dir,
                            embed_dir=embed_dir,
                            transform=transform,
                            default_att_names=default_att_names,
                    )
    return dataset