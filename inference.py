# ------------------------------------------------------------------------------
# modified from https://github.com/foivospar/Arc2Face
# Last Modified: 2025-06-26
# Author: Zhang Yuhao
# ------------------------------------------------------------------------------

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from attportrait import CLIPTextModelWrapper,hack_unet_attn_layers, ATTAttnProcessor, project_face_embs
import os
from safetensors.torch import load_file
# Arc2Face is built upon SD1.5
# The repo below can be used instead of the now deprecated 'runwayml/stable-diffusion-v1-5'
base_model = 'runwayml/stable-diffusion-v1-5'

id_text_encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)
att_text_encoder = CLIPTextModelWrapper.from_pretrained(
        base_model, subfolder="text_encoder", revision=torch.float16
)
unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

default_att_names = ['Bald','Young','Male', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                'Mouth_Slightly_Open', 'No_Beard', 'Pale_Skin']

unet.set_default_attn_processor()
hack_unet_attn_layers(unet, ATTAttnProcessor)

# Replace here with ModelCheckpoint
file_path=os.path.join('models/checkpoint','diffusion_pytorch_model.safetensors')
unet_weights = load_file(file_path)
unet.load_state_dict(unet_weights,strict=True)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=id_text_encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')
att_text_encoder=att_text_encoder.to('cuda')
generator = torch.Generator(device=pipeline.device)
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img = np.array(Image.open('assets/KevinSpacey.jpg'))[:,:,::-1]

faces = app.get(img)
faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

#replace here with the attribute label that you want to generate
atts=[0,1,1,0,0,0,0,0,0,0,0,0]


prompt='who is '
for i in range(len(atts)):
    if i==0:
        if atts[i]==1:
            prompt=prompt+default_att_names[i]+', '
    elif i==1:
        if atts[i]==1:
            prompt=prompt+'Young, '
        else:
            prompt=prompt+'Old, '
    elif i==2:
        if atts[i]==1:
            prompt=prompt+'Male, '
        else:
            prompt=prompt+'Female, '
    else:
        if atts[i]==1:
            prompt=prompt+'with '+default_att_names[i]+' '
prompt=prompt

input_ids_atts = pipeline.tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(pipeline.device)


att_emb = att_text_encoder(
    input_ids=input_ids_atts,
    return_dict=False
)[0]


num_images = 4
neg_embeddings=torch.zeros_like(att_emb)
bs_embed, seq_len, _ = att_emb.shape

att_emb = att_emb.repeat(1, num_images, 1)
att_emb = att_emb.view(bs_embed *num_images, seq_len, -1)

neg_embeddings = neg_embeddings.repeat(1, num_images, 1)
neg_embeddings = neg_embeddings.view(bs_embed * num_images, seq_len, -1)

att_emb=torch.cat([neg_embeddings, att_emb])

images = pipeline(
    prompt_embeds=id_emb, 
    num_inference_steps=25, 
    guidance_scale=3, 
    num_images_per_prompt=num_images,
    generator=generator,
    cross_attention_kwargs={"att_embedding": att_emb}
    ).images

for i, image in enumerate(images):
    image.save(f'output{i}.png')

