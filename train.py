# ------------------------------------------------------------------------------
# modified from https://github.com/huggingface/diffusers
# Last Modified: 2025-06-26
# Author: Zhang Yuhao
# ------------------------------------------------------------------------------
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import time
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import torch.nn.utils as utils
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,LCMScheduler,DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from attportrait import CLIPTextModelWrapper,bulid_dataset,iresnet100,cosine_similarity,DRtune_call,AttributeClassifier,Checkpointer,clip_embedding_forward,hack_unet_attn_layers, ATTAttnProcessor
from safetensors.torch import load_file
import itertools


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.32.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default='LAION-Face', 
        help="dataset name")
    
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the ",
            "expense of doing an extra forward pass. See: https://arxiv.org/abs/2312.00210",
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--DRtune_bp_steps",
        type=int,
        default=3,
        help="num of bp steps of DRtune",
    )
    parser.add_argument(
        "--DRtune_inference_steps",
        type=int,
        default=25,
        help="num of inference steps of DRtune",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--coef_bce",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--coef_id",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()
    StableDiffusionPipeline.__call__=DRtune_call


    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    
    id_text_encoder = CLIPTextModelWrapper.from_pretrained(
        'models', subfolder="encoder", revision=args.revision
    )
    att_text_encoder = CLIPTextModelWrapper.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        'models', subfolder="arc2face", revision=args.non_ema_revision
    )     

    if args.coef_id!=0:
        arcface_net=iresnet100()
        ckpt=torch.load('models/arcface_glint360K_r100/backbone.pth')
        arcface_net.load_state_dict(ckpt,strict=True)
        arcface_net.eval()
    clip_embedding=att_text_encoder.text_model.embeddings
    clip_embedding.forward = clip_embedding_forward.__get__(clip_embedding,type(clip_embedding))


    classifier = AttributeClassifier(n_attributes=40,pretrained=False)
    checkpointer = Checkpointer('models/classifier', {'classifier': classifier, })
    checkpointer.load_if_exists(map_location=accelerator.device, load_best=True, strict=False)



    hidden_size = id_text_encoder.config.hidden_size
    default_att_names = ['Bald','Young','Male', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                         'Mouth_Slightly_Open', 'No_Beard', 'Pale_Skin']
    default_att_dict={ default_att_names[i] : i  for i in range(len(default_att_names))}

    unet.set_default_attn_processor()
    
    # Freeze vae, text_encoder and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet=unet.to(dtype=torch.float16)
    classifier.requires_grad_(False)
    id_text_encoder.requires_grad_(False)
    att_text_encoder.requires_grad_(False)
    hack_unet_attn_layers(unet, ATTAttnProcessor)


    classifier.eval()

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=id_text_encoder,
        vae=vae,
        unet=unet,
        revision=args.revision,
        variant=args.variant,
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            id_text_encoder.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_flag=1
                for i, model in enumerate(models):
                    if isinstance(model,UNet2DConditionModel):
                        if unet_flag==1:
                            model.save_pretrained(os.path.join(output_dir, "unet"))
                            unet_flag=0
                        weights.pop(0)

                    # make sure to pop weight so that corresponding model is not saved again
                   

        def load_model_hook(models, input_dir):
            
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models[0]
                # load diffusers style into model
                if isinstance(model,UNet2DConditionModel):
                    models.pop(0)
                    load_model= UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)
                    del load_model
                    file_path=os.path.join(input_dir,'unet','diffusion_pytorch_model.safetensors')
                    unet_weights = load_file(file_path)
                    model.load_state_dict(unet_weights,strict=True)
                    del unet_weights                

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW



    uent_train_param=[]
    for n,p in unet.named_parameters():
        if 'att_to_k' in n or 'att_to_v' in n:
            uent_train_param.append(p)
        else:
            p.requires_grad=False
        
    param_dict=[
        {'params': uent_train_param, 'lr': args.learning_rate}, 
    ]
    
    bce = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optimizer_cls(
        param_dict,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(572, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    res_transform = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.Resize((224, 224)),
    ])
    id_transform=transforms.Compose([
        transforms.Resize((112, 112)),
    ])
    
    dataset=bulid_dataset('train',train_transforms,default_att_names,args)


    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    def collate_fn(examples):
        pixel_values = torch.stack([example["image"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        embeddings = torch.stack([example["embedding"] for example in examples])
        attributes = torch.stack([example["attributes"] for example in examples])
        edit_attributes = torch.stack([example["edit_attributes"] for example in examples])
        return {"images": pixel_values, "embeddings": embeddings,'attributes':attributes,'edit_attributes':edit_attributes}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet,optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,optimizer, train_dataloader, lr_scheduler
    )
    model_ls = [unet]

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encoder and vae to gpu and cast to weight_dtype

    vae.to(accelerator.device, dtype=weight_dtype)
    classifier=classifier.to(accelerator.device,dtype=weight_dtype)
    if args.coef_id!=0:
        arcface_net=arcface_net.to(accelerator.device,dtype=weight_dtype)

    id_text_encoder.to(accelerator.device, dtype=weight_dtype)
    att_text_encoder.to(accelerator.device, dtype=weight_dtype)
    


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            dataloader_state=torch.load(os.path.join(args.output_dir, path,'dataloader.pth'))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_steps=dataloader_state['step']
            resume_epoch=dataloader_state['epoch']

    else:
        initial_global_step = 0
        resume_steps=0
        resume_epoch=0

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    total_step=args.DRtune_inference_steps

    for epoch in range(resume_epoch, args.num_train_epochs):
        train_mse_loss = 0.0
        train_bce_loss = 0.0
        train_id_loss  = 0.0
        for step, batch in enumerate(train_dataloader):
            if step<resume_steps:
                continue

            with accelerator.accumulate(model_ls):
          
                ####DRtune parameters####
                bp_on_trained = True
                early_exit = False
                double_laststep = False
                fast_training = False
                

                interval = total_step // args.DRtune_bp_steps
                max_start = total_step - interval * (args.DRtune_bp_steps - 1) - 1
                start = random.randint(0, max_start)
                training_steps = list(range(start, total_step, interval))
                detach_gradient = True
                ##########################

                # Convert images to latent space
                images=batch["images"].to(accelerator.device).to(weight_dtype)
                latents = vae.encode(batch["images"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning

                face_embs=batch["embeddings"].to(accelerator.device)
                face_embs=face_embs.to(weight_dtype)

                attributes_embs=batch["attributes"].to(accelerator.device)
        

                prompts=[]
                for atts in attributes_embs:
                    prompt_pre='who is '
                    for i in range(len(atts)):
                        if i==0:
                            if atts[i]==1:
                                prompt_pre=prompt_pre+default_att_names[i]+', '
                        elif i==1:
                            if atts[i]==1:
                                prompt_pre=prompt_pre+'Young, '
                            else:
                                prompt_pre=prompt_pre+'Old, '
                        elif i==2:
                            if atts[i]==1:
                                prompt_pre=prompt_pre+'Male, '
                            else:
                                prompt_pre=prompt_pre+'Female, '
                        else:
                            if atts[i]==1:
                                prompt_pre=prompt_pre+'with '+default_att_names[i]+' '
                    prompt=prompt_pre
                    prompts.append(prompt)

                new_prompts=[]
                new_attributes_embs=attributes_embs
                for att in new_attributes_embs:
                    choice_set=set([i for i in range(11)])
                    if att[default_att_dict['Black_Hair']]==1:
                        choice_set.discard(default_att_dict['Blond_Hair'])
                        choice_set.discard(default_att_dict['Bald'])
                    if att[default_att_dict['Blond_Hair']]==1:
                        choice_set.discard(default_att_dict['Black_Hair'])
                        choice_set.discard(default_att_dict['Bald'])
                    if att[default_att_dict['Bangs']]==1:
                        choice_set.discard(default_att_dict['Bald'])
                    if att[default_att_dict['Bald']]==1:
                        choice_set.discard(default_att_dict['Black_Hair'])
                        choice_set.discard(default_att_dict['Blond_Hair'])
                        choice_set.discard(default_att_dict['Bangs'])
                    change_att=random.choice(list(choice_set))
                    if att[change_att]==1:
                        att[change_att]=0
                    else:
                        att[change_att]=1
                
                for atts in new_attributes_embs:
                    new_prompt_pre='who is '
                    for i in range(len(atts)):
                        if i==0:
                            if atts[i]==1:
                                new_prompt_pre=new_prompt_pre+default_att_names[i]+', '
                        elif i==1:
                            if atts[i]==1:
                                new_prompt_pre=new_prompt_pre+'Young, '
                            else:
                                new_prompt_pre=new_prompt_pre+'Old, '
                        elif i==2:
                            if atts[i]==1:
                                new_prompt_pre=new_prompt_pre+'Male, '
                            else:
                                new_prompt_pre=new_prompt_pre+'Female, '
                        else:
                            if atts[i]==1:
                                new_prompt_pre=new_prompt_pre+'with '+default_att_names[i]+' '
                    new_prompt=new_prompt_pre
                    new_prompts.append(new_prompt)


                arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]
                input_ids_atts = tokenizer(
                        prompts,
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(face_embs.device)
                
                edit_input_ids = tokenizer(
                        new_prompts,
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(face_embs.device)
                
                input_ids = tokenizer(
                    'photo of a id person',
                    truncation=True,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(face_embs.device)

                face_embs_padded = F.pad(face_embs, (0, hidden_size-512), "constant", 0)#[bs,768]
                input_ids=input_ids.repeat(len(face_embs), 1)
                token_embs = id_text_encoder(input_ids=input_ids, return_token_embs=True)#word embedding
                token_embs[input_ids==arcface_token_id] = face_embs_padded

                encoder_hidden_states = id_text_encoder(
                    input_ids=input_ids,
                    input_token_embs=token_embs,
                    return_dict=False
                )[0]

                encoder_hidden_states_atts = att_text_encoder(
                    input_ids=input_ids_atts,
                    return_dict=False
                )[0]

                edit_encoder_hidden_states_atts = att_text_encoder(
                    input_ids=edit_input_ids,
                    return_dict=False
                )[0]
                

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.dream_training:
                    noisy_latents, target = compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,
                        noisy_latents,
                        target,
                        final_encoder_hidden_states,
                        args.dream_detail_preservation,
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states,cross_attention_kwargs={"att_embedding":encoder_hidden_states_atts} ,return_dict=False)[0]#4G



                neg_embeddings=torch.zeros_like(edit_encoder_hidden_states_atts)
                bs_embed, seq_len, _ = edit_encoder_hidden_states_atts.shape

                edit_encoder_hidden_states_atts = edit_encoder_hidden_states_atts.repeat(1, 1, 1)
                edit_encoder_hidden_states_atts = edit_encoder_hidden_states_atts.view(bs_embed *1, seq_len, -1)

                neg_embeddings = neg_embeddings.repeat(1, 1, 1)
                neg_embeddings = neg_embeddings.view(bs_embed * 1, seq_len, -1)


                edit_encoder_hidden_states_atts=torch.cat([neg_embeddings, edit_encoder_hidden_states_atts])



                gen_images = pipeline(
                    prompt_embeds=encoder_hidden_states, 
                    num_inference_steps=total_step, 
                    guidance_scale=3.0, 
                    num_images_per_prompt=1,
                    training_timesteps=training_steps,
                    detach_gradient=detach_gradient,
                    cross_attention_kwargs={"att_embedding": edit_encoder_hidden_states_atts}
                )


                if args.coef_id!=0:
                    id_srcs=arcface_net(id_transform(images))
                    id_gens=arcface_net(id_transform(gen_images))
                    id_loss=1-cosine_similarity(id_srcs,id_gens).mean() 
                att_images=res_transform(gen_images)
                att_images=torch.clamp(att_images, min=-1.0, max=1.0)
                
                logits=classifier(att_images)
                att_logits=logits[:,dataset.attri_index]

                bce_loss = bce(att_logits.float(), new_attributes_embs.float())
         
                
                if args.snr_gamma is None:
                    mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_mse_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean()
                train_mse_loss += avg_mse_loss.item() / args.gradient_accumulation_steps
                
                avg_bce_loss = accelerator.gather(bce_loss.repeat(args.train_batch_size)).mean()
                train_bce_loss += avg_bce_loss.item() / args.gradient_accumulation_steps
                
                if args.coef_id!=0:
                    avg_id_loss = accelerator.gather(id_loss.repeat(args.train_batch_size)).mean()
                    train_id_loss += avg_id_loss.item() / args.gradient_accumulation_steps
                    loss=mse_loss+args.coef_bce*bce_loss+args.coef_id*id_loss
                else:
                    loss=mse_loss+args.coef_bce*bce_loss



                # Backpropagate
                accelerator.backward(loss)


                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"mse_loss": train_mse_loss, "bce_loss": train_bce_loss,'id_loss':train_id_loss}, step=global_step)
                train_mse_loss = 0.0
                train_bce_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        vis_image = pipeline.image_processor.postprocess(gen_images.detach())
                        for i,img in enumerate(vis_image):
                            img.save(os.path.join(save_path,f'vis_{i}.jpg'))

                        logger.info(f"Saved state to {save_path}")
                        #save step & epoch
                        dataloader_state={'step':step+1,'epoch':epoch}
                        
                        torch.save(dataloader_state,os.path.join(save_path,'dataloader.pth'))
                        logger.info(f"Saved state to {save_path}")

            if args.coef_id!=0:
                logs = {"mse_loss": mse_loss.detach().item(), "bce_loss":bce_loss.detach().item(),'id_loss':id_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            else:
                logs = {"mse_loss": mse_loss.detach().item(), "bce_loss":bce_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            # logs = {"mse_loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
   
if __name__ == "__main__":
    main()