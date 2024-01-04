import argparse
import itertools
import json
import os
import re
from functools import partial

import torch
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist

import sys

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
# from data.bbox_util import Expand2square
# from util.quant import quantize
from fairscale.nn.model_parallel import initialize as fs_init
from model.meta import MetaModel
from util.tensor_parallel import load_tensor_parallel_model_list
from util.misc import init_distributed_mode
from data.conversation.lib import conv_templates, SeparatorStyle
from PIL import Image

import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# from SPHINX import SPHINXModel
from PIL import Image
import os
import torch
import torch.distributed as dist


def collate_fn(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, input_image


class PadToSquare:
    def __init__(self, background_color):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x * 255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

def T_padded_resize(size=448):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t

def main() -> None:
    def get_args_parser():
        parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
        # Model parameters
        parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
                            help='type of llama')
        parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                            help='Path to llama model config')
        parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                            help='path to tokenizer.model')
        parser.add_argument('--img_root', type=str, default="./data/nocaps/images",
                            help='path to tokenizer.model')
        parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')

        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)

        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument('--quant', action="store_true", default=False,
                            help="enable quantization")
        parser.add_argument('--dataset', default='coco_val', type=str)
        parser.add_argument('--input_size', default=448, type=int)
        return parser


    args = get_args_parser().parse_args()

    # world_size = int(os.environ['WORLD_SIZE'])
    # rank = int(os.environ["RANK"])
    # dist.init_process_group(
    #     world_size=world_size, rank=rank,
    #     backend="nccl", init_method=f"env://",
    # )
    # torch.cuda.set_device(rank)
    init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)
    model.bfloat16().cuda()
    max_gen_len = 512
    gen_t = 0.9
    top_p = 0.5
    result = []

    # mp_group tells the model which ranks will work together
    # through model parallel to compose a complete model.
    # When mp_group is None, a single-rank process group will
    # be created and used, which means model parallel size = 1 (not enabled)
    
    # You may also, say, launch 4 processes and make [0,1] and [2,3] ranks to form mp groups, respectively.

    # it's important to make sure that ranks within the same 
    # model parallel group should always receive the same input simultaneously
    # img_path = "/mnt/petrelfs/mengfanqing/donut/donut/pie.png"
    folder_path = '/mnt/petrelfs/mengfanqing/SPHINX/ood/'
    json_path = '/mnt/petrelfs/mengfanqing/SPHINX/ood/ood_openqa/qa.json'
    with open(json_path,'r') as f:
        data = json.load(f)
    for data_tmp in data:
        # ## OCR
        # prompt = """Below is an instruction that describes a task. "
        #                     "Write a response that appropriately completes the request.\n\n"
        #                     "### Instruction:\nconvert this chart to a table.\n\n### Response:"""
        
        # ## Summ
        # prompt = """Below is an instruction that describes a task. "
        #                     "Write a response that appropriately completes the request.\n\n"
        #                     "### Instruction:\nPlease summary the chart.\n\n### Response:"""
        ## OpenQA
        question = data_tmp['question']
        prompt = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease answer my question based on the chart: {question}\n\n### Response:"""
        # ## MathQA
        # prompt = f"""Below is an instruction that describes a task. "
        #                 "Write a response that appropriately completes the request.\n\n"
        #                 "### Instruction:\nPlease use commandline to solve the math question: {question}\n\n### Response:"""
        # ## ReferQA
        # prompt = f"""Below is an instruction that describes a task. "
        #                 "Write a response that appropriately completes the request.\n\n"
        #                 "### Instruction:\nPlease use commandline to solve the math question about the referring box: {question}\n\n### Response:"""
        # img_path = data_tmp['image_path']
        img_path = os.path.join(folder_path,data_tmp['img'])
        image = Image.open(img_path).convert('RGB')
        transform_val = T_padded_resize(448)
        image = transform_val(image).unsqueeze(0)
        image = image.cuda()
        # qas = [["What's in the image?", None]]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            response = model.generate([prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
        print(response[0].split('###')[0])
        data_tmp['pred'] = response[0].split('###')[0]
        print(img_path)
        print('--------------------')
        result.append(data_tmp)
    with open('/mnt/petrelfs/mengfanqing/SPHINX/ood/pred_multitask_nocot_10000.json','w') as f:
        json.dump(result,f)
    # print(response)

    # # if you wanna continue
    # qas[-1][-1] = response
    # qas.append(["Then how does it look like?", None])
    # with torch.cuda.amp.autocast(dtype=torch.float16):
    #     response2 = model.generate_reponse(qas, image, max_gen_len=1024, temperature=0.9, top_p=0.5, seed=0)
    # print(response2)


if __name__ == "__main__":
    # launch this script with `torchrun --master_port=1112 --nproc_per_node=2 inference.py`
    main()