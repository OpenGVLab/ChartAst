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

ds_collections = {
    'coco_val': 'data/cocodet/coco_val.jsonl',
    'plotqa_ocr': '/mnt/petrelfs/path/plotQA/test/annotations3_tiny200.json',
    'chartqa_ocr': '/mnt/petrelfs/share_data/path1/path/ChartQA-main/ChartQA-Dataset/test/annotations2.json',
    'chart-to-text': '/mnt/petrelfs/path/Chart-to-text/statista/test.json',
    'openqa': '/mnt/petrelfs/path/openCQA/test_onlyqa1_ab.json',
    'chartqa': '/mnt/petrelfs/path/ChartQA-main/ChartQA-Dataset/test/test_all1.json',
    'mathqa': '/mnt/petrelfs/path/plotQA/QA/test_commandline_tiny2k_onlyqa1.json',
    'referqa': '/mnt/petrelfs/path/plotQA/referring_QA/test_referring_qa_tiny2k.json',
    'chartqa_human_math': '/mnt/petrelfs/path/donut/test/chartqahuman_ours_human.json',
    'chartqa_augment_math': '/mnt/petrelfs/path/donut/test/chartqahuman_ours_augment.json'

}




#def collate_fn(batches):
#    texts = [_['text'] for _ in batches]
#    bboxes = [_['bbox'] for _ in batches]
#    hws = [_['hw'] for _ in batches]
#    input_image = torch.cat([_['image'] for _ in batches])
#
#    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')
#
#    return texts, input_image, bboxes, hws

def collate_fn(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, input_image

def collate_fn_ocr(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    datafroms = [_['datafrom'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, input_image, datafroms


def collate_fn_summ(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    datafroms = [_['datafrom'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, input_image, datafroms


def collate_fn_openqa(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    questions = [_['question'] for _ in batches]
    prompt_questions = [_['prompt_question'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])
    question_types = [_['question_type'] for _ in batches]

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, questions, prompt_questions, input_image, question_types


def collate_fn_chartqa(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    questions = [_['question'] for _ in batches]
    prompt_questions = [_['prompt_question'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])
    question_types = [_['question_type'] for _ in batches]

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, questions, prompt_questions, input_image, question_types


def collate_fn_mathqa(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    prompt_questions = [_['prompt_question'] for _ in batches]
    questions = [_['question'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])
    answers = [_['answer'] for _ in batches]

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, prompt_questions, questions, input_image, answers



def collate_fn_referqa(batches):
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    prompt_questions = [_['prompt_question'] for _ in batches]
    questions = [_['question'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])
    answers = [_['answer'] for _ in batches]

    # input_ids = tokenizer.encode(texts, return_tensors='pt', padding='longest')

    return texts, gts, prompt_questions, questions, input_image, answers



from PIL import Image

import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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
      

class ChartOCRDataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        datafrom = data["datafrom"]
        table = data['table']
        if self.prompt == 'plotqa_ocr':
            image = os.path.join('/mnt/petrelfs/share_data/path1/path/plotQA/test/png', data['img'])
        elif self.prompt == 'chartqa_ocr':
            image = os.path.join('/mnt/petrelfs/share_data/path1/path/ChartQA-main/ChartQA-Dataset/test/png', data['img'])
        else :
            image = None 
        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': os.path.join('/mnt/petrelfs/share_data/path1/path/ChartQA-main/ChartQA-Dataset/test/png', data['img']),
            'gt': table,
            'image': image,
            'datafrom': self.prompt 
        }

class ChartSummDataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        datafrom = data["datafrom"]
        summ = data['summ']
        if datafrom == 'chart-to-text_pew':
            image = os.path.join('/mnt/petrelfs/share_data/luquanfeng/Chart-to-text/pew_dataset/dataset', data['img'])
        else:
            image = os.path.join('/mnt/petrelfs/share_data/luquanfeng/Chart-to-text/statista_dataset/dataset', data['img'])

        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': os.path.join('/mnt/petrelfs/share_data/path1/path/ChartQA-main/ChartQA-Dataset/test/png', data['img']),
            'gt': summ,
            'image': image,
            'datafrom': datafrom,   
        }


class ChartQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        # table = data['table']
        question = data["question"]
        question_type = data["question_type"]
        prompt_question_template = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease answer my question based on the chart: {question}\n\n### Response:"""

        if "answer" in data.keys():
            answer = data["answer"]
        else:
            answer = data["gt"]
        try:
            image = os.path.join('/mnt/petrelfs/path/ChartQA-main/ChartQA-Dataset/test/png', data['img'])
        except:
            image = None 
        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': os.path.join('/mnt/petrelfs/share_data/path1/path/ChartQA-main/ChartQA-Dataset/test/png', data['img']),
            'gt': answer,
            'question': question,
            'prompt_question':prompt_question_template,
            'image': image,
            'question_type': question_type   
        }


class ChartOpenQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()
        
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        # table = data['table']
        question = data["question"]
        prompt_question_template = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease answer my question based on the chart: {question}\n\n### Response:"""
        
        
        answer = data["answer"]
        
        image = os.path.join('/mnt/petrelfs/share_data/luquanfeng/OpenCQA/chart_images', data['img'])

        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': os.path.join('/mnt/petrelfs/share_data/luquanfeng/OpenCQA/chart_images', data['img']),
            'gt': answer,
            'question': question,
            'prompt_question':prompt_question_template,
            'image': image,
            'question_type': 'ab'          
        }



class ChartMathQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        # table = data['table']
        
        question = data["question"]
        
        prompt_question_template = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease use commandline to solve the math question: {question}\n\n### Response:"""
        if 'answer' in data.keys():

            answer_gt = data["answer"]
        else:
            answer_gt = ""
        if "answer_commandline" in data.keys():
            answer = data["answer_commandline"]
        elif "commandline" in data.keys():
            answer = data["commandline"]
        elif "gt" in data.keys():
            answer = data["gt"]
        
        image = os.path.join('/mnt/petrelfs/path/plotQA/test/png', data['img'])

        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': os.path.join('/mnt/petrelfs/path/plotQA/test/png', data['img']),
            'gt': answer,
            'prompt_question':prompt_question_template,
            'question': question,
            'image': image,
            'answer':answer_gt     
        }

class ChartQAMathQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        # table = data['table']
        
        question = data["question"]
        question = question.split('based on the chart:')[1]
        question = question.split('\n\n')[0].strip()
        prompt_question_template = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease use commandline to solve the math question: {question}\n\n### Response:"""
        if 'gt' in data.keys():

            answer_gt = data["gt"]
        else:
            answer_gt = ""
        if "answer_commandline" in data.keys():
            answer = data["answer_commandline"]
        elif "commandline" in data.keys():
            answer = data["commandline"]
        elif "gt" in data.keys():
            answer = data["gt"]
        else:
            answer = 'None'
        
        # image = os.path.join('/mnt/petrelfs/path/plotQA/test/png', data['img'])
        image = data['image_path']

        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': data['image_path'],
            'gt': answer,
            'prompt_question':prompt_question_template,
            'question': question,
            'image': image,
            'answer':answer_gt     
        }


class ChartReferQADataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size):
        with open(test, 'r') as f:
            self.datas = json.load(f)
        # self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform_val = T_padded_resize(input_size)
        # self.expand_transformer = Expand2square()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        # table = data['table']
        
        question = data["question"]
        prompt_question_template = f"""Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPlease use commandline to solve the math question about the referring box: {question}\n\n### Response:"""

        answer_gt = data["answer"]
        if "commandline" in data.keys():
            answer = data["commandline"]
        else:
            answer = "<s_question>" + str(question) + "</s_question>" + "<s_answer>" + str(data["answer"]) + "</s_answer>"
        
        image = os.path.join('/mnt/petrelfs/share_data/luquanfeng/referring_box/test/img', data['img'])

        image = Image.open(image).convert('RGB')
        # image_pad, _ = self.expand_transformer(image, None)
        image = self.transform_val(image).unsqueeze(0)

        return {
            'text': os.path.join('/mnt/petrelfs/share_data/luquanfeng/referring_box/test/img', data['img']),
            'gt': answer,
            'prompt_question':prompt_question_template,
            'question': question,
            'image': image,
            'answer':answer_gt     
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


if __name__ == '__main__':

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
        parser.add_argument('--batch_size', default=10, type=int)
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

    # define the model
    init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)
    # print("Quantizing model to 4bit!")

    # from transformers.utils.quantization_config import BitsAndBytesConfig

    # quantization_config = BitsAndBytesConfig.from_dict(
    #     config_dict={
    #         "load_in_8bit": False,
    #         "load_in_4bit": True,
    #         "bnb_4bit_quant_type": "nf4",
    #     },
    #     return_unused_kwargs=False,
    # )
    # quantize(model, quantization_config)

    # print("Model = %s" % str(model))
    model.bfloat16().cuda()

#     prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

# ###Human: convert this chart to a table.
# ###Assistant: """
    # tasks = ['chartqa']
    tasks = ['chartqa','ocr','opencqa','mathqa','referqa']
    for task in tasks:
        if task == 'ocr' or task == 'summ':
            if task == 'ocr':
                args.dataset = 'chartqa_ocr'
                prompt = """Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            "### Instruction:\nconvert this chart to a table.\n\n### Response:"""
                
                dataset = ChartOCRDataset(test=ds_collections[args.dataset],
                                        input_size=args.input_size,
                                        prompt=args.dataset)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_ocr),
                )
            else:
                args.dataset = 'chart-to-text'
                prompt = """Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            "### Instruction:\nPlease summary the chart.\n\n### Response:"""
                dataset = ChartSummDataset(test=ds_collections[args.dataset],
                                        input_size=args.input_size,
                                        prompt=args.dataset)

                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_summ),
                )

            max_gen_len = 2048
            gen_t = 0.9
            top_p = 0.5
            outputs = []
            idx = 0
            for _prompt, gt, image, datafrom in tqdm(dataloader):
                #print(gt, image)
                if dist.get_rank() == 0:
                    dist.barrier()
                    dist.broadcast_object_list([[prompt] * len(_prompt), image, max_gen_len, gen_t, top_p])
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        results = model.generate([prompt] * len(_prompt), image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
                    print(f'image: {_prompt[0]} pred: {results[0]}')
                    for pp, y, answer, datafrom in zip(_prompt, gt, results, datafrom):
                        answer = answer.split('###')[0]
                        # answer = answer
                        outputs.append({
                            'gt': y, 
                            'answer': answer,
                            'image': pp,
                            'datafrom': datafrom
                        })
                else:
                    dist.barrier()

                    input_data = [None for _ in range(5)]
                    dist.broadcast_object_list(input_data)
                    _prompt, image, max_gen_len, gen_t, top_p = input_data
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

            torch.distributed.barrier()

            world_size = torch.distributed.get_world_size()
            merged_outputs = [None for _ in range(world_size)]
            print(world_size,' world_size')
            torch.distributed.all_gather_object(merged_outputs, outputs)

            merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
            PATTERN = re.compile(r'\[(.*?)\]')

            if torch.distributed.get_rank() == 0:
                correct = total_cnt = 0
                final_results = []
                for i, output in enumerate(merged_outputs):
                    final_ans = {'pred': output['answer'], 'image_path': output['image'], 'gt': output['gt'],'datafrom': output['datafrom']}
                    final_results.append(final_ans)

                if isinstance(args.pretrained_path, list):
                    pre_path = args.pretrained_path[0]
                else:
                    pre_path = args.pretrained_path
                os.makedirs(f'/mnt/petrelfs/path/SPHINX/mixed_result/{task}_result/{args.dataset}', exist_ok=True)
                with open(f'/mnt/petrelfs/path/SPHINX/mixed_result/{task}_result/{args.dataset}/{task}_448_bf16_t0.9_p0.5_otherprompt_multitask_otherbasetype_ft_stock_epoch0.json', 'w') as f:
                    f.write(json.dumps(final_results))

            torch.distributed.barrier()


        elif task == 'opencqa' or task == 'chartqa':
            if task == 'opencqa':
                args.dataset = 'openqa'
                dataset = ChartOpenQADataset(test=ds_collections[args.dataset],
                                    input_size=args.input_size,
                                    prompt=args.dataset)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_openqa),
                )

            else:
                args.dataset = 'chartqa'
                dataset = ChartQADataset(test=ds_collections[args.dataset],
                                    input_size=args.input_size,
                                    prompt=args.dataset)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_chartqa),
                )

        
            

            max_gen_len = 1024
            gen_t = 0.9
            top_p = 0.5
            outputs = []
            idx = 0
            for _prompt, gt, question, prompt_question, image, question_type in tqdm(dataloader):
                #print(gt, image)
                if dist.get_rank() == 0:
                    dist.barrier()
                    dist.broadcast_object_list([prompt_question, image, max_gen_len, gen_t, top_p])
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        results = model.generate(prompt_question, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
                    print(f'image: {_prompt[0]} pred: {results[0]}')
                    for pp, y, answer, prompt_question in zip(_prompt, gt, results, prompt_question):
                        answer = answer.split('###')[0]
                        # answer = answer
                        outputs.append({
                            'gt': y, 
                            'answer': answer,
                            'question': prompt_question,
                            'image': pp,
                            'question_type': question_type
                        })
                else:
                    dist.barrier()

                    input_data = [None for _ in range(5)]
                    dist.broadcast_object_list(input_data)
                    _prompt, image, max_gen_len, gen_t, top_p = input_data
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

            torch.distributed.barrier()

            world_size = torch.distributed.get_world_size()
            merged_outputs = [None for _ in range(world_size)]
            print(world_size,' world_size')
            torch.distributed.all_gather_object(merged_outputs, outputs)

            merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
            PATTERN = re.compile(r'\[(.*?)\]')

            if torch.distributed.get_rank() == 0:
                correct = total_cnt = 0
                final_results = []
                for i, output in enumerate(merged_outputs):
                    final_ans = {'pred': output['answer'], 'image_path': output['image'], 'gt': output['gt'], 'question':output['question'], 'question_type':output['question_type']}
                    final_results.append(final_ans)

                if isinstance(args.pretrained_path, list):
                    pre_path = args.pretrained_path[0]
                else:
                    pre_path = args.pretrained_path
                os.makedirs(f'/mnt/petrelfs/path/SPHINX/mixed_result/{task}_result/{args.dataset}', exist_ok=True)
                with open(f'/mnt/petrelfs/path/SPHINX/mixed_result/{task}_result/{args.dataset}/{task}_448_bf16_t0.9_p0.5_otherprompt_multitask_otherbasetype_ft_stock_epoch0.json', 'w') as f:
                    f.write(json.dumps(final_results))

            torch.distributed.barrier()



        elif task == 'mathqa' or task == 'referqa' or task == 'chartqa_human_math' or task == 'chartqa_augment_math':
            if task == 'mathqa':
                args.dataset = 'mathqa'
                dataset = ChartMathQADataset(test=ds_collections[args.dataset],
                                    input_size=args.input_size,
                                    prompt=args.dataset)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_mathqa),
                )
            elif task == 'chartqa_human_math' or task == 'chartqa_augment_math':
                args.dataset = task
                dataset = ChartQAMathQADataset(test=ds_collections[args.dataset],
                                    input_size=args.input_size,
                                    prompt=args.dataset)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_mathqa),
                )
                
            else:
                args.dataset = 'referqa'
                dataset = ChartReferQADataset(test=ds_collections[args.dataset],
                                    input_size=args.input_size,
                                    prompt=args.dataset)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    sampler=InferenceSampler(len(dataset)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=partial(collate_fn_referqa),
                )
        
            
            max_gen_len = 512
            gen_t = 0.9
            top_p = 0.5
            outputs = []
            idx = 0
            for _prompt, gt, prompt_question, question, image, answer_gt in tqdm(dataloader):
                #print(gt, image)
                if dist.get_rank() == 0:
                    dist.barrier()
                    dist.broadcast_object_list([prompt_question, image, max_gen_len, gen_t, top_p])
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        results = model.generate(prompt_question, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
                    print(f'image: {_prompt[0]} pred: {results[0]}')
                    for pp, y, answer, prompt_question, answer_gt in zip(_prompt, gt, results, prompt_question, answer_gt):
                        answer = answer.split('###')[0]
                        # answer = answer
                        outputs.append({
                            'gt': y, 
                            'answer': answer,
                            'question': prompt_question,
                            'image': pp,
                            'answer_gt':answer_gt
                        })
                else:
                    dist.barrier()

                    input_data = [None for _ in range(5)]
                    dist.broadcast_object_list(input_data)
                    _prompt, image, max_gen_len, gen_t, top_p = input_data
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

            torch.distributed.barrier()

            world_size = torch.distributed.get_world_size()
            merged_outputs = [None for _ in range(world_size)]
            print(world_size,' world_size')
            torch.distributed.all_gather_object(merged_outputs, outputs)

            merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
            PATTERN = re.compile(r'\[(.*?)\]')

            if torch.distributed.get_rank() == 0:
                correct = total_cnt = 0
                final_results = []
                for i, output in enumerate(merged_outputs):
                    final_ans = {'pred': output['answer'], 'image_path': output['image'], 'gt': output['gt'], 'answer_gt':output['answer_gt'], 'question':output['question']}
                    final_results.append(final_ans)

                if isinstance(args.pretrained_path, list):
                    pre_path = args.pretrained_path[0]
                else:
                    pre_path = args.pretrained_path
                os.makedirs(f'/mnt/petrelfs/path/SPHINX/mixed_result/{task}_result/{args.dataset}', exist_ok=True)
                with open(f'/mnt/petrelfs/path/SPHINX/mixed_result/mixed_result/{task}_result/{args.dataset}/{task}_448_bf16_t0.9_p0.5_otherprompt_multitask_otherbasetype_ft_stock_epoch0.json', 'w') as f:
                    f.write(json.dumps(final_results))

            torch.distributed.barrier()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
