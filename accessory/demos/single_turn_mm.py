import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from model.meta import MetaModel

import argparse
import torch
import torch.distributed as dist
import gradio as gr
from examples.examples import read_examples
from PIL import Image

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

# from data.alpaca import format_prompt
from data.transform import get_transform
from util.tensor_parallel import load_tensor_parallel_model_list
from util.tensor_type import default_tensor_type
import json
import os
from copy import deepcopy
# from QA.func_lib.exec_func_lib import *
from exec_func_lib import *
import numpy as np
import re
# from QA.func_lib.exec_func_lib import func_list
from exec_func_lib import func_list
import json
import numpy
import ast


def translate_json_to_natural_language(json_str):
    json_data = ast.literal_eval(json_str)
    function_steps = []
    
    for step_key, step_value in json_data.items():
        print(step_key, step_value)
        it = step_key[-1:]
        if int(it) == 1:
            func_desc = f"第{step_key[4:]}步：执行{step_value[f'func{it}']}函数，输入{step_value[f'arg{it}']}，输出 output1:{step_value[f'output{it}']}.\n"
        else:
            func_desc = f"第{step_key[4:]}步：执行{step_value[f'func{it}']}函数，输入{step_value[f'arg{it}']}，输出{step_value[f'output{it}']}.\n"
        function_steps.append(func_desc)
    
    return ' '.join(function_steps)

dd = {'step1': {'func1': 'select',
                'arg1': 'goods and services',
                'output1': ['21826200000',
                            '27726200000',
                            '20407200000',
                            '20907200000',
                            '23156200000',
                            '21915200000']},
      'step2': {'func2': 'numpy.mean', 'arg2': 'output1', 'output2': 'mean'}}


def format_prompt(prompt):
    prompt_formated = f"""
Below is an instruction that describes a task.

Write a response that appropriately completes the request.

Instruction:

{prompt}



### Response:
"""
    return prompt_formated.strip()
    


def __get_most_likely_str(str_list: list, str1: str):  # get the most likely string in str_list
    def edit_distance(str1, str2):
        len1, len2 = len(str1), len(str2)
        # 创建一个二维数组来存储编辑距离
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        # 初始化第一行和第一列
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        # 填充二维数组
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[len1][len2]

    distance = [edit_distance(str1, s) for s in str_list]
    return str_list[np.argmin(distance)], np.min(distance)


def __fix_num(arg: list):
    def is_numeric_string(s):
        pattern = r'^-?\d+(\.\d+)?$'
        return bool(re.match(pattern, s))

    res = []
    for ele in arg:
        if isinstance(ele, list):
            ele = __fix_num(ele)
        else:
            ele = str(ele).lower()
            if is_numeric_string(ele):
                ele = float(ele)
        res.append(ele)
    return res


def __fix0(cmdline: dict):  # lower case
    return eval(str(cmdline).lower())


def __fix1(cmdline: dict):
    cmd_res = {}
    # fix step order number
    for idx, key in enumerate(cmdline.keys()):
        now_step = {}
        step_ele = [k for k in cmdline[key]]
        assert len(step_ele) == 3
        func, arg, output = cmdline[key][step_ele[0]], cmdline[key][step_ele[1]], cmdline[key][step_ele[2]]

        if not isinstance(arg, list):  # change arg type to list
            arg = [arg]
        arg = __fix_num(arg)
        if isinstance(output, list):
            output = __fix_num(output)

        now_step[f'func{idx + 1}'], now_step[f'arg{idx + 1}'], now_step[f'output{idx + 1}'] = func, arg, output
        cmd_res[f'step{idx + 1}'] = now_step

    return cmd_res


def __fix2(cmdline: dict):  # fix func
    steps = len(cmdline)
    for i in range(1, steps + 1):
        step = f'step{i}'
        func = cmdline[step][f'func{i}']
        if func not in func_list:
            fix_func, distance = __get_most_likely_str(func_list, func)
            if distance < len(func):
                cmdline[step][f'func{i}'] = fix_func

    return cmdline


def __fix3(cmdline: dict):  # fix arg
    runtime_variables = []
    steps = len(cmdline)
    for i in range(1, steps + 1):
        step = f'step{i}'
        func, params, out = cmdline[step][f'func{i}'], cmdline[step][f'arg{i}'], cmdline[step][f'output{i}']
        if func == 'select':
            runtime_variables.append(f'output{i}')
        else:
            for j in range(len(params)):
                par = params[j]
                if isinstance(par, str):
                    if par not in runtime_variables:
                        fix_par, distance = __get_most_likely_str(runtime_variables, par)
                        if distance < len(par):  # the threshold?
                            cmdline[step][f'arg{i}'][j] = fix_par

            runtime_variables.append(out)

    return cmdline


def correct(cmdline: dict):
    # todo
    try:
        cmdline = __fix0(cmdline)
        cmdline = __fix1(cmdline)
        cmdline = __fix2(cmdline)
        cmdline = __fix3(cmdline)
        return cmdline
    except:
        print('\ncorrect error-------------------------', cmdline)
        return None


def exec_all(answer):
    output_dict = {}
    try:
        steps = len(answer)
        for i in range(1, steps + 1):
            now_step = answer["step{}".format(i)]
            func_name, args_name, output_name = "func{}".format(i), "arg{}".format(i), "output{}".format(i)
            func, args, output_valuable = now_step[func_name], now_step[args_name], now_step[output_name]
            # 处理函数
            if func == 'select':  # select 直接跳过不执行
                output_dict[output_name] = output_valuable
                continue
            else:
                if func.startswith('np.') or func.startswith('numpy.'):
                    func = func[func.find('.') + 1:]
                    exec_func_wrapper = exec_np
                elif func.startswith('pd.') or func.startswith('pandas.'):
                    raise ValueError("pandas does not support")
                else:
                    exec_func_wrapper = exec_normal

            # 处理参数
            if type(args) is not list:
                raise ValueError("type: {} is not list:".format(args))
            if len(args) == 1 and type(output_dict.get(args[0])) is list:  # 处理arg替换过后是一个list的情况
                args = output_dict[args[0]]
            for j in range(len(args)):
                if type(args[j]) is not list and output_dict.get(args[j]) is not None:
                    args[j] = output_dict[args[j]]

            # 执行函数
            res = exec_func_wrapper(func, args)
            output_dict[output_name] = res
            output_dict[output_valuable] = res

            if i == steps:
                return str(res)

    except BaseException as e:
        # print(e)
        # print("execute error")
        return None


def exec_one(pred,use_corrector=True):
    pred = pred.strip()
    # print(pred)
    if use_corrector:
        pred = correct(pred)
    
    ans = exec_all(deepcopy(pred))
    if ans is None:
        raise Exception('exec error')
    return ans


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama_ens5', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/mnt/petrelfs/share_data/llm_llama2/llama2_raw/llama-2-13b/params.json', type=str, nargs="*",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="/mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default='/mnt/petrelfs/mengfanqing/SPHINX/LLaMA2-Accessory/accessory/exps/finetune/mm/output/finetune/mm/chart_multitask_instruction_tuning_gpu16_nocot/epoch0-iter49999', type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')

    parser.add_argument('--image_transform', default='padded_resize', type=str,
                        help='type of image transformation (see accessory/data/transform.py for options)')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="The dtype used for model weights and inference.")
    parser.add_argument('--quant', action='store_true', help="enable quantization")
    return parser

args = get_args_parser().parse_args()

# define the model
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(args.model_parallel_size)
target_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}[args.dtype]
with default_tensor_type(dtype=target_dtype, device="cpu" if args.quant else "cuda"):
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)

print(f"load pretrained from {args.pretrained_path}")
load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
print("load result: ", load_result)


if args.quant:
    print("Quantizing model to 4bit!")
    from util.quant import quantize
    from transformers.utils.quantization_config import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig.from_dict(
        config_dict={
            "load_in_8bit": False,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
        return_unused_kwargs=False,
    )
    quantize(model, quantization_config)

print("Model = %s" % str(model))
model.bfloat16().cuda()


@ torch.inference_mode()
def generate(
        img_path,
        prompt,
        max_gen_len,
        gen_t, top_p
):
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        image = get_transform(args.image_transform,size=448)(image).unsqueeze(0)
    else:
        image = None

    # text output
    _prompt = format_prompt(prompt)
    # _prompt = prompt
    print(_prompt)
    print('_prompt-------------------------')
    dist.barrier()
    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])

    if image is not None:
        image = image.cuda()
    with torch.cuda.amp.autocast(dtype=target_dtype):
        results = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    # text_output = results[0].strip()
    text_output = results[0].split('###')[0]
    if "Please use commandline to solve the math question:" in _prompt:
        print(text_output)
        answer_pred = exec_one(text_output)
        try:
            translated = translate_json_to_natural_language(text_output.strip())
        except:
            translated = text_output
        if answer_pred == None:
            text_output = translated
        else:
            # print(answer_pred)
            print(text_output)
            text_output = translated + '\n' + str(answer_pred)
            
    return text_output

def create_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column() as image_input:
                        img_path = gr.Image(label='Image Input ( Some Demos )', type='filepath')
                with gr.Row() as text_config_row:
                    max_gen_len = gr.Slider(minimum=1, maximum=512, value=512, interactive=True, label="Max Length")
                    # with gr.Accordion(label='Advanced options', open=False):
                    gen_t = gr.Slider(minimum=0, maximum=1, value=0.9, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.75, interactive=True, label="Top p")
                with gr.Row():
                    # clear_botton = gr.Button("Clear")
                    run_botton = gr.Button("Run", variant='primary')

            with gr.Column():
                with gr.Row():
                    prompt = gr.Textbox(lines=4, label="Question")
                # with gr.Row():
                #     question_input = gr.Textbox(lines=4, label="Question Input (Optional)")
                # with gr.Row():
                #     system_prompt = gr.Dropdown(choices=['alpaca', 'None'], value="alpaca", label="System Prompt")
                

                with gr.Row():
                    gr.Markdown("Output")
                with gr.Row():
                    text_output = gr.Textbox(lines=11, label='Text Out')

        examples = read_examples()
        gr.Examples(examples=examples, inputs=[img_path, prompt])
    # question_input = prompt
    # system_prompt = "alpaca"
    # print(question_input)
    # print(system_prompt)
    inputs = [
        img_path,
        prompt, 
        max_gen_len, gen_t, top_p,
    ]
    outputs = [text_output]
    run_botton.click(fn=generate, inputs=inputs, outputs=outputs)

    return demo


def worker_func():
    while True:
        dist.barrier()

        input_data = [None for _ in range(5)]
        dist.broadcast_object_list(input_data)
        _prompt, image, max_gen_len, gen_t, top_p = input_data
        if image is not None:
            image = image.cuda()
        with torch.cuda.amp.autocast(dtype=target_dtype):
            _ = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p, )


if dist.get_rank() == 0:
    description = f"""
    # ChartAssistant Demo 🚀
    """

    with gr.Blocks(theme=gr.themes.Default(), css="#pointpath {height: 10em} .label {height: 3em}") as DEMO:
        gr.Markdown(description)
        create_demo()
    DEMO.queue(api_open=True).launch(share=True,server_name='0.0.0.0')

else:
    worker_func()
