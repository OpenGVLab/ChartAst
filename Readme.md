# ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning

This is the PyTorch implementation of the paper **ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning**, the paper is available at https://arxiv.org/abs/2401.02384

We have developed ChartAssistant, which uses large-scale chart data to align and instruction tuning. The model possesses powerful mathematical computation capabilities and achieves state-of-the-art performance on multiple datasets without the need for downstream dataset finetuning.

**This repo is built upon [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main)**

## ChartSFT
We have released the ChartSFT in https://huggingface.co/datasets/FanqingM/ChartAssistant 🔥🔥🔥

## Note

- An online demo is avaliable at http://106.14.127.192:7861/ 🔥🔥🔥

- 5.20 We have released the ChartSFT in https://huggingface.co/datasets/FanqingM/ChartAssistant 🔥🔥🔥
  Here is some example:
  - MathQA:
    ```
    {
    "image": "/mnt/petrelfs/share_data/mengfanqing1/mengfanqing/plotQA/train/png/png/35651.png",
    "conversations": [
        {
            "from": "human",
            "value": "Please use commandline to solve the math question: What is the difference between the highest and the second highest % of children under 5 who received treatment of Malaria ?"
        },
        {
            "from": "gpt",
            "value": "{'step1': {'func1': 'select', 'arg1': 'malaria', 'output1': ['10.000', '0.900']}, 'step2': {'func2': 'numpy.sort', 'arg2': 'output1', 'output2': 'sorted_arr'}, 'step3': {'func3': 'getitem', 'arg3':['sorted_arr', '-1'], 'output3': 'maximum'}, 'step4': {'func4': 'numpy.delete', 'arg4': ['sorted_arr', '-1'], 'output4': 'new_arr'}, 'step5': {'func5': 'getitem', 'arg5': ['new_arr', '-1'], 'output5': 'second_maximum'}, 'step6': {'func6': 'numpy.subtract', 'arg6': ['maximum', 'second_maximum'], 'output6': 'diff'}}"
        }
    ]},
    ```
  - ReferQA:
  
    ```
    {
            "image": "/mnt/petrelfs/share_data/luquanfeng/referring_box/train/img/52342_1_arrow.png",
            "conversations": [
                {
                    "from": "human",
                    "value": "Please use commandline to solve the math question about the referring box: What is the median years of the legend represented by the blue arrow per amount(current us$)?"
                },
                {
                    "from": "gpt",
                    "value": "{'step1': {'func1': 'select', 'arg1': 'ecuador', 'output1': ['139370000.000', '160050000.000', '165410000.000', '174120000.000', '214220000.000']}, 'step2': {'func2': 'numpy.median', 'arg2': 'outpu
    t1', 'output2': 'median'}}"
                }
            ]
        },
    ```
  
  - Arxiv QA:
  
    ```
    {
          "image": "/mnt/petrelfs/share_data/mengfanqing1/mengfanqing/arxiv/Img_QA/vbar/1806.09244_table_0.png",
          "conversations": [
              {
                  "from": "human",
                  "value": "Please answer my question based on the chart: How many counties are included in each dataset?"
              },
              {
                  "from": "gpt",
                  "value": "The number of counties included in each dataset are as follows: \n- Brazil-Soybean: 1529 counties \n- US-Soybean: 1814 counties \n- US-Maize: 2204 counties."
              }
          ]
      },
    ```
  
    
  
    

- 2.15 We update the [paper](https://arxiv.org/abs/2401.02384),with better performance and more experiments and corrected experimental results.
  
- 1.11: The ChartAssistant, which has undergone two-stage training on ChartSFT, has been open-sourced. You can download it through the following link.
  - https://pan.baidu.com/s/1t0QPLDfULNovnYKtsQxjOQ  password: 10el
  - [HuggingFace](https://huggingface.co/FanqingM/ChartAssistant) : put consolidated.00-of-02.model.pth and consolidated.01-of-02.model.pth in one directory, and replace pretrained_path in the scipt as it. 


- 1.10: We update the paper(ChartAssistant.pdf), primarily making updates to the model, correcting some errors in the article, and providing more detailed explanations. 

## ChartAssisstant

Charts play a vital role in data visualization, understanding data patterns, and informed decision-making. However, their unique combination of graphical elements (e.g., bars, lines) and textual components (e.g., labels, legends) poses challenges for general-purpose multimodal models. While vision-language models trained on chart data excel in comprehension, they struggle with generalization. To address these challenges, we propose ChartAssistant, a chart-based vision-language model for universal chart comprehension and reasoning. ChartAssistant leverages ChartSFT, a comprehensive dataset covering diverse chart-related tasks with basic (e.g. bars and pies) and specialized (e.g. radars, and bubbles) chart types. It undergoes a two-stage training process, starting with pre-training on chart-to-table parsing to align chart and text, followed by multitask instruction-following fine-tuning. This approach enables ChartAssistant to achieve competitive performance across various chart tasks. **Experimental results demonstrate significant performance gains over the state-of-the-art UniChart and Chartllama method, especially outperforming them on real-world chart data with zero-shot setting.** 



![image-20240104143625786](./demo.png)

## Environment
It is same as [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main)

## Inference

replace pretrained_path as the pretrained model path
```
sh accessory/exps/finetune/mm/test.sh
# run accessory/single_turn_eval.py
```

## Training
```
sh accessory/exps/finetune/mm/chart.sh
# run accessory/main_finetune.py
```
## Gradio demo
```
sh accessory/demo/start.sh
```




## Concat
if you have any questions about this work, you can email Fanqing Meng using mengfanqing33@gmail.com or just by wechat: mfq2052063742

## To Do List

- [x] Create the git repository.

- [x] Open source the model and model weight.

- [x] Open source the inference script.

- [x] Open source the dataset (ChartSFT).

  

