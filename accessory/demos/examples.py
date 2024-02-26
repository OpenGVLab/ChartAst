import json
import os
from pprint import pprint
def read_examples():
    example = [
        # [f"/root/cloud_disk2/LLaMA2-Accessory/accessory/demos/examples/3.png", "Please summarize the chart."],
        [f"/root/cloud_disk2/LLaMA2-Accessory/accessory/demos/examples/15.png", "Please use commandline to solve the math question: What is the difference between the highest and the lowest Accuracy difference of Top and bottom 20th percentile PVI ?"],
        [f"/root/cloud_disk2/LLaMA2-Accessory/accessory/demos/examples/two_col_41201.png", "Generate the table of this chart."],
        [f"/root/cloud_disk2/LLaMA2-Accessory/accessory/demos/examples/qa_ood_5.png", "Please summarize the chart."],
        [f"/root/cloud_disk2/LLaMA2-Accessory/accessory/demos/examples/qa_ood_4.png", "What can you take away from considering whether to join RefinedWeb?"]
    ]
    
    return example

if __name__ == '__main__':
    examples = read_examples()
    pprint(examples)
    print(len(examples))

        
