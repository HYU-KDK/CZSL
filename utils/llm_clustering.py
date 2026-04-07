import argparse
import json
import os
import re

import torch
from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed. Please install it to run inference.")
    LLM, SamplingParams = None, None

def get_objects_from_dataset(dataset_name):
    dataset_path = DATASET_PATHS[dataset_name]
    dataset = CompositionDataset(dataset_path, phase='train', split='compositional-split-natural')
    return dataset.objs

def extract_json(text):
    # Try to find JSON block in the text
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try to parse the entire text as JSON
    try:
        return json.loads(text)
    except:
        pass
    
    return None

def run_clustering(objects, num_clusters=10, model_name="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=8):
    if LLM is None:
        raise RuntimeError("vLLM is required but not installed.")
        
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        enforce_eager=True
    )
    
    # 1. vLLM에 로드된 모델의 토크나이저 가져오기
    tokenizer = llm.get_tokenizer()
    
    prompt_text = f"""
You are an expert at semantic categorization and clustering.
I have a list of {len(objects)} object categories:
{', '.join(objects)}

Please group these objects into exactly {num_clusters} semantic super-concepts (clusters). 
Ensure that EVERY object is assigned to exactly ONE cluster. DO NOT put the same object in multiple clusters.
Provide a short, descriptive name for each super-concept (e.g., 'animals', 'vehicles', 'furniture').

Format your output STRICTLY as a JSON dictionary where the keys are the super-concept names and the values are lists of objects belonging to that super-concept.
Do not output any explanation, only the JSON block.
"""
    
    # 2. 프롬프트 메시지를 Chat 포맷으로 구조화
    messages = [
        {"role": "system", "content": "You are a helpful AI that outputs strictly valid JSON dictionaries and nothing else."},
        {"role": "user", "content": prompt_text}
    ]
    
    # 3. Llama-3 Instruct 포맷에 맞게 템플릿 적용 (중요!)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 4. 안전장치: max_tokens를 3000으로 상향 (잘림 방지)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=3000)
    
    # 생성 시 formatted_prompt 사용
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    result_text = outputs[0].outputs[0].text
    
    clusters = extract_json(result_text)
    
    if clusters is None:
        print("Failed to parse JSON from LLM output. Raw output:")
        print(result_text)
        return None
        
    return clusters
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str, default="mit-states")
    parser.add_argument("--num_clusters", help="number of clusters K", type=int, default=10)
    parser.add_argument("--model", help="vLLM model name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tp_size", help="tensor parallel size for vLLM", type=int, default=8)
    parser.add_argument("--output", help="output json file path", type=str, default="llm_clusters.json")
    
    args = parser.parse_args()
    
    print(f"Loading objects from dataset: {args.dataset}")
    objects = get_objects_from_dataset(args.dataset)
    print(f"Found {len(objects)} objects.")
    
    print(f"Running LLM clustering with model: {args.model}")
    clusters = run_clustering(
        objects, 
        num_clusters=args.num_clusters, 
        model_name=args.model, 
        tensor_parallel_size=args.tp_size
    )
    
    if clusters is not None:
        # Validate that all objects are covered
        clustered_objects = []
        for v in clusters.values():
            clustered_objects.extend(v)
            
        missing = set(objects) - set(clustered_objects)
        extra = set(clustered_objects) - set(objects)
        
        if missing:
            print(f"Warning: {len(missing)} objects were not clustered: {missing}")
            # Put missing objects in a 'miscellaneous' cluster
            if "miscellaneous" not in clusters:
                clusters["miscellaneous"] = []
            clusters["miscellaneous"].extend(list(missing))
            
        if extra:
            print(f"Warning: LLM hallucinated {len(extra)} extra objects: {extra}")
            
        # Convert super-concepts mapping to object -> super_concept mapping
        obj_to_cluster = {}
        for cluster_idx, (cluster_name, cluster_objs) in enumerate(clusters.items()):
            for obj in cluster_objs:
                if obj in objects:
                    obj_to_cluster[obj] = {
                        "cluster_idx": cluster_idx,
                        "cluster_name": cluster_name
                    }
                    
        with open(args.output, "w") as f:
            json.dump({
                "clusters": clusters,
                "obj_to_cluster": obj_to_cluster
            }, f, indent=4)
            
        print(f"Successfully saved clusters to {args.output}")