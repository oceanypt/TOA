import re
from graphviz import Digraph
import sys 
import json
import numpy as np
from tqdm import tqdm

def parse_and_visualize_tree_data(tree_data, to_print=True):
    dot = Digraph(comment='Tree Visualization')
    nodes = {}
    edges = []
    node_details = {}
    no_model_nodes = {}

    # 正则表达式，分离节点信息和父节点信息
    pattern = r"-->\s*(?P<node_info>layer: \d+; (?:model: [^;]+; )?reward: [^;]+; visits: \d+) \((?:parent: (?P<parent_info>layer: \d+; (?:model: [^;]+; )?reward: [^;]+; visits: \d+)|root)\)"

    # 解析所有节点及其父节点
    for match in re.finditer(pattern, tree_data):
        node_info = match.group('node_info')
        parent_info = match.group('parent_info')
        node_id = node_info.replace(" ", "").replace(":", "").replace(";", "_")

        # 格式化节点标签以便更易读，并添加换行符
        formatted_node_label = node_info.replace("; ", "\\n").replace("model: ", "Model: ").replace("layer: ", "Layer: ").replace("reward: ", "Reward: ").replace("visits: ", "Visits: ")
        
        nodes[node_id] = formatted_node_label
        dot.node(node_id, formatted_node_label, style='filled', fillcolor='white', shape='box')

        if parent_info:
            parent_node_id = parent_info.replace(" ", "").replace(":", "").replace(";", "_")
            edges.append((parent_node_id, node_id))
            node_details[node_id] = {
                "label": formatted_node_label,
                "parent": parent_node_id
            }
        else:
            node_details[node_id] = {
                "label": formatted_node_label,
                "parent": None
            }

        # Track nodes without a model
        if "model" not in node_info:
            reward = float(re.search("reward: ([\d\.]+)", node_info).group(1))
            no_model_nodes[node_id] = reward

    # Determine the node with the highest reward without a model
    max_reward_node = max(no_model_nodes, key=no_model_nodes.get)

    # Find the path from root to this node
    path = []
    current_node = max_reward_node
    while current_node:
        path.append(current_node)
        current_node = node_details[current_node]["parent"]

    # Add edges to the graph and highlight the path
    for parent, child in edges:
        if parent in path and child in path:
            dot.edge(parent, child, color='red')  # Highlight the path in red
            dot.node(child, node_details[child]["label"], style='filled', fillcolor='lightgray', shape='box')  # Highlight nodes in red
        else:
            dot.edge(parent, child)

    # Highlight the root node if it is part of the path
    if path[0] in nodes:
        dot.node(path[0], node_details[path[0]]["label"], style='filled', fillcolor='lightgray', shape='box')

    if to_print:
        # Render the graph
        dot.render('tree_visualization', view=True)

        # Print the path for confirmation
        #print("Path from root to the highest reward node without a model:")
        
        models = []
        
        for node in path[::-1]:
            if "Model" in node_details[node]["label"]:
                #print(node_details[node]["label"].replace('\\n', ' '))
                models.append(node_details[node]["label"].replace('\\n', ' ').split("Model:")[1].split("Reward:")[0].strip())
        
        return "<-->".join(models)
        
                
                

        



    def used_same_model():
        judge = True
        last_model = None
        for node in path[::-1]:
            if "Model:" in node_details[node]["label"]:
                model = node_details[node]["label"].split('Model:')[1].split('Reward:')[0].strip()
                if last_model is None:
                    last_model = model
                elif last_model is not None and last_model != model:
                    judge = False
                    return judge
        return judge
    
    #print(path[0])
    r = float(node_details[path[0]]["label"].split('Reward:')[1].split('Visits:')[0].replace('\\n', '').strip())

    return used_same_model(), r


def count_reward(rewards):
    dict_rewards = {}
    for r in rewards:
        if r not in dict_rewards:
            dict_rewards[r] = 0
        dict_rewards[r] += 1
    return dict_rewards

    

data = open(sys.argv[1], 'r').readlines()
data = [
    json.loads(item) for item in data
]

# num_used_same_model = 0
# rewards = []
# for item in data:
#     judge, r = parse_and_visualize_tree_data(item['tree_info'], to_print=False)
#     if judge:
#         num_used_same_model += 1
#     rewards.append(r)
#     # if parse_and_visualize_tree_data(item['tree_info'], to_print=False):
#     #     num_used_same_model += 1
# print(f'Used different models for generation: {1 - num_used_same_model / len(data):.4}')
# print(f"Avg. Max Reward: {np.mean(rewards):.4}")



model_paths = {}
for item in tqdm(data):
    model_path = parse_and_visualize_tree_data(item['tree_info'], to_print=True)
    if model_path not in model_paths:
        model_paths[model_path] = 0
    model_paths[model_path] += 1
    
#print(model_paths)

# 按照value从大到小排序并输出
sorted_dict = sorted(model_paths.items(), key=lambda item: item[1], reverse=True)

# 输出结果
for key, value in sorted_dict:
    print(f"{key}: {value}")




# item_id = int(sys.argv[2])

# d = count_reward(data[item_id]['rewards'])
# sorted_d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
# print(sorted_d)

# tree_data = data[item_id]['tree_info']

# parse_and_visualize_tree_data(tree_data)
