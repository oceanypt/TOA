import numpy as np
from reward import UltraRM, ArmoRM, Shepherd_MATH_PRM, InternRM
import random
import sys
import os
import queue
import time

def random_index(lst):
    ids = []
    for i, d in enumerate(lst):
        if d is not None:
            ids.append(i)
    if len(ids) == 0:
        print (f"No workable clients")
        sys.exit(0)
    random_id = random.randint(0, len(ids) - 1)
    
    return ids[random_id]

## =======>>>> Generation
def generate_vllm_api(args, prompt, clients, model_configs, extraction_template=None, max_attempt_time=20, interval=20):
    def call():
        c_id = random_index(clients)
        client = clients[c_id]
        model_config = model_configs[c_id]
    
        print(f"\n\n-----> Used the model: {model_config['path_to_model']}, key: {model_config['api_key']}\n\n")
    
        try:
            if isinstance(prompt, tuple) and len(prompt) == 2:
                messages = [{"role": "system", "content": prompt[0]}, {"role": "user", "content": prompt[1]}]
            else:
                messages = [{"role": "user", "content": prompt}]
        
            completion = client.chat.completions.create(
                model=model_config['path_to_model'],
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=model_config['stop_tokens'],
                #stop_token_ids = [128001, 128009],
                #frequency_penalty=0,
                #presence_penalty=0,
                #stop='',
                n=1,
                )

            response = {
                "response": completion.choices[0].message.content.strip(),
                "n_prompt_tokens": completion.usage.prompt_tokens,
                "n_completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
            
            print (f"\n\n-------->\n{prompt}\n***\n{response['response']}")
            
        except Exception as e:
            print(f"\n\nError: {e}\nModel:{model_config['path_to_model']}\n\n")
            response = None
        
        return response


    for _ in range(max_attempt_time):
        response = call()
        if response is not None:
            break
        time.sleep(interval)
    
    return response

## =======>>>> Ensemble
def sample_N_thread(args, task_queue, model_names, configs, policy_model_by_name, reward_model, output_queue):
    with open(args.path_to_translation_template, 'r') as f:
        translate_template = f.read()
    
    while True:
        try:
            input_item = task_queue.get(timeout=10)
        except queue.Empty:
            return  # If there are no tasks for an extended period, the thread exits.
        
        source_lang, action_names, temp_all_input_prompts = input_item[args.source], [], []
        
        input_prompt = translate_template.format(source = source_lang)
        
        for model_name in model_names:
            action_names += [ model_name ] * args.n_samples
            temp_all_input_prompts += [ input_prompt ] * args.n_samples
        random.shuffle(action_names)
        
        # get responses
        temp_all_results = [
            generate_vllm_api(args, input_prompt, policy_model_by_name[action_name], configs[action_name]) for action_name in action_names
        ]
        temp_all_responses = [temp['response'] for temp in temp_all_results]
        

        # cal rewards
        print (f"\n\n-----> calculating the rewards...")
        all_rewards = reward_model.cal_rewards( [source_lang] * (args.n_samples * len(model_names)), temp_all_responses  )
        
        # prepare the output
        input_item['responses'] = temp_all_responses
        input_item['actions'] = action_names
        input_item['rewards'] = all_rewards
        
        
        # record the token numbers
        input_item['n_prompt_tokens'] = [ temp['n_prompt_tokens'] for temp in temp_all_results ]
        input_item['n_completion_tokens']  = [ temp['n_completion_tokens'] for temp in temp_all_results]
        input_item['total_tokens'] = [ temp['total_tokens'] for temp in temp_all_results]
        

        input_item['best_reward'] = np.max(input_item['rewards'])
        input_item['best_response'] = input_item['responses'][np.argmax(input_item['rewards'])]
        input_item['best_model'] = input_item['actions'][np.argmax(input_item['rewards'])]

        ## return the new item
        output_queue.put(input_item)

        ## task is done
        task_queue.task_done()

## =======>>>> PRS
def PRS_thread(args, task_queue, model_names, configs, policy_model_by_name, reward_model, output_queue):
    assert len(model_names) == 1 # make sure there is only one model for generation
    with open(args.path_to_translation_template, 'r') as f:
        translate_template = f.read()
        
    with open(args.path_to_refine_template, 'r') as f:
        refine_template = f.read()
    
    while True:
        try:
            input_item = task_queue.get(timeout=10)
        except queue.Empty:
            return  
        
        source_lang, model_name = input_item[args.source], model_names[0]
        
        width_0 = int(args.n_samples / 2)
        width_1 = args.n_samples - width_0
        
        # 1. layer 0
        input_prompt = translate_template.format(source = source_lang)
        
        layer_0_results = [
            generate_vllm_api(args, input_prompt, policy_model_by_name[model_name], configs[model_name]) for _ in range(width_0)
        ]
        
        
        layer_0_rewards = reward_model.cal_rewards( [source_lang] * len(layer_0_results), [ temp['response'] for temp in layer_0_results] )
        
        best_response_in_layer_0 = layer_0_results[np.argmax(layer_0_rewards)]['response']
        
        # 2. layer 1
        ## pack the input prompt
        packed_input_prompt = refine_template.format(source = source_lang, translation = best_response_in_layer_0)
        layer_1_results = [
            generate_vllm_api(args, packed_input_prompt, policy_model_by_name[model_name], configs[model_name]) for _ in range(width_1)
        ]
        layer_1_rewards = reward_model.cal_rewards( [ source_lang ] * len(layer_1_results), [ temp['response'] for temp in layer_1_results ] )
        
        # 3. combine the results
        all_responses = [ temp['response'] for temp in layer_0_results ] + [ temp['response'] for temp in layer_1_results ]
        all_rewards = layer_0_rewards + layer_1_rewards
        
        # prepare the output
        input_item['responses'] = all_responses
        input_item['actions'] = [ f"{model_name}_layer_0" for _ in range(len(layer_0_results)) ] + [ f"{model_name}_layer_1" for _ in range(len(layer_1_results)) ]
        input_item['rewards'] = all_rewards
        
        input_item['best_reward'] = np.max(input_item['rewards'])
        input_item['best_response'] = input_item['responses'][np.argmax(input_item['rewards'])]
        input_item['best_model'] = input_item['actions'][np.argmax(input_item['rewards'])]
        
        # record the token numbers
        input_item['n_prompt_tokens'] = [ temp['n_prompt_tokens'] for temp in layer_0_results ] + [ temp['n_prompt_tokens'] for temp in layer_1_results ]
        input_item['n_completion_tokens']  = [ temp['n_completion_tokens'] for temp in layer_0_results] + [ temp['n_completion_tokens'] for temp in layer_1_results]
        input_item['total_tokens'] = [ temp['total_tokens'] for temp in layer_0_results] + [ temp['total_tokens'] for temp in layer_1_results]

        ## return the new item
        output_queue.put(input_item)

        ## task is done
        task_queue.task_done()

def Seq_thread(args, task_queue, model_names, configs, policy_model_by_name, reward_model, output_queue):
    with open(args.path_to_translation_template, 'r') as f:
        translate_template = f.read()
        
    with open(args.path_to_refine_template, 'r') as f:
        refine_template = f.read()
    
    while True:
        try:
            input_item = task_queue.get(timeout=10)
        except queue.Empty:
            return  

        source_lang = input_item[args.source]
        
        responses, action_names = [], []
        model_ids = list(range(len(model_names)))
        
        gen_count = 0
        for round_id in range(0, int(args.n_samples * len(model_names)), len(model_names)):
            random.shuffle(model_ids)
            
            for id_iter in range(len(model_names)):
                if gen_count >= args.n_samples * len(model_names):
                    break
                
                temp_input_prompt = translate_template.format(source = source_lang)  #input_prompt
                
                if id_iter > 0:
                    temp_input_prompt = refine_template.format(source = source_lang, translation = responses[-1]['response'] )
                
                model_name = model_names[ model_ids[id_iter]]
                action_names.append( f"{model_name}.round_{round_id}_iter_{id_iter}" )
                
                responses.append(
                    generate_vllm_api(args, temp_input_prompt, policy_model_by_name[model_name], configs[model_name])
                )
    
                gen_count += 1

        input_item['responses'] = [ temp['response'] for temp in responses ]
        rewards = reward_model.cal_rewards([ source_lang ] * len(input_item['responses']), input_item['responses'])
        input_item['rewards'] = rewards

        input_item['actions'] = action_names
        input_item['best_reward'] = np.max(rewards)
        input_item['best_response'] = input_item['responses'][np.argmax(rewards)]
        input_item['best_model'] = action_names[np.argmax(rewards)]
        
        input_item['n_prompt_tokens'] = [ temp['n_prompt_tokens'] for temp in responses]
        input_item['n_completion_tokens'] = [ temp['n_completion_tokens'] for temp in responses]
        input_item['total_tokens'] = [ temp['total_tokens'] for temp in responses]
        
        output_queue.put(input_item)
        ## task is done
        task_queue.task_done()



## =======>>>> MoA
def aggregate_responses(args, responses):
    if args.num_aggregation < len(responses):
        random.shuffle(responses)
    packed_response = ''
    for i, response in enumerate(responses[:args.num_aggregation]):
        packed_response += f'## Translation {i+1}:\n{response}\n\n'
    
    packed_ = packed_response.strip()
    
    print (f'{packed_}\n\n---------------------')
    
    return packed_

def MoA_thread(args, task_queue, model_names, configs, policy_model_by_name, reward_model, output_queue):
    with open(args.path_to_translation_template, 'r') as f:
        translate_template = f.read()
        
    with open(args.path_to_refine_template, 'r') as f:
        refine_template = f.read()
    
    while True:
        try:
            input_item = task_queue.get(timeout=10)
        except queue.Empty:
            return  

        source_lang = input_item[args.source]

        
        responses_by_model = {model_name: [] for model_name in model_names}
        
        gen_count = 0
        for _ in range(0, args.n_samples, args.n_iter):
            for id_iter in range(args.n_iter):
                if gen_count >= args.n_samples:
                    break
                
                temp_input_prompt = translate_template.format(source = source_lang)  #input_prompt
                
                if id_iter > 0:
                    # aggregate the responses
                    c_responses = [responses_by_model[model_name][-1]['response'] for model_name in model_names]
                    temp_input_prompt = refine_template.format(source = source_lang, translation = aggregate_responses(args, c_responses) )
                
                temp_all_responses = [
                    generate_vllm_api(args, temp_input_prompt, policy_model_by_name[model_name], configs[model_name]) for model_name in model_names
                ]

                for c_i, model_name in enumerate(model_names):
                    responses_by_model[model_name].append(temp_all_responses[c_i])
    
                gen_count += 1


        input_item['responses'] = {model_name: [ responses_by_model[model_name][round_id]['response'] for round_id in range(args.n_samples) ] for model_name in model_names}

       

        temp_responses = []
        temp_rewards = []
        temp_actions = []
        for model_name, c_responses in input_item['responses'].items():
            temp_responses += c_responses
            gen_count = 0
            for _ in range(0, args.n_samples, args.n_iter):
                for id_iter in range(args.n_iter):
                    if gen_count >= args.n_samples:
                        break
                    temp_actions.append(
                            model_name + '_' + str(id_iter)
                        )
                    gen_count += 1   
            temp_rewards += reward_model.cal_rewards([source_lang] * len(c_responses), c_responses)
        
        temp_n_prompt_tokens = []
        temp_n_completion_tokens = []
        temp_total_tokens = []
        for model_name in model_names:
            temp_n_prompt_tokens += [ 
                responses_by_model[model_name][round_id]['n_prompt_tokens'] for round_id in range(args.n_samples) 
            ]
            
            temp_n_completion_tokens += [
                responses_by_model[model_name][round_id]['n_completion_tokens'] for round_id in range(args.n_samples)
            ]
            
            temp_total_tokens += [
                responses_by_model[model_name][round_id]['total_tokens'] for round_id in range(args.n_samples)
            ]
        
        input_item['rewards'] = temp_rewards
        input_item['actions'] = temp_actions
        input_item['best_reward'] = np.max(temp_rewards)
        input_item['best_response'] = temp_responses[np.argmax(temp_rewards)]
        input_item['best_model'] = temp_actions[np.argmax(temp_rewards)]
        input_item['responses'] = temp_responses
        
        input_item['n_prompt_tokens'] = temp_n_prompt_tokens
        input_item['n_completion_tokens'] = temp_n_completion_tokens
        input_item['total_tokens'] = temp_total_tokens
        
        output_queue.put(input_item)
        ## task is done
        task_queue.task_done()

## =======>>>> MCTS
class Node():
    def __init__(self, args, layer_id, model_name, response, parent = None, model_names = None, input_prompt = None, input_response = None, node_type = '__model__'):
        ## Node info
        self.args = args
        self.layer_id = layer_id
        self.model_name = model_name # if node is '__response__', then model_name is None
        self.response = response # if node is '__model__', then response is None
        self.parent = parent
        self.model_names = model_names # the model names used for sampling
        self.input_prompt = input_prompt
        self.input_response = input_response
        self.node_type = node_type
        
        ## Node statistics
        self.visits = 1
        self.rewards = []
        self.children = []
        self.response_reward = None ##

        ## expand the node if it is root or response
        if self.node_type == '__root__' or self.node_type == '__response__':
            for model_name in self.model_names:
                self.add_child(layer_id = self.layer_id + 1,
                               model_name = model_name,
                               response = None,
                               input_response = response,
                               node_type = '__model__')

    def add_child(self, layer_id, model_name, response, input_response, node_type):
        child_node = Node(
                args=self.args,
                layer_id=layer_id,
                model_name=model_name,
                response=response,
                parent=self,
                model_names=self.model_names,
                input_prompt=self.input_prompt,
                input_response=input_response,
                node_type=node_type
            )

        self.children.append(
            child_node
        )
    
    def update(self, reward):
        self.rewards.append(reward)
        self.visits += 1
    
    def fully_expanded(self):
        if self.node_type == '__response__' or self.node_type == '__root__':
            # if node is response or root, then it has been fully expanded
            return True
        if self.node_type == '__model__':
            # if node is model
            #   if it has not reached the maximal width, then not fully expanded.
            #   else if it has reached the maximal width, 
            #       if it has the child node with higher reward than parent, then fully expanded
            #       otherwise, it has not fully expanded.
            if not self.reached_width():
                return False
            else:
                if self.child_better_than_parent():
                    return True
                else:
                    return False
        
    def reached_width(self):
        if self.cal_expanded_width() >= self.args.width:
            return True
        else:
            return False

    def cal_expanded_width(self):
        assert self.node_type == '__model__'

        root_node = self.get_root_node()
        expanded_width = self.sum_children_at_layer(root_node, self.layer_id)  #self.cal_child_num_of_layer(self, root_node, self.layer_id)
        return expanded_width
    
    def get_root_node(self):
        current_node = self
        while current_node.parent != None:
            current_node = current_node.parent
        return current_node

    def sum_children_at_layer(self, root, target_layer):
        if target_layer < 0:
            return 0
    
        queue = [(root, 0)]  # (node, layer)
        total_children_count = 0

        while queue:
            node, layer = queue.pop(0)
        
            if layer == target_layer:
                total_children_count += len(node.children)
            elif layer < target_layer:
                for child in node.children:
                    queue.append((child, layer + 1))

        return total_children_count

    
    def child_better_than_parent(self):
        # current node is model
        # response (parent) --> model --> response (children)

        assert self.node_type == '__model__'
        
        if self.parent.node_type == '__root__':
            return True

        parent_reward = self.parent.response_reward
        assert parent_reward != None

        children_reward = [ child.response_reward for child in self.children ]

        if len(children_reward) == 0:
            return False

        if max(children_reward) > parent_reward:
            return True
        else:
            return False
    
    def get_node_info(self):
        reward = 0 if len(self.rewards) == 0 else np.mean(self.rewards)
        if self.model_name is not None:
            return f"--> Node; layer_id: {self.layer_id}; model_name: {self.model_name}; reward: {reward}; visits: {self.visits}"
        else:
            return f"--> Node; layer_id: {self.layer_id}; reward: {reward}; visits: {self.visits}"

def print_tree_indent(node, parent_value=None, level=0):
    prefix = ' ' * 4 * level
    info = ''
    if parent_value is None:
        info += f"{prefix}--> layer: {node.layer_id}; reward: {np.mean(node.rewards):.4}; visits: {node.visits} (root)\n"
    else:
        if node.model_name is None:
            info += f"{prefix}--> layer: {node.layer_id}; reward: {node.response_reward:.4}; visits: {node.visits} (parent: {parent_value})\n"
        elif node.model_name is not None and len(node.rewards) != 0 and len(node.children) != 0:
            info += f"{prefix}--> layer: {node.layer_id}; model: {node.model_name}; reward: {np.mean(node.rewards):.4}; visits: {node.visits} (parent: {parent_value})\n"
    for child in node.children:
        if node.node_type == '__root__':
            info += print_tree_indent(child, f"layer: {node.layer_id}; reward: {np.mean(node.rewards):.4}; visits: {node.visits}", level + 1)
        elif node.model_name == None:
            info += print_tree_indent(child, f"layer: {node.layer_id}; reward: {node.response_reward:.4}; visits: {node.visits}", level + 1 )
        elif node.model_name is not None and len(node.children) != 0:
            info += print_tree_indent(child, f"layer: {node.layer_id}; model: {node.model_name}; reward: {np.mean(node.rewards):.4}; visits: {node.visits}", level + 1 )
    
    return info
        
def MCTS_Best_Child(args, children, n_played):
    if children[0].node_type == '__response__':
        ## if node_type == '__response__': keep the top-1 best
        response_rewards = [ child.response_reward for child in children ]
        
        temp_topk_child = args.topk_child if len(response_rewards) >= args.topk_child else len(response_rewards)
        top_k_indices = np.argpartition(response_rewards, -temp_topk_child)[-temp_topk_child:]
        
        children = [ children[id] for id in top_k_indices ]
    
    best_child, max_reward = None, None
    for child in children:
        if child.visits == 1 and len(child.rewards) == 0:
            c_rewrd = float('inf')
        else:
            # We use the global played number instead of the played number at the sub tree
            c_rewrd = np.mean(child.rewards) + args.alpha * np.sqrt(2 * np.log(n_played) / child.visits)
        if best_child is None:
            best_child = child
            max_reward = c_rewrd
        elif c_rewrd > max_reward:
            best_child = child
            max_reward = c_rewrd
    
    return best_child
    
def MCTS_Selection(args, root_node, n_played):
    ## return the model name for generation
    current_node = root_node

    while True:
        if current_node.node_type == '__response__' and current_node.visits == 1:
            # add the nodes with models, but set the input_resposne as None
            for c_child in root_node.children:
                current_node.add_child(layer_id = current_node.layer_id + 1,
                                       model_name = c_child.model_name,
                                       response = None,
                                       input_response = None,
                                       node_type = '__model__'
                                       )
                current_node.children[-1].visits = c_child.visits
                current_node.children[-1].rewards = list(c_child.rewards) ## deep copy

        if not current_node.fully_expanded():
            return current_node
        else:
            children = current_node.children
            current_node = MCTS_Best_Child(args, children, n_played)


def MCTS_Backup(node, reward):
    while node != None:
        node.visits += 1
        node.rewards.append(reward)    
        node = node.parent

def MCTS_thread(args, task_queue, model_names, configs, policy_model_by_name, reward_model, output_queue):
    with open(args.path_to_translation_template, 'r') as f:
        translate_template = f.read()
     
    with open(args.path_to_refine_template, 'r') as f:
        refine_template = f.read()   
    
    ## =======>>> sample responses using MCTS
    def sigmoid(x, tau=5):
        # normalize the rewards
        return 1 / (1 + np.exp(-x / tau))
    
    while True:
        try:
            input_item = task_queue.get(timeout=10)
        except queue.Empty:
            return  
    
        source_lang = input_item[args.source]
        
        
        ## build the trees
        tree = Node(args = args,
                   layer_id = 0,
                   model_name = None,
                   response = None,
                   parent = None,
                   model_names = model_names,
                   input_prompt = source_lang,
                   input_response = None,
                   node_type = '__root__')
        
        total_plays = args.n_samples * len(model_names)
    
        for play_id in range(total_plays):
            ## 1. Search for the node
            c_node = MCTS_Selection(args = args,
                                    root_node=tree,
                                    n_played=play_id+1
                                    )
        
            print(f"\n\nPlay Round: {play_id} (id: {input_item['id']})", flush=True)
            print(c_node.get_node_info())
        
            ## 2. pack the input prompts
            if c_node.input_response is None:
                packed_input_prompt = translate_template.format(source = source_lang)  #input_prompt
            else:
                packed_input_prompt = refine_template.format(source = c_node.input_prompt, translation = c_node.input_response)

                
            ## 3. generate responses
            action_name = c_node.model_name  # get the model name for sampling
            response = generate_vllm_api(args, packed_input_prompt, policy_model_by_name[action_name], configs[action_name])
            
            
            ## 4. calculate rewards for the responses
            reward = reward_model.cal_rewards([source_lang], [response['response']])[0]
            print (f"\n\n\n-------------> Reward: {reward}\n\n\n")
        
            ## 4.1 normalize the rewards
            tree_reward = sigmoid(reward, args.tau)
        
            ## 5. expand the node
            c_node.add_child(layer_id = c_node.layer_id+1,
                                     model_name=None,
                                     response=response['response'],
                                     input_response=None,
                                     node_type='__response__')

            ## 5.1 assign the reward for the expanded node
            c_node.children[-1].response_reward = tree_reward
        
            ## 6. Backpropagation
            MCTS_Backup(c_node, tree_reward)
            print(f"\n\nFinished --> Play Round: {play_id} (id: {input_item['id']})", flush=True)

            ## 7. update responses and estimations
            if 'actions' not in input_item:
                input_item['actions'] = []
            if 'responses' not in input_item:
                input_item['responses'] = []
            if 'rewards' not in input_item:
                input_item['rewards'] = []
            if 'n_prompt_tokens' not in input_item:
                input_item['n_prompt_tokens'] = []
            if 'n_completion_tokens' not in input_item:
                input_item['n_completion_tokens'] = []
            if 'total_tokens' not in input_item:
                input_item['total_tokens'] = []

            true_action_name = c_node.model_name + '_layer_' + str(c_node.layer_id)
            input_item['actions'].append(true_action_name)
            input_item['responses'].append(response['response'])
            input_item['rewards'].append(reward)
            
            input_item['n_prompt_tokens'].append(response['n_prompt_tokens'])
            input_item['n_completion_tokens'].append(response['n_completion_tokens'])
            input_item['total_tokens'].append(response['total_tokens'])
        
            
        best_reward = np.max(input_item['rewards'])
        best_id = np.argmax(input_item['rewards'])

        
        input_item['best_response'] = input_item['responses'][best_id]
        input_item['best_model'] = input_item['actions'][best_id]
        input_item['best_reward'] = best_reward
        input_item['tree_info'] = print_tree_indent(tree) 
    

        output_queue.put(input_item)
    
        ## task is done
        task_queue.task_done()
