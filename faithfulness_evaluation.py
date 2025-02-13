import boto3
import json
from parsers import *
from utils import read_file
from prompts import *
import random
import copy
import numpy as np
import time
from datasets import load_dataset
import argparse


bedrock_rt = boto3.client("bedrock-runtime", "us-west-2")

llama_model_id = "meta.llama3-70b-instruct-v1:0"
claude3_sonnet_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
claude3_haiku_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
claude3_opus_model_id = "anthropic.claude-3-opus-20240229-v1:0:200k"
llama_8b_model_id = "meta.llama3-8b-instruct-v1:0"
mistral_model_id = "mistral.mixtral-8x7b-instruct-v0:1"

def llama_prediction(bedrock, prompt, model_id, temperature):
    prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    body = {
        "prompt": prompt,
        "max_gen_len": 1000,
        "temperature": temperature,
        "top_p": 0.9,
    }

    body_bytes = json.dumps(body).encode('utf-8')
    inputs = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": body_bytes
    }
    ret = bedrock.invoke_model(**inputs)
    resp_body = ret["body"].read()
    resp_body_json = json.loads(resp_body.decode('utf-8'))

    return resp_body_json["generation"]

def llama_expl_prediction(bedrock, prompt, model_id, temperature):

    body = {
        "prompt": prompt,
        "max_gen_len": 1000,
        "temperature": temperature,
        "top_p": 1.0
    }

    body_bytes = json.dumps(body).encode('utf-8')
    inputs = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": body_bytes
    }
    ret = bedrock.invoke_model(**inputs)
    resp_body = ret["body"].read()
    resp_body_json = json.loads(resp_body.decode('utf-8'))

    return resp_body_json["generation"]

def claude3_prediction(bedrock, prompt, model_id, temperature):
    user_message =  {"role": "user", "content": prompt}
    messages = [user_message]
    body=json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": "",
            "messages": messages,
            "temperature": temperature,
        }  
    )  
    response = bedrock.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
   
    return response_body["content"][0]["text"]

def mistral_prediction(bedrock, prompt, model_id, temperature):
    body=json.dumps(
        {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": temperature,
            "top_p": 1.0
        }  
    )  
    response = bedrock.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
   
    return response_body["outputs"][0]['text']

## list of modules for evaluation using different settings
def ambiguity(annotated_file, output_file):
    '''
    This module can be used for the ambiguity detection task. There are two variations that can be used:
    1. The baseline ambiguity detection that receives the doc, claim and provides a response
    2. The variant that uses the generated arguments from agents and use them along with doc and claim
    '''
    ## Uncomment the following line to load the file with generated arguments from the debate for the second setup
    # args_file = read_file(args_file)
    instances = read_file(annotated_file)
    tp, tn, fp, fn = 0, 0, 0, 0
    w = open(output_file,'w')
    counter = 0
    for i, instance in enumerate(instances):
        results = {}
        print(counter)
        counter += 1
        doc = instance['doc']
        sent = ' '.join(instance['claim'].split('\n'))
        print(sent)

        ## For the settings that we have used the generated agents arguments to identify ambiguity, you 
        ## should first read the file with agents responses in the args_file and then uncomment the
        ## following line to extract agents arguments and then use it as the input to the corresponding 
        ## prompt module like this: get_prompt_ambiguity_w_args(doc,summary,args)

        # args = [exp for exp_list in instance['explanation'] for exp in exp_list]
        # get_prompt_ambiguity_w_args(doc,summary,args)

        prompt = get_prompt_ambiguity(doc, sent)
        response = llama_prediction(bedrock_rt,prompt, llama_model_id, 1.0)
        print(response)
        label , category,  explanation = parse_output_response_w_category(response)
        if label != "Unknown":
            label_val = float(label)
            if label_val >= 0.5:
                label = 1
            else:
                label = 0
        else:
            label = -1

        if label == 1 and instance['manual_sent_label'] == -1:
            tn += 1

        elif label == 0 and instance['manual_sent_label'] != -1:
            tp += 1

        elif label == 1 and instance['manual_sent_label'] != -1:
            fn += 1
        
        elif label == 0 and instance['manual_sent_label'] == -1:
            fp += 1
        
        results['pred_sent_label'] = label
        results['sent_label'] = instance['manual_sent_label']
        results['category'] = category
        results['explanation'] = explanation
        w.write(json.dumps(results, ensure_ascii=False) + '\n') 

    print(tp, tn, fp, fn)
    fpr = fp/(fp+tn)
    print("FPR: ",fpr)
    fnr = fn/(fn+tp)
    print("FNR: ",fnr)
    print("BAcc (Main): ", 1-((fpr+fnr)/2))

## module for direct baseline
def single_llm(dataset_name, output_file, data_file, benchmark):
    '''
    This setup is used for zero-shot inferences. 
    We load one of the 4 datasets and then provide the document and the claim for evaluation. 
    The datasets used in this work are all part of the LLM-AggreFact benchmark. 
    The TofuEval benchmark was used both on sentence-level and summary-level so if you want to try the second variant
    or any other datasets you have to load your own data with keys: 'doc', 'claim' and 'label' with value 1 or 0.
    '''
    
    if benchmark:
        instances = []
        dataset = load_dataset("lytang/LLM-AggreFact")
        for data in dataset['test']:
            if data['dataset'] == dataset_name:
                instances.append({'dataset':data['dataset'], 'doc': data['doc'], 'claim': data['claim'], 'label': str(data['label'])})
    
    else:
        instances = read_file(data_file)

    tp, tn, fp, fn = 0, 0, 0, 0
    w = open(output_file,'w')
    counter = 0
    for instance in instances:
        results = {}
        print(counter)
        counter += 1
        doc = instance['doc']
        sent = ' '.join(instance['claim'].split('\n'))
        print(sent)
        prompt = get_prompt_direct_eval_individual_sentences(doc, sent)
        response = llama_prediction(bedrock_rt,prompt, llama_model_id, 1.0)
        print(response)
        label , explanation = parse_output_response(response)
        if label != "Unknown":
            label_val = float(label)
            if label_val >= 0.5:
                label = 1
            else:
                label = 0
        else:
            label = -1

        if label == 1 and instance['label'] == "1":
            tn += 1

        elif label == 0 and instance['label'] == "0":
            tp += 1

        elif label == 1 and instance['label'] == "0":
            fn += 1
        
        elif label == 0 and instance['label'] == "1":
            fp += 1
        
        if benchmark:
            results['dataset'] = instance['dataset']
        else:
            results['dataset'] = 'sample' ## specify the name of the dataset you are using
        results['pred_sent_label'] = label
        results['sent_label'] = instance['label']
        results['explanation'] = explanation
        w.write(json.dumps(results, ensure_ascii=False) + '\n') 

    print(tp, tn, fp, fn)
    fpr = fp/(fp+tn)
    print("FPR: ",fpr)
    fnr = fn/(fn+tp)
    print("FNR: ",fnr)
    print("BAcc (Main): ", 1-((fpr+fnr)/2))

## module for Chain of Thought and self-consistency baselines
def single_llm_cot(dataset_name, output_file, num_rep, data_file, benchmark):
    '''
    This setup is used for chain of thought and self-consistency inferences. The num_rep parameter should be set
    to the number of times you'd like to repeat the inference before majority voting.

    We load one of the 4 datasets and then provide the document and the claim for evaluation. 
    The datasets used in this work are all part of the LLM-AggreFact benchmark. 
    The TofuEval benchmark was used both on sentence-level and summary-level so if you want to try the second variant
    or any other datasets you have to load your own data with keys: 'doc', 'claim' and 'label' with value 1 or 0.
    '''
    if benchmark:
        instances = []
        dataset = load_dataset("lytang/LLM-AggreFact")
        for data in dataset['test']:
            if data['dataset'] == dataset_name:
                instances.append({'dataset':data['dataset'], 'doc': data['doc'], 'claim': data['claim'], 'label': str(data['label'])})
    else:
        instances = read_file(data_file)

    tp, tn, fp, fn = 0, 0, 0, 0
    w = open(output_file,'w')
    counter = 0
    ## number of n for self-consistency
    num_rep = num_rep
    for instance in instances:
        label_list = []
        results = {}
        print(counter)
        counter += 1
        doc = instance['doc']
        sent = instance['claim']
        sent = ' '.join(instance['claim'].split('\n'))
        print(sent)
        prompt = get_prompt_cot(doc, sent)
        for i in range(num_rep):
            response = llama_prediction(bedrock_rt,prompt, llama_model_id, 1.0)
            # print(response)
            label , explanation = parse_output_response(response)
            if label != "Unknown":
                label_val = float(label)
                if label_val >= 0.5:
                    label = 1
                else:
                    label = 0
            else:
                label = -1
            label_list.append(label)
        
        if label_list.count(1) >= label_list.count(0):
            label = 1
        else:
            label = 0
        print(label_list)

        if label == 1 and instance['label'] == "1":
            tn += 1

        elif label == 0 and instance['label'] == "0":
            tp += 1

        elif label == 1 and instance['label'] == "0":
            fn += 1
        
        elif label == 0 and instance['label'] == "1":
            fp += 1
        
        if benchmark:
            results['dataset'] = instance['dataset']
        else:
            results['dataset'] = 'sample' ## specify the name of the dataset you are using
        results['pred_sent_label'] = label
        results['sent_label'] = instance['label']
        results['explanation'] = explanation
        w.write(json.dumps(results, ensure_ascii=False) + '\n') 

    print(tp, tn, fp, fn)
    fpr = fp/(fp+tn)
    print("FPR: ",fpr)
    fnr = fn/(fn+tp)
    print("FNR: ",fnr)
    print("BAcc (Main): ", 1-((fpr+fnr)/2))

## module for simultaneous debate
def simultaneous_debate_optional_adjudicator(dataset_name, output_file, num_debates, num_rounds, data_file, benchmark):
    '''
    This setup is used for the debate setup. num_debates should be set to determine how many simultaneous 
    debates you want and num_rounds represent the number of conversation turn in each debates. 

    We load one of the 4 datasets and then provide the document and the claim for evaluation. 
    The datasets used in this work are all part of the LLM-AggreFact benchmark. 
    The TofuEval benchmark was used both on sentence-level and summary-level so if you want to try the second variant
    or any other datasets you have to load your own data with keys: 'doc', 'claim' and 'label' with value 1 or 0.
    '''
    if benchmark:
        instances = []
        dataset = load_dataset("lytang/LLM-AggreFact")
        for data in dataset['test']:
            if data['dataset'] == dataset_name:
                instances.append({'dataset':data['dataset'], 'doc': data['doc'], 'claim': data['claim'], 'label': str(data['label'])})
        print(len(instances))

    else:
        instances = read_file(data_file)
    
    ## if reading data from a json file for a different dataset, uncomment below
    # instances = read_file(source_file)
    tp, tn, fp, fn = 0, 0, 0, 0
    w = open(output_file,'w')
    counter = 0

    for i, instance in enumerate(instances):
        # number of simultaneous debates for evaluation
        num_debates = num_debates
        eval_repeat_max = 0

        ## initilaize a dictionary to save the outputs of each separate debate
        debates_dict = dict.fromkeys([0,1,2],None)
        counter += 1
        overall_ambiguity = False

        ## keep starting debates until you reach the max numer of debates
        while eval_repeat_max != num_debates:
            ambiguous = False
            results = {}
            print(counter)
            doc = instance['doc']
            sent = ' '.join(instance['claim'].split('\n'))

            ## intial stance assignment. We use the follwoing list of utterances as the first reponse of each agent and then use 
            ## this as the chat history to start the debate. The default value is 4. You can change the number of agents by adding
            ## more utterances
            agents_responses = ["The summary is factually consistent with the document.", "The summary is factually inconsistent with the document.", "The summary is factually consistent with the document.", "The summary is factually inconsistent with the document."]#,"The summary is factually consistent with the document.", "The summary is factually inconsistent with the document."]
            updated_responses = []

            ## to keep track of previous responses of agents and provide them in each round
            message_board = ['','','','']

            ## intialize a label list to keep track of agents judgements
            label_list = [[1],[0],[1],[0]]
            all_chats = []

            ## number of rounds of debates
            turns = num_rounds
            
            mediator_feedback = ""
            ## first round of random assessment not included in the history.
            for n in range(len(agents_responses)):
                chat_history = ""
                chat_history_prompt = ''
                chat_history_prompt +=  message_board[n] + "You (Agent %s): "%str(n+1) + agents_responses[n] + "\n"
                chat_history += "You (Agent %s): "%str(n+1) + agents_responses[n] + "\n"
                other_agents_response = ""
                for nn in range(len(agents_responses)):
                    if nn != n:
                        other_agents_response += "Agent %s: "%str(nn+1) + agents_responses[nn] + "\n"
                        chat_history += "Agent %s: "%str(nn+1) + agents_responses[nn] + "\n"
                
                message_board[n] += chat_history
                chat_history_prompt += other_agents_response
                
                ## For experiments wo initial stance uncomment the following line to clear the chat history
                # chat_history_prompt = ""

                ## the parameters to prompt module include the document, the claim sentence, previous chat history and mediator feedback
                ## that you can use to modify the goals of agents
                prompt = get_prompt_debate(doc, sent, chat_history_prompt, mediator_feedback)
                argument = ""
                rep_ctr = 0
                label = -1
                label_val = -1

                ## to make sure we have enough initial diversity in responses, we repeat the following such that if the immediate
                ## response is different from the assigned stance, the agent is asked to repeat its generation. The rep_ctr is used
                ## to repaet 2 times before moving on to the next stage
                while label!="Unknown" and label_val != label_list[n][0] and rep_ctr != 2:
                    llm_response = llm_response = llama_prediction(bedrock_rt, prompt, llama_model_id, 1)
                    argument, label = parse_output_w_chat_label(llm_response)
                    rep_ctr += 1

                    ## the generated label might not be in correct format so we use the following to make sure the label format is correct
                    if label != "Unknown":
                        if len(label.split()) != 0 and ',' not in label.split()[0]:
                            label_val = float(label.split()[0])
                        elif len(label.split()) == 0 or ',' in label.split()[0]: 
                            if len(label.split(',')) != 0:
                                label_val = float(label.split(',')[0])
                            else: 
                                label_val = float(label)

                        if label_val >= 0.5:
                            label_val = 1
                        else:
                            label_val = 0        
                
                if label != "Unknown":
                    if len(label.split()) != 0 and ',' not in label.split()[0]:
                        label_val = float(label.split()[0])
                    elif len(label.split()) == 0 or ',' in label.split()[0]: 
                        if len(label.split(',')) != 0:
                            label_val = float(label.split(',')[0])
                        else: 
                            label_val = float(label)

                    if label_val >= 0.5:
                        label_list[n].append(1)
                    else:
                        label_list[n].append(0)
                else:
                    label_list[n].append(label_list[n][-1])
                argument = argument.strip()

                updated_responses.append(argument)
            agents_responses = copy.deepcopy(updated_responses)


            ## Once the first round is generated, we start the debate among agents
            message_board = ['','','','']
            for ag, ag_resp in enumerate(agents_responses):
                all_chats.append("Agent %s:\n"%str(ag+1) + ag_resp)
            
            mediator_feedback = ""

            ## The debate is continued for "turns" time.
            for cnt in range(turns):
                if len(set([lbl_list[-1] for lbl_list in label_list])) == 1:
                    break
                updated_responses = []
                for n in range(len(agents_responses)):
                    chat_history = ""
                    chat_history_prompt = ''
                    chat_history_prompt +=  message_board[n] + "You (Agent %s): "%str(n+1) + agents_responses[n] + "\n"
                    chat_history += "You (Agent %s): "%str(n+1) + agents_responses[n] + "\n"
                    other_agents_response = ""
                    for nn in range(len(agents_responses)):
                        if nn != n:
                            other_agents_response += "Agent %s: "%str(nn+1) + agents_responses[nn] + "\n"
                            chat_history += "Agent %s: "%str(nn+1) + agents_responses[nn] + "\n"
                    
                    message_board[n] += chat_history
                    chat_history_prompt += other_agents_response

                    ## to shuffle the order of chat history to remove any biases caused by order of chats
                    new_chat_history_list = []
                    chat_history_prompt_list = chat_history_prompt.split('\n')
                    chat_history_prompt_list = [chat_hist for chat_hist in chat_history_prompt_list if chat_hist != ""]
                    for pq in range(0,len(chat_history_prompt_list),len(agents_responses)):
                        shuffled_list = chat_history_prompt_list[pq:pq+len(agents_responses)]
                        random.shuffle(shuffled_list)
                        new_chat_history_list += shuffled_list
                    chat_history_prompt = '\n'.join(new_chat_history_list)
                    
                    ## you can add any type of feedback here and add them to prompt to improve the debate consensus
                    ## we do it after the first round
                    if cnt >= 1:
                        mediator_feedback = " Look back at the guidelines and how you have used them. Make sure all guidelines (and not only a subset of them) are satisfied in your assessment. Change your stance if you have made an error or if the other agents are more convincing."
                    
                    prompt = get_prompt_debate(doc, sent, chat_history_prompt,mediator_feedback)
                    llm_response = llama_prediction(bedrock_rt, prompt, llama_model_id, 1)
                    argument, label = parse_output_w_chat_label(llm_response)
                    if label != "Unknown":
                        if len(label.split()) != 0 and ',' not in label.split()[0]:
                            label_val = float(label.split()[0])
                        elif len(label.split()) == 0 or ',' in label.split()[0]: 
                            if len(label.split(',')) != 0:
                                label_val = float(label.split(',')[0])
                            else: 
                                label_val = float(label)

                        if label_val >= 0.5:
                            label_list[n].append(1)
                        else:
                            label_list[n].append(0)
                    else:
                        label_list[n].append(label_list[n][-1])
                    argument = argument.strip()

                    updated_responses.append(argument)
                    all_chats.append('Agent %s:\n'%str(n+1) + argument)
                agents_responses = copy.deepcopy(updated_responses)
                if len(set([lbl_list[-1] for lbl_list in label_list])) == 1:
                    break

            print(label_list)

            ## assign labels based on condifence: first remove agents that have a lot of variation (change stance more than once) and then
            ## how long an agent sticks to its stance to measure confidence. This setting is not reported in the paper.
            label_lists_clone = copy.deepcopy(label_list)
            label_lists_clone_proxy = [label_list_clone[1:] for label_list_clone in label_lists_clone]
            label_lists_clone = copy.deepcopy(label_lists_clone_proxy)
            all_common_list = []
            for agents_labels in label_lists_clone_proxy:
                switch = 0
                common = 0
                common_list = []
                for al in range(len(agents_labels)-1):
                    if agents_labels[al] != agents_labels[al+1]:
                        switch += 1
                        # common_list.append(common)
                        common = 0
                    else:
                        common += 1
                
                if switch > 1:
                    label_lists_clone.remove(agents_labels)
                
                else:
                    all_common_list.append(common+1)

            factuality_value = sum([1 * (lbl.count(lbl[-1])/len(lbl)) if lbl[-1]==1 else -1 * (lbl.count(lbl[-1])/len(lbl)) for c,lbl in enumerate(label_lists_clone)])
            print("factuality_value: ",[1 * (lbl.count(lbl[-1])/len(lbl)) if lbl[-1]==1 else -1 * (lbl.count(lbl[-1])/len(lbl)) for c,lbl in enumerate(label_lists_clone)])
            if sum([1 * (lbl.count(lbl[-1])/len(lbl)) if lbl[-1]==1 else -1 * (lbl.count(lbl[-1])/len(lbl)) for c,lbl in enumerate(label_lists_clone)]) > 0:
                weighted_label = 1
            elif sum([1 * (lbl.count(lbl[-1])/len(lbl)) if lbl[-1]==1 else -1 * (lbl.count(lbl[-1])/len(lbl)) for c,lbl in enumerate(label_lists_clone)]) < 0:
                weighted_label = -1
            else:
                weighted_label = "Unknown"

            pn_list = [lbl[-1] for lbl in label_list]
            debate_arguments = copy.deepcopy(all_chats[-len(agents_responses):])

            ## we record the outputs of the debate in a dictionary that was previously initialized.
            ## the "change" key keeps track of the number of agents who changes their stance during debate. 
            ## this can be used to identify the ambiguous cases directly.
            if pn_list.count(0) == pn_list.count(1):
                debates_dict[eval_repeat_max] = {'change': 0, 'label': -1,'arguments': debate_arguments,'labels': label_list, 'weighted_label': weighted_label, 'factuality_value': factuality_value}
                
                all_chats_dict = {}
                for n_agents in range(len(debate_arguments)):
                    all_chats_dict['Agent %s:'%str(n_agents+1)] = ""

                for cht_counter, cht in enumerate(debate_arguments):
                    all_chats_dict['Agent %s:'%str(cht_counter+1)] += ' '.join(cht.split('\n')[1:]) + ' '
                
                ## if there is not a winner label, we use adjudicators to decide on the final label.
                ## you can use multiple adjudicators if you want to do majority voting among them.
                adjudicator_input = [str(item) + ' ' + all_chats_dict[item] for item in all_chats_dict]
                adjudicator_prompt = get_adjudicator_prompt(doc, sent, '\n'.join(adjudicator_input))
                rep_counter = 0
                adjudicator_label_list = []
                label = ""
                explanation_list = []
                for i in range(1):
                    while label == "" and rep_counter != 5:
                        adjudicator_response = llama_prediction(bedrock_rt, adjudicator_prompt, llama_model_id, 1.0)
                        label ,  explanation  = parse_output_response(adjudicator_response)
                        explanation_list.append(explanation)
                        print(label,  explanation )
                        print('********')
                        if label != "Unknown":
                            if len(label.split()) != 0 and ',' not in label.split()[0]:
                                label_val = float(label.split()[0])
                            elif len(label.split()) == 0 or ',' in label.split()[0]: 
                                if len(label.split(',')) != 0:
                                    label_val = float(label.split(',')[0])
                                else: 
                                    label_val = float(label)
                            if label_val >= 0.5:
                                label = 1
                            else:
                                label = 0
                        else:
                            label = -1
                        rep_counter += 1
                    adjudicator_label_list.append(label)
                    label = ""
                
                if adjudicator_label_list.count(1) >= adjudicator_label_list.count(0):
                    label = 1
                else:
                    label = 0
                debates_dict[eval_repeat_max]['label'] = label

            ## if there is a winner label, we return the winner as the final label of the claim
            elif pn_list.count(0) != pn_list.count(1):
                if pn_list.count(1) >= pn_list.count(0):
                    label = 1
                else:
                    label = 0
                
                if len(set(pn_list)) == 1:
                    change = len(agents_responses)//2
                else:
                    change = len(agents_responses)//2 - 1
                debates_dict[eval_repeat_max] = {'change': change, 'label': label,'arguments': debate_arguments,'labels': label_list, 'weighted_label': weighted_label, 'factuality_value': factuality_value}
                explanation_list = debate_arguments
            
            eval_repeat_max += 1

        all_label_lists = [debates_dict[item]['labels'] for item in debates_dict]
        
        ## majority vote out of debate rounds. There is a winner for each debate and then the final winner is the one with the most votes
        debates_majority_vote_list = [debates_dict[item]['label'] for item in debates_dict]
        print(debates_majority_vote_list)
        if debates_majority_vote_list.count(1) == num_debates or debates_majority_vote_list.count(0) == num_debates:
            debate_ambiguity = False
        else:
            debate_ambiguity = True

        if debates_majority_vote_list.count(1)> debates_majority_vote_list.count(0):
            debates_majority_vote = 1
        elif debates_majority_vote_list.count(1) < debates_majority_vote_list.count(0):
            debates_majority_vote = 0
        print(debates_majority_vote)

        changes_in_debates_list = [debates_dict[item]['change'] for item in debates_dict]
        if changes_in_debates_list.count(0) == num_debates:
            ambiguous = "Full"
        elif changes_in_debates_list.count(0) == 0:
            ambiguous = "None"
        else:
            ambiguous = "Partial"

        # if changes_in_debates_list.count(0) != num_debates:
        overall_majority_list = []
        for label_list in all_label_lists:
            change = 0
            pn_list = []
            for lbl in label_list:
                if lbl[0] != lbl[-1]:
                    change += 1
                pn_list.append(lbl[-1])
            overall_majority_list += pn_list

        ## majority vote over all individual agents regardless of which debate they belong to
        if overall_majority_list.count(1)> overall_majority_list.count(0):
            overall_majority_vote = 1
        elif overall_majority_list.count(1) < overall_majority_list.count(0):
            overall_majority_vote = 0
        else:
            overall_ambiguity = True

        ## if there is a winner among the agents responses, we report the majority vote
        if changes_in_debates_list.count(0) != num_debates and overall_ambiguity == False: 
            label = overall_majority_vote
            explanation_list = [debates_dict[item]['arguments'] for item in debates_dict]
            adjudicator_list = []
            all_arguments = [debates_dict[item]['arguments'] for item in debates_dict]

        ## if there is NOT a winner among agents responses, we use adjudicators to make the final call
        elif changes_in_debates_list.count(0) == num_debates or overall_ambiguity == True:
            all_arguments = [debates_dict[item]['arguments'] for item in debates_dict]
            all_arguments = [x for xs in all_arguments for x in xs]
            all_chats_dict = {}
            for n_agents in range(len(all_arguments)):
                all_chats_dict['Agent %s:'%str(n_agents+1)] = ""

            for cht_counter, cht in enumerate(all_arguments):
                all_chats_dict['Agent %s:'%str(cht_counter+1)] += ' '.join(cht.split('\n')[1:]) + ' '
            
            adjudicator_input = [str(item) + ' ' + all_chats_dict[item] for item in all_chats_dict]
            
            label_list = []
            label = ""
            explanation_list = []
            for rep in range(3):
                random.shuffle(adjudicator_input)
                adjudicator_prompt = get_adjudicator_prompt(doc, sent, '\n'.join(adjudicator_input))
                rep_counter = 0
                while label == "" and rep_counter != 5:
                    adjudicator_response = llama_prediction(bedrock_rt, adjudicator_prompt, llama_model_id, 1.0)
                    label ,  explanation  = parse_output_response(adjudicator_response)
                    explanation_list.append(explanation)
                    print(label,  explanation )
                    print('********')
                    if label != "Unknown":
                        if len(label.split()) != 0 and ',' not in label.split()[0]:
                            label_val = float(label.split()[0])
                        elif len(label.split()) == 0 or ',' in label.split()[0]: 
                            if len(label.split(',')) != 0:
                                label_val = float(label.split(',')[0])
                            else: 
                                label_val = float(label)
                        if label_val >= 0.5:
                            label = 1
                        else:
                            label = 0
                    else:
                        label = -1
                    rep_counter += 1
                label_list.append(label)
                label = ""
            
            print(label_list)
            results['adjudicators'] = label_list
            results['adjudicators_agree'] = len(set(label_list)) == 1
            if label_list.count(1) >= label_list.count(0):
                label = 1
            else:
                label = 0
        
            overall_majority_vote = label
            adjudicator_list = label_list

        eval_repeat_max += 1
        
        ## compute final weighted label (not reported in paper)
        weighted_ambiguity = False
        all_weighted_labels = [debates_dict[item]['weighted_label'] for item in debates_dict if debates_dict[item]['weighted_label']!="Unknown"]
        if len(all_weighted_labels) == 0:
            final_weighted_label = debates_majority_vote
            weighted_ambiguity = True
        elif sum(all_weighted_labels) > 0:
            final_weighted_label = 1
        else:
            final_weighted_label = 0
        
        all_factuality_values = [debates_dict[item]['factuality_value'] for item in debates_dict]

        if label == 1 and instance['label'] == "1":
            tn += 1

        elif label == 0 and instance['label'] == "0":
            tp += 1

        elif label == 1 and instance['label'] == "0":
            fn += 1
        
        elif label == 0 and instance['label'] == "1":
            fp += 1
        
        if benchmark:
            results['dataset'] = instance['dataset']
        else:
            results['dataset'] = 'sample' ## specify the name of the dataset you are using
        results['pred_sent_label'] = label
        results['sent_label'] = instance['label']
        results['overall_majority_vote'] = overall_majority_vote
        results['debates_majority_vote'] = debates_majority_vote
        results['weighted_label'] = final_weighted_label
        results['all_labels_list'] = all_label_lists
        results['adjudicator_list'] = adjudicator_list
        results['change_ambiguity'] = ambiguous
        results['rounds_ambihuity'] = overall_ambiguity
        results['debate_ambiguity'] = debate_ambiguity
        results['weighted_ambiguity'] = weighted_ambiguity
        results['all_factuality_values'] = all_factuality_values
        results['explanation'] = all_arguments
        w.write(json.dumps(results, ensure_ascii=False) + '\n') 

    print(tp, tn, fp, fn)
    fpr = fp/(fp+tn)
    print("FPR: ",fpr)
    fnr = fn/(fn+tp)
    print("FNR: ",fnr)
    print("BAcc (Main): ", 1-((fpr+fnr)/2))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,choices=['zero-shot','cot','consistency','debate','ambiguity'],help='This is the setting that you want to use for your evaluation.')
    parser.add_argument("--data_file", type=str, required=False, default='', help='You can use this to define the path to a different dataset (not the ones used in this paper).')
    parser.add_argument("--benchmark", action='store_false', help='It this is not set, then you sould set the --dataset_name to use one out of the whole set. If set, use --data_file to specify your own data path')
    parser.add_argument("--dataset_name", type=str, required=False,choices=['AggreFact-CNN','AggreFact-XSum', 'TofuEval-MediaS', 'TofuEval-MeetB'], help='You should pick a dataset name from LLM-AggreFact benchmark.')
    parser.add_argument("--output_file_path", type=str, required=True, help='The path to where you want to save your outout file.')
    parser.add_argument("--num_rep_cot", type=int, default=41, help='Number of times you want to repeat for self-consistency approach')
    parser.add_argument("--num_debates", type=int, default=3,help='Number of separate debates')
    parser.add_argument("--num_rounds", type=int, default=2, help='Number of rounds within each debate')
    parser.add_argument("--args_file", type=str, required=False, help='The path to file with agents arguments for ambiguity detection task.')
    parser.add_argument("--annotated_file", type=str, required=False, help='The path to the file with annotated labels for ambiguity.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.model == 'zero-shot':
        single_llm(args.dataset_name, args.output_file_path, args.data_file, args.benchmark)
    elif args.model == 'cot':
        single_llm_cot(args.dataset_name, args.output_file_path, 1, args.data_file, args.benchmark)
    elif args.model == 'consistency':
        single_llm_cot(args.dataset_name, args.output_file_path, args.num_rep_cot, args.data_file, args.benchmark)
    elif args.model == 'debate':
        simultaneous_debate_optional_adjudicator(args.dataset_name, args.output_file_path, args.num_debates, args.num_rounds, args.data_file, args.benchmark)
    elif args.model == 'ambiguity':
        ambiguity(args.annotated_file, args.output_file_path)