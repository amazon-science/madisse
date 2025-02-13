import json
import pandas as pd
import krippendorff
import numpy as np

def read_file(file_path):
    all_data = []
    with open(file_path,'r') as input_file:
        for line in input_file:
            data=json.loads(line)
            all_data.append(data)
    return all_data

def correlation_aggrefact():
    '''
    Measure the Krippendorff alpha between model responses and human annotations
    '''
    data = read_file('/Users/mkoupaee/Documents/project/aggrefact_results/simultaneous_optional_adju_mediasum_sent_dev_1debate1.json')
    eval_data = np.zeros(shape=(2,len(data)),dtype=np.int64)
    for i,d in enumerate(data):
        if d['overall_majority_vote'] == 1:
            model_data = 1
        elif d['overall_majority_vote'] == 0:
            model_data = 0
        if d['sent_label'] == '1':
            human_data = 1
        elif d['sent_label'] == '0':
            human_data = 0
        
        eval_data[0,i] = model_data
        eval_data[1,i] = human_data
    
    correlation_coef = krippendorff.alpha(reliability_data=eval_data,value_domain=[0,1])
    return correlation_coef

def bacc():
    '''
    Measure balanced accuracy of an evaluator
    '''
    dataset_names = ['AggreFact-CNN','AggreFact-XSum', 'TofuEval-MediaS', 'TofuEval-MeetB']
    dataset_dict = dict.fromkeys(dataset_names,None)
    instances = read_file('/Users/mkoupaee/Documents/project/aggrefact_results/simultaneous_optional_adju_mediasum_sent_dev_1debate1.json')
    for i, instance in enumerate(instances):
        if instance['overall_majority_vote'] == 1 and instance['sent_label'] == "1":
            if dataset_dict[instance['dataset']] == None:
                dataset_dict[instance['dataset']] = {'tp': 0,'tn': 1, 'fp': 0, 'fn': 0}
            else:
                dataset_dict[instance['dataset']]['tn'] += 1

        elif instance['overall_majority_vote'] == 0 and instance['sent_label'] == "0":
            if dataset_dict[instance['dataset']] == None:
                dataset_dict[instance['dataset']] = {'tp': 1,'tn': 0, 'fp': 0, 'fn': 0}
            else:
                dataset_dict[instance['dataset']]['tp'] += 1

        elif instance['overall_majority_vote'] == 1 and instance['sent_label'] == "0":
            if dataset_dict[instance['dataset']] == None:
                dataset_dict[instance['dataset']] = {'tp': 0,'tn': 0, 'fp': 0, 'fn': 1}
            else:
                dataset_dict[instance['dataset']]['fn'] += 1
        
        elif instance['overall_majority_vote'] == 0 and instance['sent_label'] == "1":
            if dataset_dict[instance['dataset']] == None:
                dataset_dict[instance['dataset']] = {'tp': 0,'tn': 0, 'fp': 1, 'fn': 0}
            else:
                dataset_dict[instance['dataset']]['fp'] += 1

    all_results = []
    tp, tn, fp, fn = 0, 0, 0, 0
    for dataset in dataset_dict:
        if dataset == 'TofuEval-MeetB':
            results = {}
            tp = dataset_dict[dataset]['tp']
            tn = dataset_dict[dataset]['tn']
            fp = dataset_dict[dataset]['fp']
            fn = dataset_dict[dataset]['fn']

            print(dataset)
            print(tp, tn, fp, fn)
            fpr = fp/(fp+tn)
            print("FPR: ",fpr)
            fnr = fn/(fn+tp)
            print("FNR: ",fnr)
            print("BAcc (Main): ", 1-((fpr+fnr)/2))
            print("***********")
            results['dataset'] = dataset
            results['FPR'] = fpr
            results['FNR'] = fnr
            results['BAcc'] = 1-((fpr+fnr)/2)
            all_results.append(results)
    
    df = pd.DataFrame(all_results)
    df.to_csv('/Users/mkoupaee/Documents/project/aggrefact_results/short_test_set_cnn.csv', encoding='utf-8', index=False)
    return bacc

def bacc_wo_ambiguities():
    ambiguouys_ids = [6, 9, 14, 15, 17, 18, 25, 30, 32, 33, 34, 41, 42, 46, 53, 70, 76, 78, 79, 81, 83, 92, 94, 104, 105, 111, 117, 119, 122, 152, 153, 166, 169, 174, 181, 185, 190, 197, 199, 231, 241, 247, 250, 274, 284, 306, 374, 379, 427, 431, 432, 437, 439, 440, 456, 477, 481, 486, 487, 490, 494, 495, 499, 500, 504, 506, 518, 519, 520, 527, 532, 533, 547, 550, 551, 565, 579, 587, 597, 606, 609, 612, 655, 657, 659, 663, 664, 670, 673, 681, 684, 697, 709, 714, 715, 717]
    incorrect_labels_ids = [22, 62, 68, 69, 82, 93, 99, 102, 107, 112, 113, 133, 146, 147, 161, 170, 198, 201, 202, 203, 205, 207, 208, 210, 213, 215, 249, 251, 257, 287, 300, 377, 386, 399, 450, 469, 491, 496, 505, 528, 566, 601, 644, 650, 666, 675, 678, 680, 685, 713]
    tp, tn, fp, fn = 0, 0, 0, 0
    reference_instances = read_file('/Users/mkoupaee/Documents/project/aggrefact_results/annotation/mediasum_full_manual_annotation.json')
    reference_ambiguities = []
    reference_labels = []
    instances = read_file('/Users/mkoupaee/Documents/project/aggrefact_results/minicheck_mediasum_sent.json')
    for ref in reference_instances:
        reference_ambiguities.append(ref.split('"manual_ambiguous": ')[1].split(',')[0])
        reference_labels.append(ref.split('"manual_sent_label": ')[1].split(',')[0])
    
    ambiguous_ids = []
    incorrect_labels = []
    for i, instance in enumerate(instances):
        if reference_labels[i] != '"%s"'%instance['sent_label'] and reference_ambiguities[i] != "true":
            incorrect_labels.append(i)
        if reference_ambiguities[i] == "true":
            ambiguous_ids.append(i)
            continue

        if instance['pred_sent_label'] == 1 and reference_labels[i] == '"1"':
            tn += 1

        elif instance['pred_sent_label'] == 0 and reference_labels[i]== '"0"':
            tp += 1

        elif instance['pred_sent_label'] == 1 and reference_labels[i] == '"0"':
            fn += 1
        
        elif instance['pred_sent_label'] == 0 and reference_labels[i] == '"1"':
            fp += 1

    print(tp, tn, fp, fn)
    fpr = fp/(fp+tn)
    print("FPR: ",fpr)
    fnr = fn/(fn+tp)
    print("FNR: ",fnr)
    print("BAcc (Main): ", 1-((fpr+fnr)/2))
    print("***********")

    eval_data = np.zeros(shape=(2,len(instances)),dtype=np.int64)
    for i,d in enumerate(instances):
        if reference_ambiguities[i] == "true":
            continue
        if d['pred_sent_label'] == 1:
            model_data = 1
        elif d['pred_sent_label'] == 0:
            model_data = 0
        if reference_labels[i] == '"1"':
            human_data = 1
        elif reference_labels[i] == '"0"':
            human_data = 0
        
        eval_data[0,i] = model_data
        eval_data[1,i] = human_data

    correlation_coef = krippendorff.alpha(reliability_data=eval_data,value_domain=[0,1])
    print(correlation_coef)
    
    '''
    w = open('/Users/mkoupaee/Documents/project/aggrefact_results/summary_ambiguity_from_sentence_summaries_mediasum.json','w')
    full_instances = read_file('/Users/mkoupaee/Documents/project/tofueval_docs_annotations/mediasum_test_full_summaries.json')
    summary_dict = {}
    all_summaries = []
    for i, instance in enumerate(full_instances):
        all_summaries.append(instance['claim'])
    
    for sentence_level_summary, ambiguity in zip(instances,reference_ambiguities):
        for s, summary in enumerate(all_summaries):
            if sentence_level_summary['claim'] in summary:
                if s not in summary_dict:
                    summary_dict[s] = [ambiguity]
                else:
                    summary_dict[s].append(ambiguity)

    for i, instance in enumerate(full_instances):
        results = {}
        results['dataset'] = 'TofuEval-MeetB'
        if summary_dict[i].count('true') > 0:
            results['ambiguity'] = True
        else:
            results['ambiguity'] = False
        results['sent_label'] = instance['label']
        results['sentence_labels'] = summary_dict[i]
        w.write(json.dumps(results, ensure_ascii=False) + '\n')

    instances = read_file('/Users/mkoupaee/Documents/project/aggrefact_results/summary_ambiguity_from_sentence_summaries_mediasum.json')
    for i, instance in enumerate(instances):
        if instance['ambiguity'] == True:
            continue

        if instance['pred_sent_label'] == 1 and reference_labels[i] == '"1"':
            tn += 1

        elif instance['pred_sent_label'] == 0 and reference_labels[i]== '"0"':
            tp += 1

        elif instance['pred_sent_label'] == 1 and reference_labels[i] == '"0"':
            fn += 1
        
        elif instance['pred_sent_label'] == 0 and reference_labels[i] == '"1"':
            fp += 1

    print(tp, tn, fp, fn)
    fpr = fp/(fp+tn)
    print("FPR: ",fpr)
    fnr = fn/(fn+tp)
    print("FNR: ",fnr)
    print("BAcc (Main): ", 1-((fpr+fnr)/2))
    print("***********")

    eval_data = np.zeros(shape=(2,len(instances)),dtype=np.int64)
    for i,d in enumerate(instances):
        if reference_ambiguities[i] == "true":
            continue
        if d['pred_sent_label'] == 1:
            model_data = 1
        elif d['pred_sent_label'] == 0:
            model_data = 0
        if reference_labels[i] == '"1"':
            human_data = 1
        elif reference_labels[i] == '"0"':
            human_data = 0
        
        eval_data[0,i] = model_data
        eval_data[1,i] = human_data

    correlation_coef = krippendorff.alpha(reliability_data=eval_data,value_domain=[0,1])
    print(correlation_coef)
    '''

