def get_prompt_minicheck(doc,summary):

    prompt = '''
    Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent.
    
    <doc>
    %s
    </doc>

    <summary>
    %s
    </summary>

    Please assess the claim’s consistency with the document.  Provide your evaluation between <label></label> tags with values 1 (consistent) or 0 (inconsistent) and add your explanations in <explanation></explanation> XML tags. Skip the preamble.

    '''%(doc,summary)
    return prompt

def get_prompt_direct_eval(doc,summary):

    prompt = '''
    You are given a document and a summary (summarizing only a part of the document). You will go over the document in the <doc></doc> tags carefully and try to understand it fully. Then you look at the summary in <summary></summary> tags carefully. Your task is to identify whether the summary is factually consistent with the given document. A summary is factually consistent with the document if it can be entailed (either stated or implied) by it.

    <doc>
    %s
    </doc>

    <summary>
    %s
    </summary>

    Determine if the sentence is factually consistent with the document provided above. Provide your evaluation between <label></label> tags with values 1 (consistent) or 0 (inconsistent) and add your explanations in <explanation></explanation> XML tags. Skip the preamble.
    '''%(doc,summary)
    return prompt

def get_prompt_direct_eval_individual_sentences(doc,summary):

    prompt = '''
    You are given a document and a summary (summarizing only a part of the document). You will go over the document in the <doc></doc> tags carefully and try to understand it fully. Then you look at the summary in <summary></summary> tags carefully. Your task is to identify whether the summary is factually consistent with the given document. A summary is factually consistent with the document if it can be entailed (either stated or implied) by it.

    <doc>
    %s
    </doc>

    <summary>
    %s
    </summary>

    Determine if the summary is factually consistent with the document provided above. You should go over each sentence of the summary one by one and check whether there is an error or not. A summary is non-factual if there is at least one error in it. Provide your evaluation between <label></label> tags with values 1 (consistent) or 0 (inconsistent) and add your explanations in <explanation></explanation> XML tags. Skip the preamble.

    '''%(doc,summary)
    return prompt

def get_prompt_debate(doc,summary,chat_history,mediator_feedback):
    prompt = '''
    You are given a document and a summary sentence (summarizing only a part of the document). You will go over the document in the <doc></doc> tags carefully and try to understand it fully. Then you look at the summary sentence in <summary></summary> tags. You have to identify whether the summary is factually consistent with the given document. There are also other evaluator agents assigned the same task as you and you can also see the discussion history in <chat_history></chat_history> tags below. You are also given a set of guidelines in <guideline></guidelines> that you can refer to when making your arguments. Go over them carefully and make sure you remember them.

    <guidelines>
    1. You should aim for accuracy and not comprehensiveness. If individual facts are correct, the summary is factually consistent regardless of its comprehensiveness.
    2. A summary does not imply that its facts are the only ones mentioned in the dialogue.
    3. The summary is factually inconsistent if it makes an assumption that is not supported (explicitly or implicitly) by the document.
    4. The summary is factually inconsistent if it includes any information (even a minor detail) that is not present in the document or can not be entailed from the document.
    5. The summary is factually consistent if it is a paraphrase of the document and it does not change the meaning of what is stated in the document.
    6. Details (even crucial) that are present in the document but omitted in the summary do not lead to factual inconsistency.
    7. lack of coherence between summary sentences does not necessarily lead to factual inconsistency.
    8. The summary should not hallucinate new entities such as new people or locations not mentioned in the document otherwise it is factually inconsistent.
    9. The summary does not have to provide the context or focus only on the main points of the document, it can only focus on a minor concept.
    10. The summary is factually consistent even if it omits crucial details from document.
    11. The addition of details that are not mentioned in the document or can not be entailed from it, makes the summary factually inconsistent.
    12. Every word or phrase of the summary (or its paraphrase) should be present in the document otherwise the summary is factually inconsistent.
    13. If even a single part of the summary is factually inconsistent, then the whole summary is factually inconsistent.
    </guidelines>

    <doc>
    %s
    </doc>

    <summary>
    %s
    </summary>

    <chat_history> 
    %s
    </chat_history>

    The <chat_history></chat_history> tag might be empty if this is the first round of evaluation. You can see your previous responses as well as other agents responses. Continue the discussion with other evaluator agents, talk to them and state why you agree/disagree with each other bringing as many arguments as you can. An argument should be provided for each summary sentence.%s You should refer to the provided guidelines (not your own guidelines) in your arguments. Place everything between <argument></argument> tags. Also provide your assessment of factuality in <label></label> tags with 0 as non-factual and 1 as factual. Skip the preamble. 
    '''%(doc,summary,chat_history,mediator_feedback)
    return prompt

def get_adjudicator_prompt(doc, summary, chat_history):
    prompt = '''
    You are given a document, a summary sentence (summarizing only a part of the document) and multiple judgments from evaluator agents. You will go over the document in the <doc></doc> tags and the summary sentence in <summary></summary> tags carefully. A summary is factually consistent if it can be entailed from the document. You go over the discussion between the agents and their arguments shown in between <chat_history></chat_history> tags. Your task is to make the final call on whether the summary is factually consistent with the given document based on the evaluator agents responses. You are also given a set of guidelines in <guideline></guidelines> which the agents have referred to, to make their arguments. Go over the guideline carefully and try to remember them.

    <guidelines>
    1. You should aim for accuracy and not comprehensiveness. If individual facts are correct, the summary is factually consistent regardless of its comprehensiveness.
    2. A summary does not imply that its facts are the only ones mentioned in the dialogue.
    3. The summary is factually inconsistent if it makes an assumption that is not supported (explicitly or implicitly) by the document.
    4. The summary is factually inconsistent if it includes any information (even a minor detail) that is not present in the document or can not be entailed from the document.
    5. The summary is factually consistent if it is a paraphrase of the document and it does not change the meaning of what is stated in the document.
    6. Details (even crucial) that are present in the document but omitted in the summary do not lead to factual inconsistency.
    7. lack of coherence between summary sentences does not necessarily lead to factual inconsistency.
    8. The summary should not hallucinate new entities such as new people or locations not mentioned in the document otherwise it is factually inconsistent.
    9. The summary does not have to provide the context or focus only on the main points of the document, it can only focus on a minor concept.
    10. The summary is factually consistent even if it omits crucial details from document.
    11. The addition of details that are not mentioned in the document or can not be entailed from it, makes the summary factually inconsistent.
    12. Every word or phrase of the summary (or its paraphrase) should be present in the document otherwise the summary is factually inconsistent.
    13. If even a single part of the summary is factually inconsistent, then the whole summary is factually inconsistent.
    </guidelines>

    <doc>
    %s
    </doc>

    <summary>
    %s
    </summary>

    <chat_history>
    %s
    </chat_history>

    Go over the agents responses, summarize them by saying who agrees/disagrees. Then looking at the agents responses, how well they are associated with the guidelines and finally your own judgement of the summary using the provided guidelines, determine if the summary is factually consistent with the document. Provide your evaluation between <label></label> keys with values 1 (consistent) or 0 (inconsistent) and add your explanations in <explanation></explanation> XML tags. Skip the preamble.
    '''%(doc,summary,chat_history)
    #Go over the agents responses, summarize them by saying who agrees/disagrees and make sure the arguments correctly used the provided guidelines. Then based on the correctness of agents responses and your own judegment of the summary using the provided guidelines, determine if the sentence is factually consistent with the document. A summary is factually inconsistent if there is a correct argument describing an error or discrepancy in the summary. Provide your evaluation using a JSON format with keys as "label" with values 1 (consistent) or 0 (inconsistent) and "explanation" and put your response between <response></response> tags. Skip the preamble.
    return prompt

def get_prompt_cot(doc,summary):

    prompt = '''
    You are given a document and a summary sentence (summarizing only a part of the document). You will go over the document in the <doc></doc> tags carefully and try to understand it fully. Then you look at the summary sentence in <summary></summary> tags carefully. Your task is to identify whether the summary is factually consistent with the given document. A summary is factually consistent with the document if it can be entailed (either stated or implied) by it.

    <doc>
    %s
    </doc>

    <summary>
    %s
    </summary>

    Determine if the sentence is factually consistent with the document provided above. Provide your evaluation between <label></label> tags with values 1 (consistent) or 0 (inconsistent) and add your explanations in <explanation></explanation> XML tags. Before answering, please think about the question within <thinking></thinking> XML tags.
    '''%(doc,summary)
    return prompt

def get_prompt_ambiguity(doc,summary):

    prompt = '''
    You are given a document and a summary. You will go over the document in the <doc></doc> tags carefully and try to understand it fully. Then you look at the summary in <summary></summary> tags carefully. Your task is to identify whether the summary contains an ambiguity according to the provided ambiguity taxonomy in <taxonomy></taxonomy> tags. A summary is ambiguous if it can have multiple correct interpretations.
    
    <doc>
    %s
    </doc>
    
    <summary>
    %s
    </summary>

    <taxonomy>
    1. Deduction: The summarizer has made a logical deduction (well or poorly), utilizing premises from the source document to draw a conclusion that cannot be directly traced to the source document.
    2. Common-sense inference: The summarizer appears to have made an inference based on common sense notions.
    3. Value-based inference: The summarizer appears to have made an inference based on assumed values.
    4. Other implicit reasoning phenomenon: Some other kind of implicit reasoning took place that affects the summary's evaluability.
    5. Hypernymy/Generalization: A more general meaning is used in the summary than is observed in the source document (for the same topic).
    6. Hyponymy/Specialization: A more specific meaning is used in the summary than is observed in the source document (for the same topic).
    7. Synonymy/Paraphrasing: Meaning from the source document is paraphrased in such a way that interpretation is challenged. The meaning has not technically changed, but the way the meaning is built changed.
    8. Structural ambiguity: A phrase or sentence in the summary has multiple valid parses (multiple valid syntactic structures), and it is not obvious which parse is intended.
    9. Lexical ambiguity: A word in the summary has multiple valid interpretations, and it is not obvious which meaning is intended.
    10. Other ambiguity phenomenon: There is another type of linguistic ambiguity in the summary that is likely to cause difficulty in interpretation. Other types of ambiguity include scope ambiguity and pronoun reference ambiguity.
    11. Vagueness: The meaning of part of the summary is underspecified, resulting in many realities being compatible with the claim made. For this use case, it would be so many realities that there is confusion about what claim is actually being made and whether the claim can be evaluated reliably.
    12. Other meaning phenomenon: There is something else about the literal meaning of the summary that may have made it challenging to assess its factuality.
    13. Decontextualization: The summary puts forth or describes something outside of the context in which its meaning was meant to be interpreted. It takes on new meaning or loses its meaning outside of that context.
    14. Conflation: The summary joins or synthesizes pieces of information that were independently relevant in the source document. (It may have done this to good effect or to bad effect.)
    15. Other context phenomenon: Some other challenge related to the relationship between the summary's meaning and the context(s) in the source document.
    </taxonomy>

    Determine if the summary is ambiguous or not. Provide your evaluation between <label></label> tags with values 1 as ambiguous and 0 as non-ambiguous, <category></category> tags for the ambiguity type and add your explanations in <explanation></explanation> XML tags.
    '''%(doc,summary)
    return prompt

def get_prompt_ambiguity_w_args(doc,summary,args):

    prompt = '''
    You are given a document and a summary. You will go over the document in the <doc></doc> tags carefully and try to understand it fully. Then you look at the summary in <summary></summary> tags carefully. Evaluator agents have had rounds of discussion to identify whether the summary is factual or not and you can see their arguments in <arguments></arguments> tags. Different agents might have contrasting reasonings on whether the summary is factual or not and they might be correct in their judgement even though they have opposing views. Your task is to go over the arguments and identify whether the summary contains an ambiguity using the provided ambiguity taxonomy in <taxonomy></taxonomy> tags that can cause opposing views of the factuality. An ambiguity is present when the summary can be correctly classified as both factual and non-factual at the same time. Please note that the arguments might not be correct as the agents might have misused the provided guidelines in <guidelines></guidelines> tags so first make sure the agents’ arguments indeed follow the guidelines and then only consider the ones that are sound in your abiguity evaluation.

    <doc>
    %s
    </doc>
    
    <summary>
    %s
    </summary>

    <arguments>
    %s
    </arguments>

    <guidelines>
    1. You should aim for accuracy and not comprehensiveness. If individual facts are correct, the summary is factually consistent regardless of its comprehensiveness.
    2. A summary does not imply that its facts are the only ones mentioned in the dialogue.
    3. The summary is factually inconsistent if it makes an assumption that is not supported (explicitly or implicitly) by the document.
    4. The summary is factually inconsistent if it includes any information (even a minor detail) that is not present in the document or can not be entailed from the document.
    5. The summary is factually consistent if it is a paraphrase of the document and it does not change the meaning of what is stated in the document.
    6. Details (even crucial) that are present in the document but omitted in the summary do not lead to factual inconsistency.
    7. lack of coherence between summary sentences does not necessarily lead to factual inconsistency.
    8. The summary should not hallucinate new entities such as new people or locations not mentioned in the document otherwise it is factually inconsistent.
    9. The summary does not have to provide the context or focus only on the main points of the document, it can only focus on a minor concept.
    10. The summary is factually consistent even if it omits crucial details from document.
    11. The addition of details that are not mentioned in the document or can not be entailed from it, makes the summary factually inconsistent.
    12. Every word or phrase of the summary (or its paraphrase) should be present in the document otherwise the summary is factually inconsistent.
    13. If even a single part of the summary is factually inconsistent, then the whole summary is factually inconsistent.
    </guidelines>

    <taxonomy>
    1. Deduction: The summarizer has made a logical deduction (well or poorly), utilizing premises from the source document to draw a conclusion that cannot be directly traced to the source document.
    2. Common-sense inference: The summarizer appears to have made an inference based on common sense notions.
    3. Value-based inference: The summarizer appears to have made an inference based on assumed values.
    4. Other implicit reasoning phenomenon: Some other kind of implicit reasoning took place that affects the summary's evaluability.
    5. Hypernymy/Generalization: A more general meaning is used in the summary than is observed in the source document (for the same topic).
    6. Hyponymy/Specialization: A more specific meaning is used in the summary than is observed in the source document (for the same topic).
    7. Synonymy/Paraphrasing: Meaning from the source document is paraphrased in such a way that interpretation is challenged. The meaning has not technically changed, but the way the meaning is built changed.
    8. Structural ambiguity: A phrase or sentence in the summary has multiple valid parses (multiple valid syntactic structures), and it is not obvious which parse is intended.
    9. Lexical ambiguity: A word in the summary has multiple valid interpretations, and it is not obvious which meaning is intended.
    10. Other ambiguity phenomenon: There is another type of linguistic ambiguity in the summary that is likely to cause difficulty in interpretation. Other types of ambiguity include scope ambiguity and pronoun reference ambiguity.
    11. Vagueness: The meaning of part of the summary is underspecified, resulting in many realities being compatible with the claim made. For this use case, it would be so many realities that there is confusion about what claim is actually being made and whether the claim can be evaluated reliably.
    12. Other meaning phenomenon: There is something else about the literal meaning of the summary that may have made it challenging to assess its factuality.
    13. Decontextualization: The summary puts forth or describes something outside of the context in which its meaning was meant to be interpreted. It takes on new meaning or loses its meaning outside of that context.
    14. Conflation: The summary joins or synthesizes pieces of information that were independently relevant in the source document. (It may have done this to good effect or to bad effect.)
    15. Other context phenomenon: Some other challenge related to the relationship between the summary's meaning and the context(s) in the source document.
    </taxonomy>

    Determine if the summary is ambiguous or not. We are only looking for cases where the ambiguity in the summary would lead to opposing factuality labels so if there is a general ambiguity but it does not lead to disagreement on factuality, it should be considered as non-ambiguous. Remember that your task is not evaluating the factuality of the summary. Provide your evaluation between <label></label> tags with values 1 as ambiguous and 0 as non-ambiguous, <category></category> tags for the ambiguity type and add your explanations in <explanation></explanation> XML tags.
    '''%(doc,summary,args)

    return prompt
