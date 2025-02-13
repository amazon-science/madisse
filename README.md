## Faithful, Unfaithful or Ambiguous? Multi-Agent Debate with Initial Stance for Summary Evaluation

Authors: Mahnaz Koupaee, Jake W. Vincent, Saab Mansour, Igor Shalyminov, Han He, Hwanjun Song, Raphael Shu, Jianfeng He, Yi Nian, Amy Wing-mei Wong, Kyu J. Han, Hang Su

Please check out our paper [here](https://arxiv.org/abs/2502.08514).

## Madisse
Our debate approach for summary faithfulness evaluation consisting of a group of agnets with initial imposed beleifs of faithfulness which would engage in discussions to resolve any inconsistencies is shown below.
Each debate session consists of three stages: **1) stance initialization**, in which agents are assigned a belief of the summary faithfulness (faithful or unfaithful), **2) debate**, where evaluator agents engage in multiple rounds of debate to persuade each other of whether the summary is faithful or not, and **3) adjudication**, where based on the arguments from the debate, the final label is assigned to the summary. Madisse can have simultaneous debate sessions

![alt text](https://github.com/amazon-science/madisse/blob/main/images/overview_wo_ambiguity.jpg)

## Ambiguity annotation on MeetingBank 

 `MeetingBank_ambiguity_annotated.json` in the `data` folder contains the ambiguity annotations for MeetingBank summaries. The followings are descriptions of column names.

 | Column Name | Description |
 | -------- | ----- |
 | doc | source document |
 | summary | a generated summary sentence for the given document |
 | ambiguity | `0` if the given summary is not ambiguous or `1` if the summary is ambiguous |
 | category | if the summary is deemed ambiguous, then the selected high-level ambiguity category| 
 | sub-category | if the summary is deemed ambiguous, the selected fine-grained ambiguity sub-category form the taxonomy|
 | explanation | a short description of why there exists an ambiguity in the given summary |

## Madisse with ambiguity detection module

An ideal faithfulness evaluation system should handle ambiguities first. This can be done by identifying the ambiguous summaries and filtering them out and then evaluating the non-ambiguous summaries. 
The overall view of a faithfulness evaluator with the ambiguity detection module is shown below:

![alt text](https://github.com/amazon-science/madisse/blob/main/images/overview_w_ambiguity.jpg)

## Citation
```
@misc{koupaee2025faithfulunfaithfulambiguousmultiagent,
      title={Faithful, Unfaithful or Ambiguous? Multi-Agent Debate with Initial Stance for Summary Evaluation}, 
      author={Mahnaz Koupaee and Jake W. Vincent and Saab Mansour and Igor Shalyminov and Han He and Hwanjun Song and Raphael Shu and Jianfeng He and Yi Nian and Amy Wing-mei Wong and Kyu J. Han and Hang Su},
      year={2025},
      eprint={2502.08514},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08514}, 
}
```
