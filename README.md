## Faithful, Unfaithful or Ambiguous? Multi-Agent Debate with Initial Stance for Summary Evaluation

Authors: Mahnaz Koupaee, Jake W. Vincent, Saab Mansour, Igor Shalyminov, Han He, Hwanjun Song, Raphael Shu, Jianfeng He, Yi Nian, Amy Wing-mei Wong, Kyu J. Han, Hang Su

Please check our paper [here](https://arxiv.org/abs/2502.08514).

## Madisse
Our debate approach for summary faithfulness evaluation consisting of a group of agnets with initial imposed beleifs of faithfulness which would engage in discussions to resolve any inconsistencies.

## Ambiguity annotation on MeetingBank

 `MeetingBank_ambiguity_annotated.json` contains the ambiguity annotations for MeetingBank summaries. The followings are descriptions of column names.

 | Column Name | Description |
 | -------- | ----- |
 | doc | source document |
 | summary | a generated summary sentence for the given document |
 | ambiguity | `0` if the given summary is not ambiguous or `1` if the summary is ambiguous |
 | category | if the summary is deemed ambiguous, then the selected high-level ambiguity category| 
 | sub-category | if the summary is deemed ambiguous, the selected fine-grained ambiguity sub-category form the taxonomy|
 | explanation | a short description of why there exists an ambiguity in the given summary |
