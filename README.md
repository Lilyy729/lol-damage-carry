---
layout: default
title: Who Really Carried?
---

# Who Really Carried?

## Kill Leaders and Damage Carries in League of Legends

**Lily Han**

## Introduction

Does having the most kills accurately identify a team's damage carry?

This project uses 2022 professional League of Legends match data from [Oracle's Elixir](https://oracleselixir.com/tools/downloads). The original dataset contains 150,348 rows. Each game normally has 12 rows: ten player rows and two team-summary rows.

League of Legends is played between two teams of five. I define a team's **damage carry** as the player who dealt the most total damage to enemy champions on that team. Kills are often used to describe who carried a game, but kills do not capture every contribution. I investigate how often the kill leader and damage carry are the same player, then use information available at 15 minutes to predict the eventual damage carry.

| Column | Description |
| --- | --- |
| `gameid` | Unique identifier for a game |
| `league` | Professional league in which the game was played |
| `side` | Blue or Red team |
| `position` | Player's role: top, jungle, mid, bot, or support |
| `kills` | Number of enemy champions killed by the player |
| `damagetochampions` | Total damage dealt to enemy champions |
| `result` | Whether the player's team won (`1`) or lost (`0`) |
| `goldat15` | Player's gold at 15 minutes |
| `gamelength` | Game length in seconds |

## Data Cleaning and Exploratory Data Analysis

The original data contains both player and team-summary rows, so I first kept only the five player positions. I grouped rows by `gameid` and `side`, making each group one team in one game. I retained team-games with all five players and complete `kills` and `damagetochampions` values. I then created indicators for whether each player was a kill leader or damage carry. Teams tied for highest damage were excluded because the prediction target requires one damage carry; ties for most kills were retained, and all tied players were considered kill leaders.

The cleaned data used for the exploratory analysis contains 125,255 player observations across 25,051 team-games. Its first five rows are shown below.

| gameid | side | position | kills | damagetochampions | result | is_kill_leader | is_damage_carry |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| ESPORTSTMNT01_2690210 | Blue | top | 2 | 15,768 | 0 | True | True |
| ESPORTSTMNT01_2690210 | Blue | jng | 2 | 11,765 | 0 | True | False |
| ESPORTSTMNT01_2690210 | Blue | mid | 2 | 14,258 | 0 | True | False |
| ESPORTSTMNT01_2690210 | Blue | bot | 2 | 11,106 | 0 | True | False |
| ESPORTSTMNT01_2690210 | Blue | sup | 1 | 3,663 | 0 | False | False |

### Univariate Analysis

<iframe src="assets/kills-distribution.html" width="100%" height="520" frameborder="0"></iframe>

The distribution of kills is right-skewed. Most players finish with relatively few kills, while a much smaller number have very high kill totals.

### Bivariate Analysis

<iframe src="assets/kill-leader-vs-damage-carry.html" width="100%" height="520" frameborder="0"></iframe>

The kill leader was also the damage carry for about 56.2% of winning teams and 48.9% of losing teams. The overlap is higher among winners, but neither percentage is large enough for kills alone to identify the damage carry consistently.

### Interesting Aggregates

| Position | Players | Average Kills | Median Damage | Damage-Carry Rate (%) |
| --- | ---: | ---: | ---: | ---: |
| Top | 25,051 | 2.80 | 14,365 | 22.95 |
| Jungle | 25,051 | 3.06 | 9,225 | 4.02 |
| Mid | 25,051 | 3.50 | 16,018 | 34.45 |
| Bot | 25,051 | 4.26 | 16,398 | 38.21 |
| Support | 25,051 | 0.87 | 4,742 | 0.38 |

Bot and Mid players have the highest median damage and damage-carry rates, while Support players have the lowest. These differences suggest that position is relevant when trying to predict the damage carry.

## Assessment of Missingness

I examined missing values in `goldat15` because this column is used in my prediction model. I selected the Blue-side Top player from each game so that each game was counted once, giving 12,529 observations.

For each permutation test, the null hypothesis was that `goldat15` missingness was independent of the selected column. The alternative hypothesis was that missingness depended on that column. I used total variation distance (TVD), shuffled the missingness labels 1,000 times, and used a significance level of 0.05.

For `league`, the observed TVD was approximately 0.9926 and the p-value was approximately 0.0010. I rejected the null hypothesis and found evidence that `goldat15` missingness depends on league. For the Blue team's `result`, the observed TVD was approximately 0.0130 and the p-value was approximately 0.2987, so I failed to reject the null hypothesis.

<iframe src="assets/missingness-permutation.html" width="100%" height="520" frameborder="0"></iframe>

I do not have a strong reason to believe `goldat15` is MNAR. Several leagues have no recorded 15-minute gold values at all, suggesting that differences in data collection or coverage may explain the missingness. This is consistent with a possible MAR mechanism related to league, although these tests alone cannot rule out MNAR. Data-source coverage records or collection-failure logs would help explain the missing values.

## Hypothesis Testing

**Null hypothesis:** Within each team, damage values are exchangeable among the five players while kill counts stay fixed. Under this random-assignment model, damage-carry status is not linked to kill-leader status.

**Alternative hypothesis:** A team's damage carry is also a kill leader more often than expected under the random-assignment model.

The test statistic was the proportion of team-games whose damage carry was also a kill leader. I independently shuffled damage values among the five players on each team 10,000 times while keeping their kill counts fixed. This was a right-tailed test with a significance level of 0.05.

The observed matching proportion was approximately 0.5256, compared with a mean simulated proportion of 0.2510. None of the 10,000 simulated proportions were at least as large as the observed value. With the Monte Carlo correction, the p-value was approximately 0.0001, so I rejected the null hypothesis.

This provides evidence that damage carries are also kill leaders more often than chance would predict. However, they matched in only about 52.6% of teams, so kills alone do not consistently identify the damage carry. The test establishes an association, not a causal relationship.

## Framing a Prediction Problem

I predict whether a player will finish a game as their team's damage carry using information available at 15 minutes. This is a binary classification problem. The response variable is `is_damage_carry`, which is `True` for the player with the highest final damage to enemy champions on their team and `False` otherwise.

The prediction is made at 15 minutes in games that are still in progress. Final damage, final kills, final game length, and the final result are not used as features because they are unavailable at prediction time. Final damage is used only to create the target label.

My main evaluation metric is the F1-score for the damage-carry class. About 20% of player rows are positive, so a model that always predicts “not the carry” could achieve about 80% accuracy without finding any carries. F1 is more appropriate because it combines precision and recall.

I split the data by `gameid`, keeping all players from a game together in either the training or test set. This prevents information from the same game from appearing in both sets.

## Baseline Model

My baseline model is logistic regression using two features: `position`, a nominal categorical feature, and `goldat15`, a quantitative feature. I one-hot encoded `position`, standardized `goldat15`, and combined the transformations and classifier in one sklearn Pipeline. I used balanced class weights because only 20% of players are damage carries.

The model was trained on 8,509 games and evaluated on 2,128 held-out games.

| Dataset | F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| Train | 0.5077 | 0.3588 | 0.8676 |
| Test | 0.5062 | 0.3565 | 0.8724 |

The similar training and test F1-scores suggest little overfitting. On the test set, the model found about 87.2% of actual damage carries, but only about 35.7% of its carry predictions were correct. I consider it a useful starting point, but its low precision leaves room for improvement.

## Final Model

My final model is a Random Forest Classifier. I retained `position` and `goldat15` and added two engineered quantitative features:

- `gold_share_at15`: the player's share of the team's total gold at 15 minutes.
- `gold_diff_team_avg_at15`: the difference between the player's gold and the team's average gold at 15 minutes.

These features compare a player with their teammates, which is useful because damage-carry status is also defined within a team. The Pipeline created the engineered features, standardized the quantitative features, one-hot encoded `position`, and trained the classifier.

I tuned `max_depth` and `min_samples_leaf` using three-fold cross-validation grouped by `gameid`, with F1 as the selection metric. The best settings were `max_depth=10` and `min_samples_leaf=1`, with a cross-validation F1-score of approximately 0.5229.

| Model | Test F1 | Test Precision | Test Recall |
| --- | ---: | ---: | ---: |
| Baseline | 0.5062 | 0.3565 | 0.8724 |
| Final | 0.5142 | 0.3716 | 0.8346 |

The final model improved test F1 from 0.5062 to 0.5142 and precision from 0.3565 to 0.3716. Recall decreased, meaning the final model missed slightly more actual carries but made fewer incorrect carry predictions. Since F1 was the primary metric, I consider the final model an improvement.

## Fairness Analysis

I compared the final model's performance for Mid players (Group X) and Bot players (Group Y), using F1-score as the evaluation metric.

**Null hypothesis:** The model is fair between Mid and Bot players. Their F1-scores are equal, and the observed difference is due to random chance.

**Alternative hypothesis:** The model performs worse for Mid players than for Bot players.

The test statistic was Mid F1 minus Bot F1, and the significance level was 0.05. The Mid F1-score was approximately 0.5259 and the Bot F1-score was approximately 0.5763, giving an observed difference of approximately -0.0504. I shuffled the Mid and Bot labels 10,000 times while keeping the actual labels and model predictions fixed.

<iframe src="assets/fairness-permutation.html" width="100%" height="520" frameborder="0"></iframe>

None of the simulated differences were as small as the observed difference. With the Monte Carlo correction, the p-value was approximately 0.0001. I rejected the null hypothesis, providing evidence that the final model performs worse for Mid players than for Bot players according to F1-score. This result identifies a performance disparity on the test data, but it does not prove that the model is unfair in every setting.

