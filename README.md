<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-fair-federation-flower/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  RecSys-Fed-Fair: Fairness in Federated Recommendation
</h3>

<p align="center">Fairness algorithm aimed at reducing group unfairness in recommendation systems. </p>

## :page_with_curl: About the project <a name="-about"/></a>

Recommender systems play a fundamental role in assisting users in discovering relevant content amidst an abundance of data. However, fairness and justice in recommendations have emerged as critical issues, particularly for marginalized groups. This study presents the RecSys-Fed-Fair framework, aimed at integrating fairness into recommender systems operating under the federated learning paradigm. By formalizing individual and group fairness metrics, we demonstrate that the FedFair(l) method significantly reduces group unfairness (Rgrp) compared to traditional methods such as FedAvg(n) and FedAvg(l). Our results reveal that while there is a slight increase in the Root Mean Square Error (RMSE), it occurs at a level that does not compromise the model's predictive effectiveness. This research contributes to understanding how federated learning techniques can be leveraged to promote fairer recommendations, highlighting the importance of considering ethics in the application of recommendation algorithms.


### :notebook_with_decorative_cover: Algorithm <a name="-algorithm"/></a>

<img src="https://github.com/ravarmes/recsys-fair-federation-flower/blob/main/assets/recsys-fair-federation-flower-1.png" width="700">


### Files

| File                                 | Description                                                                                                                                                                                                                                   |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Class to promote fairness in recommendations of recommendation system algorithms.                                                                                                                                                              |
| AlgorithmUserFairness                | Classes to measure fairness (polarization, individual fairness, and group fairness) of recommendations of recommendation system algorithms.                                                                                                    |
| FedAvg-Example-05-02-GoodBooks            | Script to run a federated recommendation system simulation using the FedAvg aggregation method based on sample numbers for the GoodBooks dataset. The number of evaluations per round is 5 for the favored group and 2 for the disadvantaged group. |
| FedAvg-Example-26-14-MovieLens            | Script to run a federated recommendation system simulation using the FedAvg aggregation method based on sample numbers for the MovieLens dataset. The number of evaluations per round is 26 for the favored group and 14 for the disadvantaged group. |
| FedAvg-Loss-05-02-GoodBooks               | Script to run a federated recommendation system simulation using the FedAvg aggregation method based on local error (loss) for the GoodBooks dataset. The number of evaluations per round is 5 for the favored group and 2 for the disadvantaged group.  |
| FedAvg-Loss-26-14-MovieLens               | Script to run a federated recommendation system simulation using the FedAvg aggregation method based on local error (loss) for the MovieLens dataset. The number of evaluations per round is 26 for the favored group and 14 for the disadvantaged group. |
| FedFair-Loss-Activity-05-02-GoodBooks     | Script to run a federated recommendation system simulation using the FedFair aggregation method, including fairness regulation, considering user grouping by activity level for the GoodBooks dataset.                                                        |
| FedFair-Loss-Activity-26-14-MovieLens     | Script to run a federated recommendation system simulation using the FedFair aggregation method, including fairness regulation, considering user grouping by activity level for the MovieLens dataset.                                                        |
| FedFair-Loss-Age-26-14-MovieLens          | Script to run a federated recommendation system simulation using the FedFair aggregation method, including fairness regulation, considering user grouping by age group for the MovieLens dataset.                                                             |
| FedFair-Loss-Gender-26-14-MovieLens       | Script to run a federated recommendation system simulation using the FedFair aggregation method, including fairness regulation, considering user grouping by gender for the MovieLens dataset.                                                                   |
