# Starbucks Capstone Project

**The link to the Blog post is [here](https://github.com/JoseEstevan/DS-Experiments/blob/main/Starbucks_Capstone/Starbucks_Project_Report.pdf)**
### Table of Contents

1. [Overview](#installation)
2. [Project Motivation](#motivation)
3. [Project Components](#files)
4. [Conclusion](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

***

## 1. Project Overview <a name="installation"></a>

The data set contains simulations of user activity on the Starbucks Rewards mobile app. These simulations imitate the behavior of real customers. Starbucks will send a promotion to customers who have downloaded its mobile app once every few days. The term "offer" can refer to either a simple advertisement for a beverage or a genuine offer, such as a price reduction or buy one get one free deal (buy one get one free). There is a possibility that certain users will not receive any offer during particular weeks.

The fact that not all consumers are given the same offer constitutes the mystery that must be solved using this data collection.

My job is to aggregate data on transactions, demographics, and offers in order to figure out which demographic groupings respond the most favorably to which kind of offers. Because the underlying simulator only has one product, this data set is a simplified version of the genuine Starbucks app. In reality, Starbucks sells dozens of different products, but the simulator only has one product.

Due to the fact that this is a capstone project, I am at liberty to do the data analysis in any way that I see appropriate. For instance, I could develop a model using machine learning that estimates how much money a person will spend on the basis of demographic information and the kind of offer they are considering. Alternately, I might construct a model that can determine in advance whether or not someone will accept an offer.

## 2. Project Motivation <a name="motivation"></a>
As part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025), I needed to develop a project following these [steps]().

In this project, I use the data to answer a business questions:
<ul>
    <li>Which demographic groups respond best to which offer type.</li>
<li>And develop a model using machine learning to determine the likelihood that a consumer would complete a transaction after viewing an offer in order to take advantage of the offer..</li>
 </ul>


## 3. Project Components <a name="files"></a>

<ol>
    <li> Loading Libraries and Data </li>
    <li> Data Wrangling </li>
    <li> Data Exploration </li>
    <li> Model Building </li>
    <li> Hyperparameter Tuning </li>
    <li> Conclusion </li>
</ol>


## 6. Conclusion <a name="results"></a>
I found this project challenging, mainly due to the structure of the data in thetranscript dataset.
Majority classes are performing well but the minorities are not.Problem of imbalanceddataset
Most of the events are wrongly predicted as 'offer received'; offer received is the mostoccuring event or class.
The main goal I chose was to build something practical the company could use tomake their choices more efficient.
But the results of the model seem not so good . There is no change in rate of accuracy;it remains constant.


## 7. Credits and Acknowledgements

Thanks to [Udacity](https://www.udacity.com/) for providing the dataset and the challenge.
