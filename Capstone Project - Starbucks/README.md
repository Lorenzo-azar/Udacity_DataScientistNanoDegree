# Starbucks Capstone Project

# Table of Contents

1. [Project Overview](#overview)
2. [Problem Statements and Strategy](#problem)
3. [Libraries Used](#libraries)
4. [File Description](#file-desc)
5. [Dataset](#data)
6. [Summary](#summary)

---
## Project Overview: <a name="overview"></a>

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

Not all users receive the same offer, and that is the challenge to solve with this data set. This project will try to optimize the decision of choosing users to receive offers.

---

## Problem Statements and Strategy: <a name="problem"></a>

This project will solve the problem of predicting if the user is most likely to buy after seeing an ad. This will be tackled by creating embedding for the users and generating recommendations of the top N (i.e. top 10) similar users and then predicting from these top users if the main user is more likely to buy after getting a promotion offer.

---

## Libraries Used: <a name="libraries"></a>

The libraries used for this project are mainly `pandas`, `numpy` and `sklearn`

---

## File Description <a name="file-desc"></a>

* [**Starbucks_Capstone_notebook.ipynb**](Starbucks_Capstone_notebook.ipynb): 
Notebook that contain the all the work and explanation done.
* [**Starbucks_Capstone_notebook.html**](Starbucks_Capstone_notebook.html): 
HTML version of the notebook.
* [**data**](data): Folder containing all the datasets mentioned below.
* [**Images**](Images): Folder containing all the images used in the blog post.

## Dataset <a name="data"></a>

**Portfolio.json**
  - id (string) - offer id
  - offer_type (string) - type of offer ie BOGO, discount, informational
  - difficulty (int) - minimum required spend to complete an offer
  - reward (int) - reward given for completing an offer
  - duration (int) - time for offer to be open, in days
  - channels (list of strings)

**profile.json**
- age (int) - age of the customer
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

**transcript.json**
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record


## Summary of the results: <a name="summary"></a>

You can find a summary of the results in [this blog post](https://lorenzo-azar.medium.com/starbucks-capstone-challenge-5d0fd052960f).