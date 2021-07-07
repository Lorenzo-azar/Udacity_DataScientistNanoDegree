# Starbucks Portfolio Exercise
---
For a full description of what Starbucks provides to candidates see the [instructions available here](https://drive.google.com/file/d/18klca9Sef1Rs6q8DW4l7o349r8B70qXM/view).

---
## Description

### About the Data
- The Dataset Provided is originally used as a take-home assignment provided by Starbucks for their job candidates. 
- Test an advertising promotion to see if it would bring more customers to purchase a specific product priced at $10. (Each promotion costs the company 0.15 to send out.)
- It consists of about 120,000 data points split in a 2:1 ratio among training and test files.
- Each data point includes one column indicating if the individual purchased that product. Aditionally, each individual has 7 additional features provided abstractly as V1-V7

---
### Objective
- Use the training data to understand what patterns in V1-V7 indicate that a promotion should be provided to a user.
- Try to limit the promotion only to those that are most receptive to it.
### Maximize the following Metrics
####  Incremental Response Rate (IRR)
Shows how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion.

![alt IRR](https://github.com/Lorenzo-azar/Udacity_DataScientistNanoDegree/blob/main/Bonus-%20Starbucks%20Portfolio%20Interview%20Take-Home%20Assignment/data/readme_images/IRR.jpg)

Where:
1. treat --> treatment group - group that received the promotion
2. ctrl --> control group - non-promotional group
3. purch --> number of purchasers 
4. cust --> number of customers

#### Net Incremental Revenue (NIR)
Shows how much is made (or lost) by sending out the promotion.

![alt NIR](https://github.com/Lorenzo-azar/Udacity_DataScientistNanoDegree/blob/main/Bonus-%20Starbucks%20Portfolio%20Interview%20Take-Home%20Assignment/data/readme_images/NIR.jpg)

---
### Instructions
Download the libraries needed

    pip install -r requirements.txt

Run the notebook Starbucks.ipynb

### Folder Structure
    .
    ├── data                    # includes the train and test raw data.
    ├── driver                  # includes driver modules (for testing)
    ├── Starbucks.ipynb         # jupyter notebook solution of the Portfolio Exercise
    ├── requirements.txt        # includes libraries required to run the notebook
    └── README.md

