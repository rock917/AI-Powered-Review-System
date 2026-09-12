# app/utils/demo_data.py
# FeedbackIQ — Hardcoded Demo Dataset
# This file ships with the code so deployment works without
# the full processed CSV files.

import pandas as pd
import numpy as np

DEMO_REVIEWS = [
    {"Text": "This coffee is absolutely amazing. The flavor is rich and smooth. I have been ordering for years and it never disappoints.", "Score": 5, "Time": 1325376000},
    {"Text": "Terrible product. Arrived damaged and smelled awful. Complete waste of money. Do not buy this.", "Score": 1, "Time": 1317340800},
    {"Text": "Great taste and fast shipping. Will definitely order again. My whole family loves it.", "Score": 5, "Time": 1320192000},
    {"Text": "The quality has gone down significantly. Used to be my favorite but not anymore.", "Score": 2, "Time": 1322784000},
    {"Text": "Absolutely love this product. Best purchase I have made in a long time. Highly recommend.", "Score": 5, "Time": 1315526400},
    {"Text": "Very disappointed. The packaging was destroyed and half the product was missing.", "Score": 1, "Time": 1318032000},
    {"Text": "Good product overall. Tastes fresh and the price is reasonable for the quality.", "Score": 4, "Time": 1323648000},
    {"Text": "Stopped working after two days. Customer service was unhelpful and rude.", "Score": 1, "Time": 1319760000},
    {"Text": "Excellent quality as always. Fast delivery and perfect packaging. Five stars.", "Score": 5, "Time": 1326240000},
    {"Text": "Not worth the price. Bland flavor and poor texture. Expected much better.", "Score": 2, "Time": 1316736000},
    {"Text": "My go-to product for years. Consistent quality and great customer service.", "Score": 5, "Time": 1324512000},
    {"Text": "Disgusting smell when opened. Clearly not fresh. Threw the whole thing away.", "Score": 1, "Time": 1321056000},
    {"Text": "Really good product. Arrived quickly and well packaged. Would recommend.", "Score": 4, "Time": 1327104000},
    {"Text": "The worst product I have ever bought. Broke immediately. Total scam.", "Score": 1, "Time": 1318896000},
    {"Text": "Perfect flavor and great value for money. Will keep buying this.", "Score": 5, "Time": 1325808000},
    {"Text": "Mediocre at best. Nothing special about this product. Overpriced.", "Score": 2, "Time": 1320624000},
    {"Text": "Outstanding product. Exceeded my expectations in every way possible.", "Score": 5, "Time": 1323216000},
    {"Text": "Arrived late and damaged. Not impressed with the quality or service.", "Score": 1, "Time": 1317772800},
    {"Text": "Very good quality. Fresh and tasty. My family really enjoys this product.", "Score": 4, "Time": 1326672000},
    {"Text": "Horrible experience. Product was stale and the return process was painful.", "Score": 1, "Time": 1319328000},
    {"Text": "Love the taste and convenience. Easy to prepare and delicious every time.", "Score": 5, "Time": 1324944000},
    {"Text": "Not as described. Color was wrong and texture was off. Very misleading.", "Score": 2, "Time": 1321488000},
    {"Text": "Fantastic product at a great price. Fresh and flavorful every single time.", "Score": 5, "Time": 1327536000},
    {"Text": "Completely useless. Does not work as advertised. Requesting a refund.", "Score": 1, "Time": 1318464000},
    {"Text": "Good value for money. Solid product that does exactly what it should.", "Score": 4, "Time": 1325040000},
    {"Text": "Awful taste and terrible smell. Nothing like what was shown in the photos.", "Score": 1, "Time": 1316304000},
    {"Text": "This is my favorite product. I buy it every month without fail.", "Score": 5, "Time": 1323780000},
    {"Text": "Very poor quality control. Found foreign objects inside the package.", "Score": 1, "Time": 1320816000},
    {"Text": "Delicious and fresh. Exactly what I was looking for. Great product.", "Score": 5, "Time": 1326384000},
    {"Text": "Disappointing purchase. The product looks nothing like the pictures.", "Score": 2, "Time": 1319904000},
    {"Text": "Incredible flavor and aroma. This is hands down the best I have tried.", "Score": 5, "Time": 1324128000},
    {"Text": "Product was expired when it arrived. Completely unacceptable.", "Score": 1, "Time": 1317168000},
    {"Text": "Really happy with this purchase. Fresh product and fast delivery.", "Score": 4, "Time": 1327920000},
    {"Text": "Terrible quality. Fell apart after one use. Very poorly made.", "Score": 1, "Time": 1318128000},
    {"Text": "Best product in its category. Consistent quality and great taste.", "Score": 5, "Time": 1325520000},
    {"Text": "Not fresh at all. Tasted stale and old. Will not buy again.", "Score": 2, "Time": 1321920000},
    {"Text": "Amazing product. My kids love it and I feel good about what they eat.", "Score": 5, "Time": 1323432000},
    {"Text": "Broken on arrival. Packaging was inadequate for shipping.", "Score": 1, "Time": 1320384000},
    {"Text": "Great purchase. Exactly as described and arrived in perfect condition.", "Score": 4, "Time": 1326816000},
    {"Text": "Waste of money. Completely different from what was advertised online.", "Score": 1, "Time": 1319472000},
    {"Text": "Superb quality and taste. Will definitely be ordering more of this.", "Score": 5, "Time": 1324656000},
    {"Text": "Very average product. Nothing special. Could find better elsewhere.", "Score": 3, "Time": 1321200000},
    {"Text": "Excellent product. Great value and shipped quickly. Very satisfied.", "Score": 5, "Time": 1327248000},
    {"Text": "Product was moldy when it arrived. Disgusting. Health hazard.", "Score": 1, "Time": 1317600000},
    {"Text": "Good product but slightly overpriced compared to similar alternatives.", "Score": 4, "Time": 1325184000},
    {"Text": "Absolutely horrible. Nothing worked as promised. Deeply disappointed.", "Score": 1, "Time": 1320048000},
    {"Text": "Love this product. Has become a daily staple in our household.", "Score": 5, "Time": 1323996000},
    {"Text": "Below average quality. Expected more given the price point.", "Score": 2, "Time": 1318320000},
    {"Text": "Perfect product. Does exactly what it says. Would buy again.", "Score": 5, "Time": 1326528000},
    {"Text": "Terrible experience from start to finish. Avoid at all costs.", "Score": 1, "Time": 1319616000},
]


def get_demo_dataframe() -> pd.DataFrame:
    """
    Returns a demo DataFrame with 50 real-looking reviews.
    This ships with the code so deployment works without
    the processed CSV files.
    """
    df = pd.DataFrame(DEMO_REVIEWS)
    df['date'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
    df['year'] = df['date'].dt.year
    df['cleaned_text'] = df['Text']
    df['sentiment'] = df['Score'].apply(
        lambda x: 1 if x >= 4 else 0
    )
    df['sentiment_label'] = df['sentiment'].map(
        {1: 'Positive', 0: 'Negative'}
    )
    return df