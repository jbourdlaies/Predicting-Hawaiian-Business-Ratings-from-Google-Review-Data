# Predicting-Hawaiian-Business-Ratings-from-Google-Review-Data

My final project for DSC80 at UCSD

## Introduction

This project analyzes Google Maps business review data from Hawaii using two datasets: one containing individual customer reviews and another containing metadata about businesses such as category, average rating, and number of reviews. These datasets are linked through a shared identifier (`gmap_id`), allowing us to connect user-level feedback with business-level characteristics.

The goal of this project is to understand what factors are associated with higher or lower review ratings, and whether business features can be used to predict individual reviews.

### Question

Can we predict Google business review ratings using business characteristics and review metadata in Hawaii?

### Significance

Understanding what drives customer ratings is important for business owners, especially in tourism-heavy regions like Hawaii. If it is possible to identify which factors (such as business type, popularity, or existing rating trends) influence customer satisfaction, businesses can make more informed decisions about operations and service improvements.

### Dataset Overview

- Number of review records: 1,504,347
- Number of business records: 21,507

### Relevant Columns

#### Reviews dataset
- `rating`: Star rating of 1-5 given by the user (target variable)
- `text`: Written review content
- `time`: Timestamp of the review
- `gmap_id`: Business identifier used for joining datasets

#### Business metadata dataset
- `category`: Business category (such as restaurant or store)
- `avg_rating`: Average rating of 1-5 of the business on Google Maps
- `num_of_reviews`: Total number of reviews for the business
- `price`: Price level (when available)
- `gmap_id`: Unique business identifier

## Data Cleaning and Exploratory Data Analysis

Several data cleaning steps were performed to prepare the dataset for analysis. These steps are based on the structure of the raw data and the way it was generated (user-generated reviews linked to business metadata).

### Type conversions

Key identifier columns such as user_id and gmap_id were converted to strings. Although these values may appear numeric, they function as categorical identifiers used to link users, reviews, and businesses. Treating them as strings prevents unintended numerical operations and preserves identifier integrity.

### Datetime conversion

The time column in the reviews dataset was originally stored as a Unix timestamp in milliseconds. This was converted into a proper datetime format using pd.to_datetime(), allowing better analysis of trends over time and extraction of features like year and month.

### Missing text handling

Missing values in the text column were replaced with NaN. This ensures consistency in missing value representation and allows us to analyze whether the absence of written reviews is meaningful (such as users leaving only star ratings).

### Merging datasets

The reviews dataset was merged with the business metadata dataset using the shared gmap_id column. This combines user-level review data with business-level attributes such as category, average rating, and number of reviews, enabling critical analysis.

### Feature extraction (primary_category)

The category column in the metadata contains lists of multiple business categories. To simplify analysis, the first category was extracted as primary_category, representing the main business type. This makes grouping and visualization more interpretable.

### Cleaned Dataset Preview

|   rating | primary_category   |   avg_rating |   num_of_reviews | datetime                   |
|---------:|:-------------------|-------------:|-----------------:|:---------------------------|
|        5 | Recreation center  |          4.1 |               18 | 2020-06-11 01:45:03.487000 |
|        5 | Recreation center  |          4.1 |               18 | 2020-06-11 01:45:03.487000 |
|        5 | Recreation center  |          4.1 |               18 | 2019-09-09 19:56:58.979000 |
|        5 | Recreation center  |          4.1 |               18 | 2019-09-09 19:56:58.979000 |
|        5 | Recreation center  |          4.1 |               18 | 2020-07-16 07:46:28.335000 |
|        5 | Recreation center  |          4.1 |               18 | 2020-07-16 07:46:28.335000 |
|        5 | Recreation center  |          4.1 |               18 | 2019-12-10 04:12:11.613000 |
|        5 | Recreation center  |          4.1 |               18 | 2019-12-10 04:12:11.613000 |
|        3 | Recreation center  |          4.1 |               18 | 2019-11-06 21:45:23.916000 |
|        3 | Recreation center  |          4.1 |               18 | 2019-11-06 21:45:23.916000 |

### Univariate Analysis

<iframe src="assets/Distribution_of_Review_Ratings_Histogram.html" width="800" height="600" frameborder="0" ></iframe>
Distribution of Review Ratings: Most reviews are concentrated at the high end of the scale (4 to 5 stars), showing a strong positive bias typical of online review platforms.

<iframe src="assets/Top_10_Business_Categories_by_Number_of_Reviews_Bar_Chart.html" width="800" height="600" frameborder="0" ></iframe>
Top 10 Busineess Categories by Number of Reviews: A small number of business categories account for most reviews, indicating that user activity is heavily concentrated in a few popular business types.

### Bivariate Analysis

<iframe src="assets/Ratings_by_Top_Business_Categories_Boxplot.html" width="800" height="600" frameborder="0" ></iframe>
Ratings by Top Business Categories: Some business categories show noticeable variation in rating distributions, suggesting that customer satisfaction depends on the type of business.

<iframe src="assets/Average_Rating_vs_Number_of_Reviews_Scatterplot.html" width="800" height="600" frameborder="0" ></iframe>
Average Rating vs Number of Reviews: There is little to no strong relationship between the number of reviews and average rating, suggesting that popularity does not strongly predict satisfaction.

### Grouped Table Analysis

| primary_category     |   avg_rating |   review_count |
|:---------------------|-------------:|---------------:|
| Shopping mall        |      4.25189 |         114600 |
| Restaurant           |      4.29301 |          79488 |
| Park                 |      4.51472 |          75867 |
| Fast food restaurant |      3.90775 |          50191 |
| Hawaiian restaurant  |      4.40168 |          47886 |
| Seafood restaurant   |      4.39293 |          45688 |
| Hotel                |      4.34842 |          45640 |
| Grocery store        |      4.30913 |          36845 |
| Tourist attraction   |      4.52697 |          33095 |
| Coffee shop          |      4.38879 |          30114 |
This table highlights which business categories are most common in the dataset and how they perform in terms of average rating. While some categories dominate in volume, their ratings remain relatively similar, suggesting that popularity does not necessarily imply higher satisfaction.

## Assessment of Missingness

I investigated whether the missingness of the text column in the reviews dataset is related to observed or unobserved factors, and whether it can be classified as NMAR (Not Missing At Random).

I believed that the text column may be NMAR, since whether a user leaves a written review likely depends on unobserved factors such as their level of engagement, motivation, or effort. For example, users who leave very quick or neutral ratings may choose not to write a review at all, even though this behavior is not fully explained by variables already present in the dataset. To better explain this missingness mechanism and potentially convert it into MAR (Missing at Random), additional data such as time spent writing a review, whether users were prompted for written feedback, or user review history would be useful.

To further investigate missingness, I conducted permutation tests to determine whether the absence of review text is independent of several observed variables.

First, I tested whether missingness of text depends on the review rating. The observed test statistic was approximately 0.053, with a p-value of 0.0. Since the p-value is extremely small, the null hypothesis is rejected and it can be concluded that missingness in review text is dependent on rating. This suggests that users giving different ratings have different likelihoods of leaving written feedback.

Next, I tested whether missingness depends on primary_category. The observed statistic was approximately 0.144, again with a p-value of 0.0. The null hypothesis can be rejected here as well, indicating that the likelihood of leaving written feedback varies across business categories.

Finally, I tested whether missingness depends on gmap_id (individual businesses). The observed statistic was approximately 0.538 with a p-value of 0.498. In this case, the null hypothesis fails to be rejected, suggesting that missingness does not significantly depend on individual business identity.

### Missingness Visualization

To visualize missingness behavior, I created a Plotly histogram comparing the distribution of ratings for reviews with missing text versus those with written text.

<iframe
  src="Rating_Distribution_Missing_vs_Not_Missing_Review_Text.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
This plot shows that the distribution of ratings differs between reviews with and without written text. In particular, higher ratings appear more likely to have missing text, suggesting that satisfied users may actually be less inclined to leave detailed written feedback.

Overall, these results suggest that missingness in the text column is not random. It is influenced by observable variables such as rating and category, while also potentially depending on unobserved user behavior, making NMAR a plausible explanation.

## Hypothesis Testing

In this step, I investigated whether different business categories in Hawaii have statistically different average review ratings.

### Null/Alternate Hypothesis

The null hypothesis states that all selected business categories have the same average rating, and any observed differences are due to random chance. The alternative hypothesis states that at least one business category has a different average rating. This question is important because it helps determine whether customer satisfaction is consistent across different types of businesses or whether certain categories tend to receive systematically higher or lower ratings.

To test this, the standard deviation of the mean ratings across business categories is a good test statistic. This is an appropriate choice because it directly measures how spread out the category-level average ratings are. If all categories truly have similar ratings, this value should be small. Larger values indicate greater differences between categories, which would provide evidence against the null hypothesis.

I conducted a permutation test by randomly shuffling ratings across categories and recomputing the test statistic many times. This simulates the distribution of the test statistic under the assumption that category has no relationship with rating. Here, the significance level of 0.05 is used, which is a standard threshold for determining whether an observed effect is unlikely to have occurred by chance.

The observed test statistic is approximately 0.229, and the resulting p-value is approximately 0.0. Since the p-value is extremely small and below the significance level, the null hypothesis is rejected. This suggests that it is unlikely that all business categories share the same average rating.

However, this conclusion should not be interpreted as absolute proof that category determines rating. Instead, it indicates that the differences observed between categories are unlikely to be explained by random variation alone.

The chosen test statistic is appropriate because it captures variation in average ratings across categories, which directly relates to the research question. The permutation test is also suitable because it does not rely on strong distributional assumptions and instead builds an empirical null distribution by breaking any relationship between category and rating.

## Framing a Prediction Problem

In this project, the problem is a regression task, where the goal is to predict the numerical star rating of a Google Maps review in Hawaii. The response variable is rating, which ranges from 1 to 5 and represents the user’s evaluation of a business. This variable was chosen because it directly reflects customer satisfaction at the review level and serves as a natural quantitative target for understanding how business and review characteristics influence user opinions.

The Root Mean Squared Error (RMSE) is the primary evaluation metric. RMSE is appropriate for this problem because it measures the average magnitude of prediction error in the same units as the target variable (stars). This makes it easy to interpret how far, on average, predicted ratings are from actual ratings. Compared to metrics such as Mean Absolute Error (MAE), RMSE places a slightly higher penalty on larger errors, which is useful in this context because large mispredictions (such as predicting a 5 star review as 1 star) are more undesirable than small deviations. Accuracy is not suitable here because the target variable is continuous rather than categorical.

To ensure a realistic prediction scenario, only features that would be available at the time the rating is made or could be known beforehand should be used. These include business-level attributes such as primary_category, avg_rating, and num_of_reviews, as well as engineered features derived from the review text and timestamp (such as text length, whether text is present, and temporal features like review month and year). Importantly, any information that would directly leak the target variable or depend on future knowledge must be avoided.

Overall, this formulation allows us to model how both business characteristics and review behavior relate to individual user ratings while maintaining a realistic and time-consistent prediction setup.

## Baseline Model

I first built a baseline model to predict the rating variable in order to establish a simple reference point for model performance. This is a regression problem, since the target variable is continuous and represents star ratings ranging from 1 to 5.

The baseline model uses three features: primary_category, avg_rating, and num_of_reviews. The avg_rating and num_of_reviews features are quantitative variables that capture the overall quality and popularity of a business. The primary_category feature is a nominal categorical variable representing the type of business (such as restaurant, park, or hotel). Since machine learning models require numerical inputs, we applied one-hot encoding to primary_category to convert it into a set of binary indicator variables. The numeric features were passed through without transformation.

I used a linear regression model trained on these processed features. This baseline model is intentionally simple and assumes a linear relationship between the input features and the review rating. Its purpose is to provide a benchmark for evaluating whether more complex feature engineering and modeling approaches meaningfully improve predictive performance. Since it has a RMSE of approximately 0.889, which is relatively small, I would say the model is adequate. However, marginal improvements could still be made, as seen with the final model.

## Final Model

In the final model, I expanded the feature set and introduced a more flexible modeling approach to better capture nonlinear relationships in the data-generating process.

The additional features include text_length, has_text, review_year, and review_month. The text_length and has_text features are derived from the review text and reflect user engagement, where longer reviews or the presence of written feedback may indicate stronger opinions. The review_year and review_month features capture temporal variation in review behavior, allowing the model to account for potential changes in user preferences or business performance over time. These features are useful because review ratings are influenced not only by business characteristics but also by how and when users interact with the platform.

I selected a Random Forest Regressor as the final modeling algorithm because it can model complex nonlinear relationships and interactions between features without requiring explicit functional assumptions. This makes it well-suited for structured tabular data where relationships between variables are not strictly linear.

To optimize model performance, I used GridSearchCV with 3-fold cross-validation to tune key hyperparameters, including n_estimators, max_depth, and min_samples_leaf. The best-performing configuration used 100 trees (n_estimators = 100), a maximum depth of 10 (max_depth = 10) or no restriction depending on the split, and minimum leaf sizes of 1 or 5. These hyperparameters control model complexity and help balance the bias-variance tradeoff by limiting overfitting while maintaining expressive power.

The final model was evaluated using Root Mean Squared Error (RMSE), resulting in a test error of approximately 0.881. This represents an improvement over the baseline model RMSE of approximately 0.889. Although the improvement is modest, it suggests that nonlinear modeling combined with additional engineered features provides more predictive signal than a simple linear approach, even though individual review ratings remain inherently noisy and difficult to predict precisely.

## Fairness Analysis

Does the performance of the final model differs between businesses with low review counts and those with high review counts? Let's define Group X as businesses with a number of reviews less than or equal to the median number of reviews, and Group Y as businesses with a number of reviews greater than the median. The goal of this analysis is to determine whether the model performs differently depending on how much information is available about a business, since businesses with fewer reviews may be harder to predict accurately.

I use Root Mean Squared Error (RMSE) as the evaluation metric because it measures the average magnitude of prediction error in the same units as the target variable (star ratings). RMSE is appropriate here because it penalizes larger prediction errors more heavily, which is important when evaluating model fairness across groups.

The null hypothesis states that the model is fair, meaning that the difference in RMSE between Group X (low review count businesses) and Group Y (high review count businesses) is due to random chance. The alternative hypothesis states that the model is unfair, meaning that RMSE is higher for businesses with fewer reviews and that the observed difference is not due to random variation.

To test this, the difference in RMSE between the two groups can be used as the test statistic (RMSE for Group X minus RMSE for Group Y). I performed a permutation test by randomly shuffling group labels many times to generate a null distribution of the test statistic. As usual a significance level of 0.05 is used to determine whether the observed difference is statistically significant.

The observed test statistic is approximately 0.082, and the resulting p-value is approximately 0.0. Since the p-value is below the significance level, the null hypothesis is rejected. This suggests that the difference in model performance between low-review and high-review businesses is unlikely to be due to random chance.

However, this does not imply absolute unfairness in a causal sense. Instead, it indicates that model performance appears to vary systematically between the two groups, likely due to differences in data availability. Businesses with fewer reviews provide less information for the model to learn from, which may lead to higher prediction error in that group. Another limitation is that 
this is a “10-core” dataset, meaning all remaining users and businesses have at least 10 reviews. As a result, the data may overrepresent more active reviewers and more popular businesses, introducing potential selection bias into the analysis.