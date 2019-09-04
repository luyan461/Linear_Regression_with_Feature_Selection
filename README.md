# Linear_Regression_with_Feature_Selection
In this project, we are going to implement and discuss Exploratory Data Analysis with various fundamental concepts in linear regression using RStudio.

**Keywords**: Linear regression, diagnostic plots, multicollinearity, model selection criteria, stepwise regression algorithms

## Outline
1. Understanding and Loading Data
2. Visualization and Prelimarary Exploration
3. Diagnostic Plots
4. Model Comparisons
5. Sequential Variable Selection Algorithm
6. Summary

## Understanding and Loading Data
The first file we are going to explore is *Pollution.csv*, which contains data on 60 U.S. cities with variables describe below

* City = Name of the city.
* Mort = Total age-adjusted mortality from all causes, in deaths per 100,000 population 
* Precip = Mean annual precipitation (in inches)
* Educ = Median number of school years completed for people aged 25 or older
* NonWhite = Percentage of 1960 population that is nonwhite
* NOX = Relative pollution potential* of oxides of nitrogen
* SO2 = Relative pollution potential of sulfur dioxide

We are interested in modeling mortality here. Thus, *Mort* is the response variable. Let's first load the data as below.

```
# set your own working directory and load the Pollution.csv data
setwd("C:/Users/Metalboy/Desktop/Course_Project/Assignments")
Pollution <- read.csv("Pollution.csv", header=T)
attach(Pollution)
```

## Visualization and Prelimarary Exploration
Let’s explore the response variable vs pollution/climate/socioeconomic variables by first starting with a matrix of scatterplots as below. Since *City* is neither the response variable nor the predictor, we can exclude it from the dataframe.

```
# exclude the variable City
Pollution <- within(Pollution, rm(City))
# matrix of scatterplots
pairs(Pollution, pch=16)
```
<img src="Figures/matrix of scatterplots.png">

From the above plot, it does not seem to be that *Mort* is strongly associated with either of the pollution variables, i.e., *NOX* or *SO2*. If we further check the correlation matrix, it proves that there is no strong association between *Mort* and *NOX* as the correlation value is very close to 0. However, there is some association between *Mort* and *SO2*.

```
# matrix of correlations
signif(cor(Pollution), digits=3)
```

|        |  Mort  | Precip |  Educ | NonWhite |  NOX  |  SO2  |
|--------|--------|--------|-------|----------|-------|-------|
|Mort    |  1.0000|  0.509 |-0.511 |  0.6440  |-0.0774|  0.426|
|Precip  |  0.5090|  1.000 |-0.490 |  0.4130  |-0.4870| -0.107|
|Educ    | -0.5110| -0.490 | 1.000 | -0.2090  | 0.2240| -0.234|
|NonWhite|  0.6440|  0.413 |-0.209 |  1.0000  | 0.0184|  0.159|
|NOX     | -0.0774| -0.487 | 0.224 |  0.0184  | 1.0000|  0.409|
|SO2     |  0.4260| -0.107 |-0.234 |  0.1590  | 0.4090|  1.000|

It needs to be noticed that there seems to be two data points as outliers on *NOX* variable. An initial thought is to eliminate those two points since they might be measurement errors. However, if we check the initial spreadsheet, it turns out that those two points are *NOX* at 319 for Los Angeles and at 171 for San Francisco. Since these two are metropolis in California and the air quality is not really good there, those two measurements could be the real data that cannot be dropped for further analysis.

Instead of dropping those two points, we can consider some transformations of the predictor variables when necessary. In this case, we use log transformation of predictor variable *NOX* as *logNOX*.

```
# log transformation of predictor variable NOX
logNOX <- log(NOX)
Pollution1 <- cbind(Pollution, logNOX)
Pollution1 <- within(Pollution1, rm(NOX))
```

## Diagnostic Plots
Then we can fit a model containing all first-order predictors using the transformed *logNOX*, and we check the assumptions of the linear regression model using diagnostic plots.

```
# fit the model with all first-order predictors
model1 <- lm(Mort~Precip+Educ+NonWhite+SO2+logNOX, data=Pollution1)
plot(fitted.values(model1),residuals(model1),pch=16,xlab="Fitted Value",ylab="Residual",main=" ")
abline(h=0,lty=2)
```
<img src="Figures/residual plot_pollution.png">

The above residual plot suggests that variation in residuals is constant across all range of the predictor variables. Thus, the assumption of homoscedasticity seems reasonable. In addition, there is no strong evidence of systematic lack of fit and no serious outliers. Correspondingly, we can generate QQ plot as below.

```
qqnorm(residuals(model1),pch=16,main=" ")
```
<img src="Figures/QQ plot_pollution.png">

The above QQ plot further implies that there is no obvious evidence for non-normal distribution of the residuals. Furthermore, we can check the variation inflation factor (VIF), which is obtained as

```
# VIF is in the car library
library(car)
vif(model1)
```
|Precip  |   Educ  | NonWhite|      SO2|   logNOX|
|--------|---------|---------|---------|---------|
|2.129191| 1.487353| 1.458193| 2.180626| 2.628977| 

VIF values are all less than 10, suggesting that there is no problem of multicollinearity in the model. Then, we can compute case-influence statistics to investigate any outliers and their influence in the model. We first check the Studentized Residuals as

```
plot(rstudent(model1), ylab="Studentized Residual")
abline(h=2,lty=2)
abline(h=-2,lty=2)
```
<img src="Figures/Studentized Residual.png">

Here the threshold is set at 2. From the above plot, it can be seen that there are two points above +2 and two points below -2, which are deemed as outliers. However, just because those four points are outliers does not mean we should worry. We worry if the outliers are heavily influencing other predictions. Therefore, we need to further calculate the Hat Matrix for influential as follows.

```
plot(hatvalues(model1), ylab="Diagonal Hat Matrix Elements")
abline(h=2*mean(hatvalues(model1)), lty=2)
```
<img src="Figures/Hat Matrix.png">

In Hat Matrix, if the diagonal h<sub>ii</sub> is over twice the average of all diagonal elements, then the i<sup>th</sup> data point is deemed influential. From the plot above, it can be seen that there are 6 points as influential outliers above the threshold at 0.2. Furthermore, we can use Cook’s distance as a deletion diagnostic to measure the influence of outliers as below.

```
plot(cooks.distance(model1), ylab="Cook's Distance")
abline(h=0.075, lty=2)
```
<img src="Figures/Cook's distance.png">

The dashed line is threshold set at 0.075, which comes from 4/(N-k-1) where N is the number of observations and k is the number of explanatory variables.

To summarize the above diagnostics for outliers, we can conclude that
1. From Studentized Residuals plot, it can be observed that there are four outliers. Concretely, data index 4 and 7 are less than threshold of -2, and data point 50 and 60 are larger than threshold of 2.
2. From Hat matrix diagonal elements, case index 3, 8, 27, 48, 56 and 60 are over the threshold and deemed as influential.
3. From Cook’s distance plot, data index 1, 4, 7, 8 and 60 are over the threshold and deemed as influential.
4. In general, data point 60 is definitely an influential outlier in this model. Data point 4 and 7 may be influential outliers as they only show up in Cook’s distance plot but are normal in Hat matrix. Data point 50 is an outlier, but it is not influential since it does not fall beyond reasonable threshold in either Cook’s distance plot or Hat matrix.

## Model Comparisons
Now let's expand the problem. The file *BigPollution.csv* contains more data on the 60 cities mentioned above. The variables include the same variables in *Pollution.csv*, plus the following:
* Humidity = Percent relative humidity (annual average at 1pm) 
* JanTemp = Mean January temperature (in degrees Fahrenheit) 
* JulyTemp = Mean July temperature (in degrees Fahrenheit) 
* Over65 = percentage of the population aged 65 years or over 
* House = Population per household (average)
* Sound = Percentage of housing units that are “sound” (having all facilities) 
* Density = Population density in persons per square mile of urbanized area 
* WhiteCol = Percentage of employment in white-collar occupations
* Poor = Percentage of households with annual income below $3,000
* HC = Relative pollution potential of hydrocarbons

In this case, there are 10 more potential predictor variables, and the response variable is still *Mort*. Based on model selection criteria, we can use a C<sub>p</sub> plot and the BIC to select a model using only socioeconomic and climate variables (no pollution variables) as explanatory.

```
BigPollution <- read.csv("BigPollution.csv", header=T)
attach(BigPollution)
library(car)
library(leaps)
model2 <- regsubsets(Mortality~Precip+Humidity+JanTemp+JulyTemp+Over65+House+Educ+
                     Sound+Density+NonWhite+WhiteCol+Poor, data=BigPollution)
subsets(model2, statistic="cp")
subsets(model2, statistic="bic")
```
<img src="Figures/Cp plot.png">
<img src="Figures/BIC plot.png">

In general, the lower the C<sub>p</sub> and BIC values, the better of the model fit. From the above plots, the minimum of C<sub>p</sub> occurs at subset size of 6, while the minimum of BIC is at subset size of 4. Therefore, to accommodate for both criteria, we could choose subset size of 6 since it is the minimum on C<sub>p</sub> plot and the second minimum on BIC plot. The explanatory variables will be *Precip*, *JanTemp*, *JulyTemp*, *Density*, *Educ* and *NonWhite* for the model. Then we can fit the model by including three pollution explanatory variables (transformed to their logarithms) as below.

```
logHC <- log(HC)
logNOX <- log(NOX)
logSO2 <- log(SO2)
BigPollution1 <- cbind(BigPollution, logHC, logNOX, logSO2)
model3 <- lm(Mortality~Precip+JanTemp+JulyTemp+Density+Educ+NonWhite+logHC+logNOX+logSO2, data=BigPollution1)
aov(model3)
```

The utility of adding the three predictor variables simultaneously to the starting model can be assessed using a partial *F*-test as well as the corresponding *p*-value. The calculation is skipped here, but the conclusion is that at a significance level of *α* = 0.01, it is better to include at least one of the three pollution variables with logarithms transformation in the model.

## Sequential Variable Selection Algorithm
We can explore the stepwise regression algorithms with both forward selection and backward elimination in RStudio.
```
step(model3, data=BigPollution1, direction="both")
```
From the output results, it can be seen that the stepwise regression with both directions choose to drop the *JulyTemp*, *Density* and *logSO2* variables. In this case, AIC value will be reduced from 426.7 to 424.65, which is more preferable. 
However, one thing that needs attention is the multicollinearity, where we can check the VIF values as follows. 

```
model4 <- lm(Mortality~Precip+JanTemp+Educ+NonWhite+logHC+logNOX, data=BigPollution1)
vif(model4)
```
|Precip  | JanTemp |    Educ | NonWhite|     logHC|    logNOX| 
|--------|---------|---------|---------|----------|----------|
|2.209209| 1.371826| 1.532457| 1.767443| 12.492852| 11.740626|

However, it seems that *logHC* and *logNOX* variables have the multicollinearity issue as their VIF values are above 10. Therefore, it is better to eliminate the *logHC* variable for the finalized model and double check VIF for multicollinearity.

Now we have come up with this finalized model, and we can further assess this finalized model using the regular diagnostic plots presented before. This calculation is skipped here, but it is highly recommended that you try it on your own to see if the model is good or not.

## Summary
In this project, we explored the data and fitted a regression model containing all first-order predictors. We investigated serveral key concepts, including diagnostic plots, multicollinearity, model selection and etc. Based on our analyses above, we come up with a finalized model where *Precip*, *JanTemp*, *Educ*, *NonWhite* and *logNOX* are the appropriate variables to best predict the response variable *Mort*.

If you have further questions or comments, please email: luyan461@gmail.com.
