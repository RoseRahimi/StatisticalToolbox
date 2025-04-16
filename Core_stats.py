#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:28:44 2024

@author: fatimarahimi
"""
import pylab
import scipy.stats as stats
from scipy.stats import tukey_hsd
from scipy.stats import ttest_ind
from scipy.stats import norm
from numpy import sqrt


def mean(data):
    return sum(data) / len(data)

#sum of squares computational
def sos_comp(data):
    sum1 = sum(data)  # sum of data points
    # calculating the sum of x squared
    sum1_squared = 0
    for x in data:
        sum1_squared +=  x ** 2
    N = len(data)
    # computational formula
    ss = sum1_squared - (sum1 ** 2) / N 
    return ss

# calculates variance of a sample
def var_sample(data):
    # sum of squares comp formula
    ss = sos_comp(data) 
    N = len(data)
    # Variance formula for sample
    return round(ss / (N - 1) ,3) 

 #  calculates standard deviation of a sample
def std_dev_samp(data):
     # call sample variance function
     var_samp = var_sample(data) 
     return round(pylab.sqrt(var_samp) , 3)

#n is number of scores whithin each condition
#g of conditions
# SS Between function
def SSbet(groups):
    # Calculate the sum of squared group means
    sum_of_squared_means = 0
    total_n = 0
    sum_of_means = 0
    
    for group in groups:
        group_mean = mean(group)
        #print(group_mean)
        n = len(group)
        total_n += n
        sum_of_means += group_mean
        #print(sum_of_means)
        sum_of_squared_means += (group_mean ** 2)
        #print(sum_of_squared_means)
    
    # Calculate the second term: square of the sum of means divided by the number of groups
    num_groups = len(groups)
    squared_sum_of_means = (sum_of_means ** 2) / num_groups
    
    
    #print(squared_sum_of_means)
    
    ssb = n * (sum_of_squared_means - squared_sum_of_means)
    
    return round(ssb, 3)

def SSWith(groups):
    sswith = 0  #  starting it with 0
    for group in groups:
        sswith += sos_comp(group)  # add the sum of squares of the current group
    return round(sswith, 3)

def dfbet(groups):
    dfbet = len(groups) - 1 
    return dfbet     
 
def dfwith (groups):
    N = 0
    for group in groups:
        N += len(group)
    return N - len(groups)


def msbet(groups):
    msbet = SSbet(groups)/dfbet(groups)
    return round(msbet,3)


def mswith(groups):
    mswith = SSWith(groups)/dfwith(groups)
    return round(mswith,3)


def F(groups):
    f = msbet(groups)/ mswith(groups)
    return round(f,3)


# Function to calculate F-statistic and p-value
def calculate_pvalue(groups):
    # Total number of observations (N) and number of groups (g)
    N = sum(len(group) for group in groups)
    g = len(groups)
    
    # Calculate SSB and SSW
    ssb = SSbet(groups)  # Sum of Squares Between
    ssw = SSWith(groups)  # Sum of Squares Within
    
    # Degrees of freedom
    dfb = g - 1  # Between groups
    dfw = N - g  # Within groups
    
    # Mean squares
    msb = ssb / dfb  # Mean Square Between
    msw = ssw / dfw  # Mean Square Within
    
    # F-statistic
    F = msb / msw
    
    # p-value
    p_value = 1 - stats.f.cdf(F, dfb, dfw)
   
    
    return round(p_value, 5)


def partial_eta_squared(groups):
    effect_size = SSbet(groups) / (SSbet(groups) + SSWith(groups))
    return round(effect_size,3)


def perform_tukey_hsd(groups, alpha=0.05):
   
    # Pass each group to the tukey_hsd function
    result = tukey_hsd(*groups)  # Unpack the groups into individual arguments
    print(result)
    # retrieve comparison details
    significant_results = []
    for i in range(len(groups)):
       for j in range(i + 1, len(groups)):
           # Check if the p-value for this comparison is significant
           if result.pvalue[i, j] < alpha:
               significant_results.append(
                   (
                       f"Group {i + 1} vs Group {j + 1}",
                       f"Statistic: {result.statistic[i, j]:.3f}",
                       f"P-value: {result.pvalue[i, j]:.5f}",
                       f"Significant at alpha={alpha}: Yes"
                   )
               )
   
   # if no significant results
    if not significant_results:
       return "No significant differences were found."

    return significant_results

# Define the paired samples t-test function
def paired_samples_t_test(sample1, sample2):
    
    # Calculate the differences between paired samples
    differences = [sample1[i] - sample2[i] for i in range(len(sample1))]
    #differences = sample1 - sample2
    # Calculate mean difference and standard deviation of differences
    mean_diff = sum(differences) / len(differences)
    variance_diff = sum((x - mean_diff) ** 2 for x in differences) / (len(differences) - 1)
    std_dev_diff = pylab.sqrt(variance_diff)
    
    # Degrees of freedom
    df = len(differences) - 1
    
    # Calculate the t-statistic and p-value
    t_stat, p_value = stats.ttest_1samp(differences, 0)
    ttest = round(t_stat, 3)
    pval = round(p_value, 3)
    # Calculate the 95% confidence interval
    confidence_level = 0.95
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
   
    margin_of_error = t_critical * (std_dev_diff / pylab.sqrt(len(differences)))
    confidence_interval = (round(float(mean_diff - margin_of_error), 3), round(float(mean_diff + margin_of_error), 3))
    
    # Display results
    print(f"Mean Difference:                            {round(mean_diff,3)}")
    print(f"Standard Deviation of Difference:           {round(std_dev_diff,3)}")
    print(f"T-Statistic:                                {ttest}")
    print(f"Degrees of Freedom:                         {df}")
    print(f"Confidence Interval (95%):                  {confidence_interval}")
    print(f"P-Value (2-tailed):                         {pval}")
    
def independent_ttest(data1, data2, label1="Group 1", label2="Group 2"):
    """Performs an independent samples t-test on two datasets and displays the t-value, df, and p-value."""
    # Perform the t-test assuming equal variance
    t_stat, p_value = ttest_ind(data1, data2, equal_var=True)
    
    # Calculate degrees of freedom
    df = len(data1) + len(data2) - 2
    
    # Display results
    print(f"Independent Samples T-Test between {label1} and {label2}:")
    print(f"  t-value: {t_stat:.3f}")
    print(f"  Degrees of Freedom (df): {df}")
    print(f"  p-value: {p_value:.3f}")
    
    # Interpret p-value
    if p_value < 0.05:
        print("  Result: The difference is statistically significant (p < 0.05)")
    else:
        print("  Result: The difference is not statistically significant (p >= 0.05)")

    #to find the z scores
def zScore (data):
    mean_data = mean(data)
    st_dev = std_dev_samp(data)
    final = [round(float((datai - mean_data ) / st_dev) ,3) for datai in data]
    return final


# Pearson correlation function
def pCorrelation(x, y):
    z_x = zScore(x)
    z_y = zScore(y)
    
    # Element-wise multiplication of z-scores
    r_top = sum(z_x[i] * z_y[i] for i in range(len(z_x)))
    r_final = r_top / (len(x) - 1)

    return round(r_final, 3)


# Effect size calculation (r squared)
def effect_size(x, y):
    r = pCorrelation(x, y)
    r_squared = r ** 2
    return round(r_squared, 3)


# Define the zTestMean function
def zTestMean(sMean, nSamples, normMean, stdDev, oneSided=True):
    # Calculate the z-score
    zScore = abs((sMean - normMean) / (stdDev / sqrt(nSamples)))
    print(zScore)
    
    # Calculate the probability using the cumulative distribution function
    prob = norm.cdf(-zScore)
    
    # For a two-tailed test, multiply the probability by 2
    if not oneSided:
        prob *= 2
    
    return prob

# Define the Cohen's d function (effect size calculation)
def cohen_d(sample_mean, population_mean, population_std):
    return (sample_mean - population_mean) / population_std


# Function to calculate the t-statistic for a single-sample t-test
def t_statistic(sample, population_mean):
    sample_mean = mean(sample)
    sample_std = std_dev_samp(sample)
    n = len(sample)
    t_stat = (sample_mean - population_mean) / (std_samp / sqrt(n))
    return t_stat

# Function to calculate p-value from t-statistic
def p_value_from_t(t_stat, df, one_sided=False):
    # Calculate p-value for a one-tailed test
    p_value = 1 - stats.t.cdf(abs(t_stat), df)
    
    # For a two-tailed test, multiply the p-value by 2
    if not one_sided:
        p_value *= 2
    
    return p_value


