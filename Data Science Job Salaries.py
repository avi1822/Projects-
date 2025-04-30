# Data Preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path ="C:/Users/Sanika/Downloads/Data Science Job Salaries.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Change abbreviations to full country names
import country_converter as cc

cc = cc.CountryConverter()
data['company_location'] = cc.convert(data['company_location'], to='name_short')
data['employee_residence'] = cc.convert(data['employee_residence'], to='name_short')

# Rename and map job_type values
data.rename(columns={'remote_ratio': 'job_type'}, inplace=True)
data['job_type'] = data['job_type'].map({100: 'remote', 0: 'onsite', 50: 'hybrid'})

# Quick glance at the data
print(data.head())

# Data Analysis
# Salary Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data['salary'], bins=30, kde=True)
plt.title('Salary Distribution', fontsize=16)
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Mean Salary vs Experience Level
mean_salary_exp_level = data.groupby('experience_level')['salary'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_salary_exp_level.index, y=mean_salary_exp_level.values, palette='spring')
plt.title('Mean Salary vs Experience Level', fontsize=16)
plt.xlabel('Experience Level')
plt.ylabel('Mean Salary')
plt.show()

# Employment Type vs Salary
mean_salary_emp_type = data.groupby('employment_type')['salary'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_salary_emp_type.index, y=mean_salary_emp_type.values, palette='autumn')
plt.title('Mean Salary vs Employment Type', fontsize=16)
plt.xlabel('Employment Type')
plt.ylabel('Mean Salary')
plt.show()

# Company Size vs Salary
mean_salary_company_size = data.groupby('company_size')['salary'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_salary_company_size.index, y=mean_salary_company_size.values, palette='spring')
plt.title('Mean Salary vs Company Size', fontsize=16)
plt.xlabel('Company Size')
plt.ylabel('Mean Salary')
plt.show()

# Job Type vs Salary
mean_salary_job_type = data.groupby('job_type')['salary'].mean()
plt.figure(figsize=(12, 6))
sns.barplot(x=mean_salary_job_type.index, y=mean_salary_job_type.values, palette='Set2')
plt.title('Mean Salary vs Job Type', fontsize=16)
plt.xlabel('Job Type')
plt.ylabel('Mean Salary')
plt.show()

# Top 10 Data Science Roles by Mean Salary
top_roles_by_salary = data.groupby('job_title')['salary'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(14, 6))
sns.barplot(y=top_roles_by_salary.index, x=top_roles_by_salary.values, palette='Set2')
plt.title('Top 10 Data Science Roles by Mean Salary', fontsize=16)
plt.xlabel('Mean Salary')
plt.ylabel('Job Title')
plt.show()

# Top 10 Countries by Mean Salary
top_countries_by_salary = data.groupby('company_location')['salary'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(14, 6))
sns.barplot(y=top_countries_by_salary.index, x=top_countries_by_salary.values, palette='Set2')
plt.title('Top 10 Countries by Mean Salary', fontsize=16)
plt.xlabel('Mean Salary')
plt.ylabel('Country')
plt.show()
