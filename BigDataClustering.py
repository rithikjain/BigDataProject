from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt

spark = SparkSession.builder.master('local[*]').appName('test_spark_app').getOrCreate()

df = spark.read.csv("file:///C:/Users/rithi/Downloads/dataset.csv", inferSchema =True, header=True)
df1 = df.toPandas()

df.printSchema()
df.describe()

# Running Query for College Stress
# Bar Chart
fig, ax = plt.subplots()
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = df1['college_stress'].to_list()
counts = []
for i in labels:
    counts.append(x.count(i))
rects1 = ax.bar(labels, counts, 0.8, color='r')

ax.set_ylim(0,250)
ax.set_ylabel('Frequency')
ax.set_xlabel('Stress Score')
ax.set_title('Stress Count')
plt.show()

#Creating Visualization
fig = plt.figure(figsize =(10, 7))
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x = df1['college_stress'].to_list()
counts = []
for i in labels:
    counts.append(x.count(i))
plt.pie(counts, labels = labels)
plt.title("Stress Level faced due to College")
# show plot
plt.show()

#Running Query for faculty involvement
#Creating Visualization
fig = plt.figure(figsize =(10, 7))
labels = [1, 2, 3, 4, 5]
x = df1['faculty_involvement'].to_list()
counts = []
for i in labels:
    counts.append(x.count(i))
plt.pie(counts, labels = labels, autopct='%1.2f%%')
# show plot
plt.title("How important is faculty involvement?")
plt.show()

#Running Query for friends involvement
#Creating Visualization
fig = plt.figure(figsize =(10, 7))
labels = [1, 2, 3, 4, 5]
x = df1['friend_timetable'].to_list()
counts = []
for i in labels:
    counts.append(x.count(i))
plt.pie(counts, labels = labels, autopct='%1.2f%%')
# show plot
plt.title("How important is having friends in the same class?")
plt.show()

#Running Query 9 pointers
#Creating Visualization
fig = plt.figure(figsize =(10, 7))
labels = [1, 2, 3, 4, 5]
x = df1['9_pointers'].to_list()
counts = []
for i in labels:
    counts.append(x.count(i))
plt.pie(counts, labels = labels, autopct='%1.2f%%')
# show plot
plt.title("Does having too many 9 pointers affect you in dropping?")
plt.show()

# Preprocessing
def index_column(df, column_name, output_column):
    indexer = StringIndexer(inputCol=column_name, outputCol=output_column) 
    indexed = indexer.fit(df).transform(df)

    df = indexed
    df = df.drop(column_name)
    return df

df = index_column(df, 'Which of the following course category will you most likely drop?', 'courseCat')
df = index_column(df, "Assuming you have the given number of credits, how likely is it for you to drop a course? [23]", 'cred_23')
df = index_column(df, 'Assuming you have the given number of credits, how likely is it for you to drop a course? [24]', 'cred_24')
df = index_column(df, 'Assuming you have the given number of credits, how likely is it for you to drop a course? [25]', 'cred_25')
df = index_column(df, 'Assuming you have the given number of credits, how likely is it for you to drop a course? [26]', 'cred_26')
df = index_column(df, 'Assuming you have the given number of credits, how likely is it for you to drop a course? [27]', 'cred_27')
print(df.columns)

df = df.drop('Timestamp')

# K Means
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = df.columns, outputCol = 'features')

final_data = assembler.transform(df)

from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol = 'features', outputCol = 'scaledFeatures')
scaler_model = scaler.fit(final_data)
final_data = scaler_model.transform(final_data)
kmeans = KMeans(featuresCol = "scaledFeatures", k = 5)
model = kmeans.fit(final_data)

# Evaluating
from pyspark.ml.evaluation import ClusteringEvaluator
# Make predictions
predictions = model.transform(final_data)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

