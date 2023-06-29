from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, LogisticRegression, NaiveBayes#, OneRClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum, expr, stddev_samp, udf
from pyspark.sql.types import DoubleType

PREPROCESS = True
columns = [ "Age", "Flight Distance", "Inflight wifi service", "DepartureArrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness", "Departure Delay in Minutes"]
columns_normalized =  [ "Age_S",  "Flight Distance_S", "Inflight wifi service_S", "DepartureArrival time convenient_S", "Ease of Online booking_S", "Gate location_S", "Food and drink_S", "Online boarding_S", "Seat comfort_S", "Inflight entertainment_S", "On-board service_S", "Leg room service_S", "Baggage handling_S", "Checkin service_S", "Inflight service_S", "Cleanliness_S", "Departure Delay in Minutes_S"]

spark = SparkSession.builder.appName("Airline-company Classificator").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("./airline-company.csv")




if(PREPROCESS):
    print("\nDataset:\n")
    data.show()

    data = data.drop("row")
    print("\nrow kolona je uklonjena:")
    data.show()

    data = data.drop("id")
    print("\nid kolona je uklonjena:")
    data.show()

    data = data.drop("Gender")
    data = data.drop("Customer Type")
    data = data.drop("Type of Travel")
    data = data.drop("Class")
    data = data.drop("Arrival Delay in Minutes")

    print("\nDataset nakon uklonjenih kolona:\n")
    data.show()

    duplicate_count = data.dropDuplicates().count() - data.count()

    print(f"\nBroj dupliranih redova: {duplicate_count}\n")
    
    print("\nBroj null polja u koloni:")
    data.select([sum(col(column).isNull().cast("int")).alias(column) for column in data.columns]).show()

    print("\nDa li vrednosti atributa satisfaction balansirane?")
    data.groupBy("satisfaction").count().show()

    
    statistics = data.select([stddev_samp(column).alias(column) for column in columns]).first() 
    lower_bounds = [(statistics[column] - 3 * statistics[column]) for column in columns]
    upper_bounds = [(statistics[column] + 3 * statistics[column]) for column in columns]

    print("\nFiltrirani podaci:")
    data.createOrReplaceTempView("dataView") 

    query = 'SELECT * FROM dataView WHERE ' + \
            ' AND '.join([f'(`{column}` >= {lower_bound} \
                              AND `{column}` <= {upper_bound})' \
                               for column, lower_bound, upper_bound in \
                               zip(columns, lower_bounds, upper_bounds)])


    result = spark.sql(query)
    result.show()


    print("\nData: \n")
    data.show()
    print("\nNormalizovani podaci:")
    numeric_columns = [column for column in data.columns if column not in ["row", "id", "Gender", "Customer Type", "Type of Travel", "Class", "Arrival Delay in Minutes", "satisfaction"]]
   
    unlist = udf(lambda x: round(float(list(x)[0]),15), DoubleType())
    for i in numeric_columns:
        assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect") 
        scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_S") 
        pipeline = Pipeline(stages=[assembler, scaler])  
        data = pipeline.fit(data).transform(data).withColumn(i+"_S", unlist(i+"_S")).drop(i+"_Vect") 

    columns_to_drop = ["Age", "Flight Distance", "Inflight wifi service", "DepartureArrival time convenient", "Ease of Online booking",
                       "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
                        "Leg room service",  "Baggage handling", "Checkin service", "Inflight service", "Cleanliness", "Departure Delay in Minutes"]
    
    print("\nData: \n")
    data.show()
    data = data.drop(*columns_to_drop)

    print("\nData nakon dropa: \n")
    data.show()

    data = data.select(*( [col(c) for c in data.columns[1:]] + [col(data.columns[0])]))
    print("\nNew data:\n")
    data.show()


indexer = StringIndexer(inputCol="satisfaction", outputCol="label")
indexedData = indexer.fit(data).transform(data)

data.show()


assembler = VectorAssembler(inputCols=columns_normalized if PREPROCESS else columns, outputCol="features")
assembledData = assembler.transform(indexedData) 


(trainingData, testData) = assembledData.randomSplit([0.8, 0.2], seed=13156123) 


lr = LogisticRegression()
nb = NaiveBayes()
svm = LinearSVC(maxIter=10)
dt = DecisionTreeClassifier()


lrModel = lr.fit(trainingData) 
nbModel = nb.fit(trainingData) 
svmModel = svm.fit(trainingData) 
dtModel = dt.fit(trainingData)


lrPredictions = lrModel.transform(testData)
nbPredictions = nbModel.transform(testData)
svmPredictions = svmModel.transform(testData)
dtPredictions = dtModel.transform(testData)


lrEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
nbEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
svmEvaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
dtEvaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

lrAccuracy = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "accuracy"})
nbAccuracy = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "accuracy"})
svmAccuracy = svmEvaluator.evaluate(svmPredictions)
dtAccuracy = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "accuracy"})



lrPredictionAndLabels = lrPredictions.select("prediction", "label").rdd
lrMetrics = MulticlassMetrics(lrPredictionAndLabels)
lrConfusionMatrix = lrMetrics.confusionMatrix()
print("Logistic Regression Confusion Matrix:")
print(lrConfusionMatrix)


nbPredictionAndLabels = nbPredictions.select("prediction", "label").rdd
nbMetrics = MulticlassMetrics(nbPredictionAndLabels)
nbConfusionMatrix = nbMetrics.confusionMatrix()
print("Naive Bayes Confusion Matrix:")
print(nbConfusionMatrix)


svmPredictionAndLabels = svmPredictions.select("prediction", "label").rdd
svmMetrics = MulticlassMetrics(svmPredictionAndLabels)
svmConfusionMatrix = svmMetrics.confusionMatrix()
print("SVM Confusion Matrix:")
print(svmConfusionMatrix)


dtPredictionAndLabels = dtPredictions.select("prediction", "label").rdd
dtMetrics = MulticlassMetrics(dtPredictionAndLabels)
dtConfusionMatrix = dtMetrics.confusionMatrix()
print("Decision Tree Confusion Matrix:")
print(dtConfusionMatrix)


accuracyLr = lrMetrics.accuracy
print("Logistic Regression Accuracy: ", accuracyLr)


precisionLr = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "weightedPrecision"})
print("Logistic Regression Precision: ", precisionLr)
precisionNb = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "weightedPrecision"})
print("Naive Bayes Precision: ", precisionNb)
#precisionSVM = svmEvaluator.evaluate(svmPredictions, {svmEvaluator.metricName: "weightedPrecision"})
#print("SVM Precision: ", precisionSVM)
precisionDT = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "weightedPrecision"})
print("Decision Tree Precision: ", precisionDT)



lrRecall = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "weightedRecall"})
nbRecall = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "weightedRecall"})
#svmRecall = svmEvaluator.evaluate(svmPredictions, {svmEvaluator.metricName: "weightedRecall"})
dtRecall = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "weightedRecall"})
print("Logistic Regression Recall: " + str(lrRecall))
print("Naive Bayes Recall: " + str(nbRecall))
#print("SVM Recall: " + str(svmRecall))
print("Decision Tree Recall: " + str(dtRecall))


#f1ScoreLr = lrMetrics.fMeasure()
#f1ScoreLr = (2*precisionLr*lrRecall)/(precisionLr + lrRecall) # ovo NE BRISI
f1ScoreLr = lrEvaluator.evaluate(lrPredictions, {lrEvaluator.metricName: "f1"})
print("Logistic Regression F1 Score: ", f1ScoreLr)
#f1ScoreNB = (2*precisionNb*nbRecall)/(precisionNb + nbRecall) # NE BRISI
f1ScoreNB = nbEvaluator.evaluate(nbPredictions, {nbEvaluator.metricName: "f1"})
print("Naive Bayes F1 Score: ", f1ScoreNB)
#f1ScoreSVM = (2*precisionSVM*svmRecall)/(precisionSVM + svmRecall)
#print("SVM F1 Score: ", f1ScoreSVM)
#f1ScoreDT = (2*precisionDT*dtRecall)/(precisionDT + dtRecall) #NE BRISI
f1ScoreDT = dtEvaluator.evaluate(dtPredictions, {dtEvaluator.metricName: "f1"})
print("Decision Tree F1 Score: ", f1ScoreDT)



lrcm = lrConfusionMatrix.toArray()
TP = lrcm[0][0]
TN = lrcm[1][1]
FP = lrcm[1][0]
FN = lrcm[0][1]
P = TP + FN
N = FP + TN
lrsensitivity = TP/P #(TP) / (TP + FN)
lrspecificity = TN/N #(TN) / (TN + FP)
lrpe = (TP+FP)*(TP+FN)/N*N + (TN+FN)*(TN+FP)/N*N  
print("Logistic Regression Sensitivity: ", lrsensitivity)
print("Logistic Regression Specificity: ", lrspecificity)
print("Logistic Regression Kappa: ", lrpe)


nbcm = nbConfusionMatrix.toArray()
TP = nbcm[0][0]
TN = nbcm[1][1]
FP = nbcm[1][0]
FN = nbcm[0][1]
P = TP + FN
N = FP + TN
nbsensitivity = TP/P 
nbspecificity = TN/N 
nbpe = (TP+FP)*(TP+FN)/N*N + (TN+FN)*(TN+FP)/N*N  
print("Naive Bayes Sensitivity: ", nbsensitivity)
print("Naive Bayes Specificity: ", nbspecificity)
print("Naive Bayes Kappa: ", nbpe)

#accurracy = (TP+TN)/(P+N)
svmcm = svmConfusionMatrix.toArray()
TP = svmcm[0][0]
TN = svmcm[1][1]
FP = svmcm[1][0]
FN = svmcm[0][1]
P = TP + FN
N = FP + TN
svmsensitivity = TP/P 
svmspecificity = TN/N 
precisionSVM = TP/(TP+FP)
svmRecall = TP/(TP+FN)
svmf1 = (2*precisionSVM*svmRecall)/(precisionSVM + svmRecall)
svmpe = (TP+FP)*(TP+FN)/N*N + (TN+FN)*(TN+FP)/N*N  
print("SVM Sensitivity: ", svmsensitivity)
print("SVM Specificity: ", svmspecificity)
print("SVM Precision: ", precisionSVM)
print("SVM Recall: ", svmRecall)
print("SVM F1: ", svmf1)
print("SVM Kappa: ", svmpe)


dtcm = svmConfusionMatrix.toArray()
TP = dtcm[0][0]
TN = dtcm[1][1]
FP = dtcm[1][0]
FN = dtcm[0][1]
P = TP + FN
N = FP + TN
dtsensitivity = TP/P 
dtspecificity = TN/N 
dtpe = (TP+FP)*(TP+FN)/N*N + (TN+FN)*(TN+FP)/N*N 
print("Decision Tree Sensitivity: ", dtsensitivity)
print("Decision Tree Specificity: ", dtspecificity)
print("Decision Tree Kappa: ", dtpe)



print("Logistic Regression Accuracy: " + str(lrAccuracy))
print("Naive Bayes Accuracy: " + str(nbAccuracy))
print("SVM Accuracy: " + str(svmAccuracy))
print("Decision Tree Accuracy: " + str(dtAccuracy))



lrErrorRate = 1-lrAccuracy
nbErrorRate = 1-nbAccuracy
svmErrorRate = 1-svmAccuracy
dtErrorRate = 1-dtAccuracy
print("Logistic Regression Error Rate: " + str(lrErrorRate))
print("Naive Bayes Error Rate: " + str(nbErrorRate))
print("SVM Error Rate: " + str(svmErrorRate))
print("Decision Tree Error Rate: " + str(dtErrorRate))

with open('rezultati.txt', 'w') as f:
    f.write("Logistic Regression Accuracy: " + str(lrAccuracy) + "\n")
    f.write("Logistic Regression Error Rate: " + str(lrErrorRate)+ "\n")
    f.write("Logistic Regression Sensitivity: "+ str(lrsensitivity)+ "\n")
    f.write("Logistic Regression Specificity: "+ str(lrspecificity)+ "\n")
    f.write("Logistic Regression Precision: "+ str(precisionLr)+ "\n")
    f.write("Logistic Regression Recall: "+ str(lrRecall)+ "\n")
    f.write("Logistic Regression F-score: "+ str(f1ScoreLr)+ "\n")
    f.write("Logistic Regression Kappa: "+ str(lrpe) + "\n")
    f.write("Logistic Regression Confusion Matrix:")
    for row in lrcm:
        f.write(" ".join(str(int(x)) for x in row))
    
    f.write("Naive Bayes Accuracy: " + str(nbAccuracy) + "\n")
    f.write("Naive Bayes Error Rate: " + str(nbErrorRate)+ "\n")
    f.write("Naive Bayes Sensitivity: "+ str(nbsensitivity)+ "\n")
    f.write("Naive Bayes Specificity: "+ str(nbspecificity)+ "\n")
    f.write("Naive Bayes Precision: "+ str(precisionNb)+ "\n")
    f.write("Naive Bayes Recall: "+ str(nbRecall)+ "\n")
    f.write("Naive Bayes F-score: "+ str(f1ScoreNB)+ "\n")
    f.write("Naive Bayes Kappa: "+ str(nbpe)+"\n")
    f.write("Naive Bayes Confusion Matrix:")
    for row in nbcm:
        f.write(" ".join(str(int(x)) for x in row))

    f.write("SVM Accuracy: " + str(svmAccuracy) + "\n")
    f.write("SVM Error Rate: " + str(svmErrorRate) + "\n")
    f.write("SVM Sensitivity: "+ str(svmsensitivity)+ "\n")
    f.write("SVM Specificity: "+ str(svmspecificity)+ "\n")
    f.write("SVM Precision: "+ str(precisionSVM) + "\n")
    f.write("SVM recall: "+ str(svmRecall) + "\n")
    f.write("SVM F-score: " + str(svmf1) + "\n")
    f.write("SVM Kappa: " + str(svmpe) + "\n")
    f.write("SVM Confusion Matrix:")
    for row in svmcm:
        f.write(" ".join(str(int(x)) for x in row))

    f.write("Decision Tree Accuracy: " + str(dtAccuracy) + "\n")
    f.write("Decision Tree Error Rate: " + str(dtErrorRate)+ "\n")
    f.write("Decision Tree Sensitivity: "+ str(dtsensitivity)+ "\n")
    f.write("Decision Tree Specificity: "+ str(dtspecificity)+ "\n")
    f.write("Decision Tree Precision: "+ str(precisionDT)+ "\n")
    f.write("Decision Tree Recall: "+ str(dtRecall)+ "\n")
    f.write("Decision Tree F-score: "+ str(f1ScoreDT)+ "\n")
    f.write("Decision Tree Kappa: "+ str(dtpe) + "\n")
    f.write("Decision Tree Confusion Matrix:")
    for row in dtcm:
        f.write(" ".join(str(int(x)) for x in row))

spark.stop()