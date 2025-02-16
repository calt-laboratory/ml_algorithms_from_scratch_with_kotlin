package org.example

import Scaler
import org.example.dataPreProcessing.trainTestSplit
import org.example.regressionAlgorithms.LinearRegression
import org.example.Math.meanSquaredError
import org.example.dataConversion.toDoubleArray
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.remove
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.dataframe.api.*

fun main() {
    val data = DataFrame.readCSV(fileOrUrl = "src/main/resources/BostonHousing.csv")
    /*
    bostonData.describe()
    println("Boston data size: ${bostonData.size()}")
    println("Boston rows count: ${bostonData.rowsCount()}")
    println("Boston column name: ${bostonData.columnNames()}")
    println("Boston columns count: ${bostonData.columnsCount()}")
    println("Boston first row: ${bostonData.first()}")
     */

    // val data = DataFrame.readCSV(fileOrUrl = "src/main/resources/original_breast_cancer_classification.csv")

    //val (preProcessedData, xData, yData) = dataPreProcessing(data)
    data.head(numRows = 5)
    val yData = data["medv"]
    val xData = data.remove("medv")
    val (xTrain, xTest, yTrain, yTest) = trainTestSplit(xData = xData, yData = yData, testSize = 0.2, randomState = 42)
    val scaler = Scaler()
    println("data.mean(skipNA = true): ${data.mean(skipNA = true)}")
    // scaler.fit(data = xTrain)
    val linearRegressor = LinearRegression(learningRate = 0.001, numberOfIterations = 100)
    linearRegressor.fit(xData = xTrain, yData = yTrain)
    val yPredicted = linearRegressor.predict(xData = xTest)
    println("#################################################################")
    println("Predicted values: ${yPredicted.contentToString()}")
    val mse = meanSquaredError(yTrue = yTest.toDoubleArray(), yPredicted = yPredicted)
    println("Mean squared error: $mse")
}