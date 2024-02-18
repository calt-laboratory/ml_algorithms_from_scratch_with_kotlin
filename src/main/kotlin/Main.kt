package org.example

import org.example.dataPreProcessing.dataPreProcessing
import org.example.regressionAlgorithms.LinearRegression
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.io.readCSV
import org.jetbrains.kotlinx.dataframe.size

fun main() {
    val data = DataFrame.readCSV(fileOrUrl = "src/main/resources/original_breast_cancer_classification.csv")

    val (preProcessedData, xData, yData) = dataPreProcessing(data)
    LinearRegression().fit(xData, yData)
}