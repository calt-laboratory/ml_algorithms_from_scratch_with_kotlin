package org.example.dataPreProcessing

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import kotlin.random.Random


data class PreProcessedDataset(val df: DataFrame<Any?>, val xData: DataFrame<Any?>, val yData: DataColumn<*>)

/**
 * Removes the id column and maps the diagnosis column to 0 and 1.
 * @param df Original dataframe
 * @return PreProcessedDataset Contains the dataframe, the features and the target
 */
fun dataPreProcessing(df: DataFrame<*>): PreProcessedDataset {
    // Remove id column
    val dfWithoutID = df.remove("id")
    // Map B (= benign) to 0 and M (= malignant) to 1
    val updatedDiagnosis = dfWithoutID["diagnosis"].map { if (it == "B") 0 else 1 }
    val updatedDataFrame = dfWithoutID.replace { it["diagnosis"] }.with { updatedDiagnosis }
    val yData = updatedDataFrame["diagnosis"]
    val xData = updatedDataFrame.remove("diagnosis")
    return PreProcessedDataset(updatedDataFrame, xData, yData)
}

data class SplitData(
    val xTrain: DataFrame<Any?>,
    val xTest: DataFrame<Any?>,
    val yTrain: DataColumn<Any?>,
    val yTest: DataColumn<Any?>
)

/**
 * Splits the data into train and test sets.
 * @param xData Features
 * @param yData Target
 * @param testSize Size of the test set
 * @param randomState Random state used to create the random indices
 * @return SplitData Contains the train and test set
 */
fun trainTestSplit(xData: DataFrame<Any?>, yData: DataColumn<*>, testSize: Double, randomState: Int): SplitData {
    val random = Random(seed = randomState)
    // Create random indices (= integers) aligned w/ the number of rows in the dataframe
    val indices = (0 until xData.count()).shuffled(random)
    // Index where to split the dataframe
    val splitIndex = (xData.count() * (1 - testSize)).toInt()

    // Create indices for train and test set
    val trainIndices = indices.subList(0, splitIndex)
    val testIndices = indices.subList(splitIndex, xData.count())

    // Create train and test sets for features and target
    val xTrain = xData[trainIndices]
    val xTest = xData[testIndices]
    val yTrain = yData[trainIndices]
    val yTest = yData[testIndices]

    return SplitData(xTrain, xTest, yTrain, yTest)
}