package org.example.regressionAlgorithms

import org.example.Math.dotProduct
import org.example.Math.subtract
import org.example.Math.transpose
import org.example.dataConversion.convertDataFrameToDoubleArray
import org.example.dataConversion.toDoubleArray
import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.size

class LinearRegression (private var learningRate: Double = 0.001, private var numberOfIterations: Int = 1000) {
    private lateinit var weights: DoubleArray
    private var bias: Double = 0.0

    fun fit(xData: DataFrame<Any?>, yData: DataColumn<Any?>) {
        val (nFeatures, nSamples) = xData.size()
        println("nSamples: $nSamples, nFeatures: $nFeatures")
        weights = DoubleArray(size = nFeatures) { 0.0 }
        println("Weights: ${weights.size}")
        bias = 0.0


        repeat(numberOfIterations) {
            val convertedDataFrame = convertDataFrameToDoubleArray(xData)
            val dotProductResult = dotProduct(matrix = convertedDataFrame, vector = weights)
            val yPredicted = dotProductResult.map { it + bias }.toDoubleArray()

            // Gradient descent computation
            val dwArray = dotProduct(matrix = transpose(convertedDataFrame), vector = subtract(yPredicted, yData.toDoubleArray()))
            val dw = dwArray.map { it * (1.0 / nSamples) }.toDoubleArray()
            val db = (1.0 / nSamples) * subtract(yPredicted, yData.toDoubleArray()).sum()

            // Update parameters (weights and bias)
            weights = weights.mapIndexed { idx, weight -> weight - learningRate * dw[idx] }.toDoubleArray()
            bias = bias.minus(learningRate * db)
        }
    }

    fun predict(xData: DataFrame<Any?>) : DoubleArray {
        val convertedDataFrame = convertDataFrameToDoubleArray(xData)
        val dotProductResult = dotProduct(matrix = convertedDataFrame, vector = weights)
        val yPredicted = dotProductResult.map { it + bias }.toDoubleArray()
        return yPredicted
    }
}
