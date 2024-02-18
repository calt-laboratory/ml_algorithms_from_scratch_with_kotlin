package org.example.regressionAlgorithms

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.rows
import org.jetbrains.kotlinx.dataframe.size

class LinearRegression (private var learningRate: Double = 0.001, private var numberOfIterations: Int = 1000) {
    private var weights: DoubleArray? = null
    private var bias: Double? = null

    fun fit(xData: DataFrame<Any?>, yData: DataColumn<Any?>) {
        // Parameter initialization
        val (nFeatures, nSamples) = xData.size()
        println("nSamples: $nSamples, nFeatures: $nFeatures")
        weights = DoubleArray(size = nFeatures) { 0.0 }
        println("Weights: ${weights?.size}")
        bias = 0.0


        repeat(numberOfIterations) {
            val convertedDataFrame = convertDataFrameToDoubleArray(xData)
            val dotProductResult = dotProduct(matrix = convertedDataFrame, vector = weights)
            val yPredicted = dotProductResult.map { it + bias!! }.toDoubleArray()

            // Gradient descent computation
            val dwArray = dotProduct(matrix = transpose(convertedDataFrame), vector = subtract(yPredicted, yData.toDoubleArray()))
            val dw = dwArray.map { it * (1.0 / nSamples) }.toDoubleArray()
            val db = (1.0 / nSamples) * subtract(yPredicted, yData.toDoubleArray()).sum()

            // Update parameters (weights and bias)
            weights = weights?.mapIndexed { idx, weight -> weight - learningRate * dw[idx] }?.toDoubleArray()
            bias = bias?.minus(learningRate * db)
        }
    }
}

fun convertDataFrameToDoubleArray(df: DataFrame<Any?>): Array<DoubleArray> {
    return df.rows().map { row ->
        row.values().mapNotNull {
            when (it) {
                is Number -> it.toDouble()
                else -> null // Ignore non-numeric values
            }
        }.toDoubleArray()
    }.toTypedArray()
}

fun dotProduct(matrix: Array<DoubleArray>, vector: DoubleArray?): DoubleArray {
    return matrix.map { row ->
        if (row.size != vector?.size) throw IllegalArgumentException("Vector must have the same size!")
        row.zip(vector) { a, b -> a * b }.sum()
    }.toDoubleArray()
}

fun transpose(matrix: Array<DoubleArray>): Array<DoubleArray> {
    // Checks if the matrix is empty to determine the number of columns of the first row
    if (matrix.isEmpty()) return arrayOf()
    // Determines the number of columns in the original matrix
    val cols = matrix.first().size
    return Array(cols) { col ->
        DoubleArray(matrix.size) { row ->
            matrix[row][col]
        }
    }
}

fun subtract(a: DoubleArray, b: DoubleArray): DoubleArray {
    if (a.size != b.size) throw IllegalArgumentException("Arrays must have the same size!")
    return DoubleArray(a.size) { index ->
        a[index] - b[index]
    }
}

fun DataColumn<Any?>.toDoubleArray(): DoubleArray {
    return this.values().mapNotNull {
        when (it) {
            is Number -> it.toDouble()
            else -> null
        }
    }.toDoubleArray()
}
