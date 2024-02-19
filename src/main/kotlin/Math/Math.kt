package org.example.Math

import kotlin.math.pow

fun dotProduct(matrix: Array<DoubleArray>, vector: DoubleArray): DoubleArray {
    return matrix.map { row ->
        if (row.size != vector.size) throw IllegalArgumentException("Vector must have the same size!")
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


fun meanSquaredError(yTrue: DoubleArray, yPredicted: DoubleArray): Double {
    if (yTrue.size != yPredicted.size) throw IllegalArgumentException("Arrays must have the same size!")
    return (1.0 / yTrue.size) * yTrue.zip(yPredicted) { a, b -> (a - b).pow(2.0) }.sum()
}


fun calculateZScore(x: Double, mean: Double, std: Double) : Double {
    return (x - mean) / std
}
