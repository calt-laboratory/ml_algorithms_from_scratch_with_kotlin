import org.example.Math.calculateZScore
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.DataRow
import org.jetbrains.kotlinx.dataframe.api.*


class Scaler {
    private lateinit var means: DataRow<Double>
    private lateinit var stdDeviation: DataRow<Double>
    private lateinit var matrix: Array<DoubleArray>

    fun fit(data: DataFrame<Double>) {
        means = data.mean(skipNA = true)
        stdDeviation = data.std(skipNA = true)
        val matrix = Array(means.count()) { DoubleArray(stdDeviation.count()) }
    }


    // TODO: implement transform
    fun transform(data: DataFrame<Any?>) : DataFrame<Any?> = data
}



