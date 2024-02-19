package org.example.dataConversion

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.rows

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

fun DataColumn<Any?>.toDoubleArray(): DoubleArray {
    return this.values().mapNotNull {
        when (it) {
            is Number -> it.toDouble()
            else -> null
        }
    }.toDoubleArray()
}