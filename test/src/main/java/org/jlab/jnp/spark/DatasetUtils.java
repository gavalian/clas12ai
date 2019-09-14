/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import java.util.ArrayList;
import java.util.List;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 *
 * @author gavalian
 */
public class DatasetUtils {
    private List<Row> dataRows = new ArrayList<>();
    
    public DatasetUtils(){
        
    }
    
    public void init(){
        
    }
    
    public void addRow(double[] array, double label){
         Row row = RowFactory.create(label,Vectors.dense(array));
         dataRows.add(row);
    }
    
    public Dataset<Row> createDataset(){
        
        StructType schema = new StructType(new StructField[]{
            new StructField("label", DataTypes.DoubleType, true, Metadata.empty()),
            //new StructField("features", DataTypes.createArrayType(DataTypes.DoubleType), false, Metadata.empty())
            new StructField("features", new VectorUDT(), true, Metadata.empty())
        });
        
        Dataset<Row> data = DCSparkSession.getInstance().getSession().createDataFrame(dataRows, schema);
        //data.show();
        return data;
    }
    
    public int getSize(){
        return this.dataRows.size();
    }
    
    public Dataset<Row> readFromFile(String filename){
        Dataset<Row> dataFrame = DCSparkSession.getInstance().getSession().read().format("libsvm").load(filename);
        //int index =  dataFrame.schema().toString();
        dataFrame.show();
        System.out.println(dataFrame.schema().toString());
        return dataFrame;
    }
    public void randomize(double[] array){
        for(int i = 0; i < array.length; i++)
            array[i] = Math.random();
    }
    
    public Dataset<Row> createRandom(int count){
        double[] array = new double[36];
        for(int i = 0; i < count; i++){
            randomize(array);
            double threshold = Math.random();
            double label     = 0.0;
            if(threshold > 0.5) label = 1.0;
            addRow(array, label);
        }
        return createDataset();
    }
    
    public static void main(String[] args){
        
        DatasetUtils utils = new DatasetUtils();
        double[] array = new double[36];
        for(int i = 0; i < 10; i++){    
            utils.randomize(array);
            utils.addRow(array, 1.0);
        }
        Dataset<Row> dataFrame = utils.createDataset();
        dataFrame.show();
        
        utils.readFromFile("evaluator_5.txt");
        System.out.println(dataFrame.schema().toString());
    }
}
