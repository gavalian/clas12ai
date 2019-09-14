/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author gavalian
 */
public class DCSparkSession {
    SparkSession spark;
    public static DCSparkSession dcSession = null;
    
    public DCSparkSession(){
        spark = SparkSession.builder().appName("DCML").master("local[4]").getOrCreate();
    }
    
    public static DCSparkSession getInstance(){
        if(dcSession==null){
            dcSession = new DCSparkSession();
        }
        return dcSession;
    }
    
    public SparkSession getSession(){ return spark;}
    
}
