/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.jlab.jnp.clas.dc.DCSectorGroup;
import org.jlab.jnp.clas.dc.DCTrack;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.jlab.jnp.hipo4.operations.BankIterator;
import org.jlab.jnp.hipo4.operations.BankSelector;
import org.jlab.jnp.utils.benchmark.ProgressPrintout;

/**
 *
 * @author gavalian
 */
public class CLAS12DataTrainer {
    
    DatasetUtils dsUtils = new DatasetUtils();
    DatasetUtils dsUtilsPositive = new DatasetUtils();
    DatasetUtils dsUtilsNegative = new DatasetUtils();
    
    public CLAS12DataTrainer(){
        
    }
    
    
    public void processFile(String file){
        HipoReader reader = new HipoReader();
        reader.open(file);
        
        Event event = new Event();
        BankIterator indexSet = new BankIterator(500);
        BankSelector selector_na = new BankSelector(reader.getSchemaFactory().getSchema("HitBasedTrkg::HBHits"));
        selector_na.add("sector==1&&trkID>0");
        Bank  dcBank = new Bank(reader.getSchemaFactory().getSchema("DC::tdc"));
        ProgressPrintout printout = new ProgressPrintout();
        
        while(reader.hasNext()==true){
            printout.updateStatus();
            reader.nextEvent(event);
            event.read(dcBank);
            
            selector_na.getIndexSet(event, indexSet);
            
            DCTrack track_na = new DCTrack();
            for(int i = 0; i < indexSet.count(); i++){
                int superlayer = selector_na.getBank().getInt("superlayer", indexSet.getIndex(i));
                int layer = selector_na.getBank().getInt("layer", indexSet.getIndex(i));
                int wire = selector_na.getBank().getInt("wire", indexSet.getIndex(i));
                int trueLayer = (superlayer-1)*6+(layer-1);
                track_na.setWire(trueLayer,wire-1);
            }
            
            if(track_na.hitCount()>24&&track_na.hitCount()<50){
                DCSectorGroup group = new DCSectorGroup();
                group.updateIterator(dcBank);
                group.group(dcBank);
                group.clean();
                List<DCTrack> tc = group.getCombinations(dcBank);
                if(tc.size()<=30){
                    //System.out.println(" TRACK SIZE = " + track_na.hitCount() + " combinations =  " + tc.size() );
                    int index = track_na.getBestMatch(tc, 0);
                    if(index>=0){
                        int difference = track_na.difference(tc.get(index));
                        int index_2 = track_na.getBestMatch(tc, difference+1);
                        
                        if(index_2>=0){
                            int diff_2 = track_na.difference(tc.get(index_2));
                            int coin_2 = track_na.coincidence(tc.get(index_2));
                            int index_3 = track_na.getBestMatch(tc, diff_2+1);
                            //System.out.printf(" %5d : %5d %5d \n",index_2,diff_2,coin_2);
                            int charge = track_na.getCharge();
                            //System.out.printf(" %5d : %5d %5d charge = %4d\n",index_2,diff_2,coin_2,charge);
                            //double[] rightOne = track_na.getFeatures();
                            //double[] wrongOne = tc.get(index_2).getFeatures();
                            double[] rightOne = track_na.getLayerFeatures();
                            double[] wrongOne = tc.get(index_2).getLayerFeatures();
                            /*for(int i = 0; i < rightOne.length; i++){
                                System.out.printf(" layer = %5d, feature = %8.4f\n",i,rightOne[i]);
                            }*/
                            if(charge>0){
                                this.dsUtils.addRow(rightOne, 1.0);
                                this.dsUtils.addRow(wrongOne, 0.0);
                                
                                this.dsUtilsPositive.addRow(rightOne, 1.0);
                                this.dsUtilsNegative.addRow(wrongOne, 0.0);
                                if(index_3>=0){
                                    double[] wrongTwo = tc.get(index_3).getLayerFeatures();
                                    this.dsUtils.addRow(wrongTwo, 0.0);
                                }
                            }
                        }
                    }
                }
                
            }
        }
        reader.close();
        System.out.println(" TRAINING SET SIZE = " + dsUtils.getSize());
    }
    
    public Dataset<Row> getDatasetPositive(){
        return this.dsUtilsPositive.createDataset();
    }
    
    public Dataset<Row> getDatasetNegative(){
        return this.dsUtilsNegative.createDataset();
    }
    
    public Dataset<Row> getDataset(){
        return this.dsUtils.createDataset();
    }
    
    public static void main(String[] args){
        CLAS12DataTrainer trainer = new CLAS12DataTrainer();
        trainer.processFile("/Users/gavalian/Work/Software/project-7a.0.0/cooked_005038.01102.hipo");
        
        DCTrackFitter fitter = new DCTrackFitter();
        LogisticRegressionTraining fitter2 = new LogisticRegressionTraining();
        
        Dataset<Row> dataFrame = trainer.getDataset();
        Dataset<Row> dataFramePositive = trainer.getDatasetPositive();
        Dataset<Row> dataFrameNegative = trainer.getDatasetNegative();
        //fitter.loadModel("DCML.nnet", dataFramePositive);
        fitter.fit(dataFrame);
        //fitter2.fit(dataFrame);
    }
}
