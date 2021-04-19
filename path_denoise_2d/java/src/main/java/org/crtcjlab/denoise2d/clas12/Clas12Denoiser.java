/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.crtcjlab.denoise2d.clas12;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.crtcjlab.denoise2d.models.DenoisingAutoEncoder;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.jlab.jnp.hipo4.io.HipoWriter;
import org.jlab.jnp.utils.options.OptionParser;
import org.jlab.jnp.utils.options.OptionStore;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author gavalian
 */
public class Clas12Denoiser {
    
    DenoisingAutoEncoder model = null;
            
    long executionTimeML = 0L;
    long executionCountML = 0L;
    
    long executionTime = 0L;
    long executionCount = 0L;
    
    
    boolean hitRecovery = false;
    
    
    public Clas12Denoiser(){
        
    }
    
    public Clas12Denoiser(String config, String weights){
        init(config,weights);
    }
    
    public static Clas12Denoiser withFile(String config, String weights){
        Clas12Denoiser denoiser = new Clas12Denoiser();
        denoiser.init(config, weights);
        return denoiser;
    }
    
    
    public Clas12Denoiser setHitRecovery(boolean flag){
        System.out.println("****************************************************");
        System.out.println("* TURN HIT RECOVERY : " + flag);
        System.out.println("****************************************************");
        this.hitRecovery = flag; return this;
    }
    
    protected void init(String config, String weights){
        model = new DenoisingAutoEncoder();
        try {
            model.loadKerasModel(config, weights);
        } catch (UnsupportedKerasConfigurationException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InvalidKerasConfigurationException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public List<INDArray>  getOutput(List<INDArray> features, double threshold){
        long then = System.currentTimeMillis();
        List<INDArray> predictions = model.predict(features, 2, 0, threshold, hitRecovery);
        long now = System.currentTimeMillis();
        executionTimeML += (now-then);
        executionCountML++;
        return predictions;
    }
    
    public List<INDArray>  getDataFeatures(Bank bank){
        int x = 36;
        int y = 112;
        long then = System.currentTimeMillis();
        List<INDArray>  list = new ArrayList<>();
        for(int i = 0; i < 6; i++){
            list.add(Nd4j.zeros(1,x,y,1));
        }
        int nrows = bank.getRows();
        for(int i = 0; i < nrows; i++){
            int    sector = bank.getInt("sector", i);
            int     layer = bank.getInt("layer", i);
            int component = bank.getInt("component", i);
            
            list.get(sector-1).putScalar(new int[]{0,layer-1,component-1,0}, 1.0);
        }
        long now = System.currentTimeMillis();
        this.executionCount++;
        this.executionTime += (now-then);
        
        return list;
    }
    
    public void show(INDArray array, double threshold){
        for(int r = 0; r < 36; r++){
            for(int c = 0; c < 112; c++){
                double value = array.getDouble(new int[]{0,r,c,0});
                if(value<threshold){
                    System.out.print("-");
                } else {
                    System.out.print("X");
                }
            }
            System.out.println();
        }
    }
    
    protected List<TDC> getList(Bank bank, int sector, INDArray array){
        List<TDC> list = new ArrayList<>();
        
        int nrows = bank.getRows();
        for(int i = 0; i < nrows; i++){
            int sec = bank.getInt("sector", i);
            if(sec==sector){
                int layer = bank.getInt("layer", i);
                int  wire = bank.getInt("component",i);
                if(array.getDouble(new int[]{0,layer-1,wire-1,0})>0.5){
                    TDC entry = new TDC();
                    entry.sector = (byte) sector;
                    entry.layer = (byte) layer;
                    entry.component = (short) wire;
                    entry.order = bank.getByte("order", i);
                    entry.tdc = bank.getInt("TDC", i);
                    list.add(entry);
                }
            }
        }
        return list;
    }
    
    public Bank reduce(Bank dc, List<INDArray> prediction){
        long then = System.currentTimeMillis();
        List<TDC>  entries = new ArrayList<>();    
        for(int i = 0; i < 6; i++){
            List<TDC> entry = this.getList(dc, i+1, prediction.get(i));
            entries.addAll(entry);
        }        
        int nrows = entries.size();
        Bank dcnuevo = new Bank(dc.getSchema(),nrows);
        for(int row = 0; row < nrows; row++){
            TDC entry = entries.get(row);
            dcnuevo.putByte("sector", row, entry.sector);
            dcnuevo.putByte("layer", row, entry.layer);
            dcnuevo.putShort("component", row, entry.component);
            dcnuevo.putByte("order", row, entry.order);
            dcnuevo.putInt("TDC", row, entry.tdc);                        
        }
        long now = System.currentTimeMillis();
        this.executionCount++;
        this.executionTime += (now-then);
        return dcnuevo;
    }
    
    public void showStats(){
        double rate = ((double) (executionTimeML))/executionCountML ;
        double rateAlgo = ((double) (executionTime))/executionCount ;
        double   networkInferenceRate = this.model.getExecutionTime();
        double   paddingRate = this.model.getPaddingTime();
        System.out.printf("ML >>> exeuction time %12d, # events = %12d , rate = %8.2f msec/event, procedure = %8.2f, inference %8.4f msec/event, padding %8.4f msec/event\n",
                executionTimeML, executionCountML, rate,rateAlgo,networkInferenceRate,paddingRate);
    }
    
    public static class TDC {
        byte sector = 0;
        byte layer  = 0;
        short component = 0;
        byte  order = 0;
        int tdc = 0;
    }
    public static void speedAssess(){
         String file = "/Users/gavalian/Work/DataSpace/luminocity/rec_merged_005418_90nA.hipo";
        
        HipoReader reader = new HipoReader();
        reader.open(file);
        
        Bank dc = reader.getBank("DC::tdc");
        Event event = new Event();
        
        
        
        Clas12Denoiser denoiser = Clas12Denoiser.withFile("models/cnn_autoenc_config.json","models/cnn_autoenc_weights.h5");
        
        List<INDArray> features = new ArrayList<>(); 
        
        //reader.getEvent(event, 1200);
        //event.read(dc);
        for(int k = 0; k < 10; k++){            
            reader.getEvent(event, 1200 + k);
            event.read(dc);
            List<INDArray> f = denoiser.getDataFeatures(dc);
            features.addAll(f);
        }
        
        System.out.println(" benchmark sample size = " + features.size());
        for(int i = 0; i < 4; i++){
            List<INDArray> output = denoiser.getOutput(features, 0.05);
        }
        
        denoiser.showStats();
        
    }
    
    public static void main(String[] args){
        
        OptionParser parser = new OptionParser();
        parser.addOption("-r", "false", "turn on hit recovery");
        parser.addRequired("-o", "output file name");
        parser.parse(args);
        

            
        List<String> inputFiles = parser.getInputList();
        String          recover = parser.getOption("-r").stringValue();
        
        String file = inputFiles.get(0);
        String outputFile = parser.getOption("-o").stringValue();
        
        HipoReader reader = new HipoReader();
        reader.open(file);
        
        Bank dc = reader.getBank("DC::tdc");
        Event event = new Event();
        
        Clas12Denoiser denoiser = Clas12Denoiser.withFile("models/cnn_autoenc_config.json","models/cnn_autoenc_weights.h5");
        if(recover.compareTo("true")==0){ denoiser.setHitRecovery(true);}
        
        HipoWriter writer = new HipoWriter(reader.getSchemaFactory());        
        writer.open(outputFile);
        
        int counter = 0;
        
        while(reader.hasNext()){
            
            reader.nextEvent(event);
            event.read(dc);            
            List<INDArray> features = denoiser.getDataFeatures(dc);            
            List<INDArray> output = denoiser.getOutput(features, 0.05);            
            /*System.out.println("----> event");
            denoiser.show(features.get(0), 0.5);
            System.out.println("----> after the fix");
            denoiser.show(output.get(0), 0.5);
            */
            Bank dcnuevo = denoiser.reduce(dc, output);
            //System.out.printf("event (#%8d) DC TDC size = %4d , reduced = %4d\n",counter,dc.getRows(), dcnuevo.getRows());
            event.remove(dc.getSchema());
            
            event.write(dcnuevo);
            writer.addEvent(event);
            counter++;
            if(counter%100==0) denoiser.showStats();
            //System.out.printf("event (#%8d) DC TDC size = %4d , reduced = %4d\n",counter,dc.getRows(), dcnuevo.getRows());           
        }
        writer.close();       
    }
}
