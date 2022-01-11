/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.crtcjlab.denoise2d.clas12;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.crtcjlab.denoise2d.models.DenoisingAutoEncoder;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author gavalian
 */
public class Clas12DenoiserBenchmark {
    
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
        
        return list;
    }
    
    public List<INDArray>  getDataFromFile(String filename, int iter){
        List<INDArray>  output = new ArrayList<>();
        HipoReader reader = new HipoReader();
        reader.open(filename);
        Event event = new Event();
        Bank     dc = reader.getBank("DC::tdc");
        
        for(int i = 0; i < iter; i++){
            reader.nextEvent(event);
            event.read(dc);
            List<INDArray>  data = this.getDataFeatures(dc);
            output.addAll(data);
        }
        return output;
    }
        
    public static void main(String[] args) throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException{
        Clas12DenoiserBenchmark bm = new Clas12DenoiserBenchmark();
        List<INDArray>  input = bm.getDataFromFile("/Users/gavalian/Work/DataSpace/evio/clas_003852.evio.981.hipo", 150);
        System.out.println("\t[] data loaded : " + input.size());
        
        DenoisingAutoEncoder model = new DenoisingAutoEncoder();
        model.loadKerasModel("models/cnn_autoenc_config.json","models/cnn_autoenc_weights.h5");
        int iter = 2;
        System.out.printf(" starting benchmark test\n");
        long then = System.currentTimeMillis();
        for(int i = 0; i < iter; i++){
            List<INDArray> predictions = model.predict(input, 2, 0, 0.05, false);
        }
        long now = System.currentTimeMillis();
        long executionTimeML = (now-then);
        int  count = iter*input.size();
        double time = ((double) executionTimeML)/count;
        System.out.printf("[benchmark] inference time %.2f ms/sample, %.2f ms/event\n",
                time,time*6);
        System.out.printf("[benchmark] internal padding %.2f ms/sample, process %.2f ms/event\n",
                model.getPaddingTime(),model.getExecutionTime());
    }
}
