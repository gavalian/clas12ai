/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.crtcjlab.denoise2d.clas12;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.crtcjlab.denoise2d.models.DenoisingAutoEncoder;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

/**
 *
 * @author gavalian
 */
public class DenoiserModelPool extends AbsObjectPool<DenoisingAutoEncoder> {
    
    private String config = "models/cnn_autoenc_config.json";
//"models/cnn_autoenc_config.json","models/cnn_autoenc_weights.h5"
    private String weights = "models/cnn_autoenc_weights.h5";
    
    public DenoiserModelPool(final int poolSize){
        super(poolSize);
    }
    
    public DenoiserModelPool(final int poolSize, String cfg, String wgt){
        super(poolSize);
        config = cfg; weights = wgt;
    }
    
    @Override
    protected DenoisingAutoEncoder createObject() {
        DenoisingAutoEncoder model = new DenoisingAutoEncoder();
        try {
            model.loadKerasModel(config, weights);
        } catch (UnsupportedKerasConfigurationException | IOException | InvalidKerasConfigurationException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        }
        return model;
    }
    
}
