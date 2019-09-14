/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.jnp.spark;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.jlab.jnp.clas.dc.DCSectorGroup;
import org.jlab.jnp.clas.dc.DCTrack;
import org.jlab.jnp.clas.ui.DCTrackCanvas2D;
import org.jlab.jnp.clas.ui.DCTrackView2D;
import org.jlab.jnp.graphics.base.Node2D;
import org.jlab.jnp.hipo4.data.Bank;
import org.jlab.jnp.hipo4.data.Event;
import org.jlab.jnp.hipo4.data.Schema;
import org.jlab.jnp.hipo4.io.HipoReader;
import org.jlab.jnp.hipo4.operations.BankIterator;
import org.jlab.jnp.hipo4.operations.BankSelector;

/**
 *
 * @author gavalian
 */
public class SparkTrackViewer extends JPanel implements ActionListener {
    
    private final DCTrackCanvas2D    canvas2D = new DCTrackCanvas2D();
    //private final DCTrackCanvas      canvasData = new DCTrackCanvas(0.3,0.3);

    DCTrackFitter                   fitter = new DCTrackFitter();
    
    private JPanel        buttonPanel = null;
    private final HipoReader             reader = new HipoReader();
    
    public SparkTrackViewer(){
        super();
        initUI();
        initViews();
        
        fitter.loadModel("DCML.nnet");
    }
    
    private void initViews(){
        
        /*DCTrackView2D viewRaw = new DCTrackView2D();
        DCTrackView2D viewHb  = new DCTrackView2D();
        DCTrackView2D viewNa  = new DCTrackView2D();
        DCTrackView2D viewMY  = new DCTrackView2D();
        
        viewRaw.setBoundsBind(0.0, 0.0, 1.0, 0.25);
        viewRaw.alignMode(Node2D.ALIGN_RELATIVE);
        viewHb.setBoundsBind(0.0, 0.25, 1.0, 0.25);
        viewHb.alignMode(Node2D.ALIGN_RELATIVE);
        viewNa.setBoundsBind(0.0, 0.5, 1.0, 0.25);
        viewNa.alignMode(Node2D.ALIGN_RELATIVE);
        viewMY.setBoundsBind(0.0, 0.75, 1.0, 0.25);
        viewMY.alignMode(Node2D.ALIGN_RELATIVE);
        
        canvas2D.addView(viewRaw);
        canvas2D.addView(viewHb);
        canvas2D.addView(viewNa);
        canvas2D.addView(viewMY);
        
        viewHb.setFillColor(Color.BLUE);
        viewNa.setFillColor(Color.BLACK);*/
        
        for(int i = 0; i < 8; i++){
            DCTrackView2D view = new DCTrackView2D();
            canvas2D.addView(view);
        }
        canvas2D.getCanvas().divide(2, 4);
    }
    
    private void initUI(){
        setLayout(new BorderLayout());
        add(canvas2D,BorderLayout.CENTER);
        buttonPanel = new JPanel();
        buttonPanel.setLayout(new FlowLayout());
        JButton nextButton = new JButton(">");
        nextButton.addActionListener(this);
        buttonPanel.add(nextButton);
        add(buttonPanel,BorderLayout.PAGE_END);
    }
    
    public void next(){
       Event event = new Event();
       Bank  dcBank = new Bank(reader.getSchemaFactory().getSchema("DC::tdc"));
       if(reader.hasNext()==true){
           reader.nextEvent(event);
           event.read(dcBank);
           BankSelector selector = new BankSelector(reader.getSchemaFactory().getSchema("DC::tdc"));
           selector.add("sector==1");
           BankIterator indexSet = new BankIterator(1200);
           selector.getIndexSet(event, indexSet);
           DCTrack track = new DCTrack();
           track.setData(selector.getBank(), indexSet);
           canvas2D.getTrackViews().get(0).setTrack(track);
           canvas2D.repaint();
           Schema sc  = reader.getSchemaFactory().getSchema("HitBasedTrkg::HBHits");
           sc.show();
           BankSelector selector_hb = new BankSelector(reader.getSchemaFactory().getSchema("HitBasedTrkg::HBHits"));
           selector_hb.add("sector==1&&clusterID>0");
           selector_hb.getIndexSet(event, indexSet);
           System.out.println(" LENGTH = " + indexSet.count());
           DCTrack track_hb = new DCTrack();
           for(int i = 0; i < indexSet.count(); i++){
               int superlayer = selector_hb.getBank().getInt("superlayer", indexSet.getIndex(i));
               int layer = selector_hb.getBank().getInt("layer", indexSet.getIndex(i));
               int wire = selector_hb.getBank().getInt("wire", indexSet.getIndex(i));
               int trueLayer = (superlayer-1)*6+(layer-1);
               track_hb.setWire(trueLayer,wire-1);
           }
           /*DCTrack track_hb = new DCTrack();
           track_hb.setData(selector_hb.getBank(), indexSet);*/
           canvas2D.getTrackViews().get(1).setTrack(track_hb);
           canvas2D.repaint();
           
           BankSelector selector_na = new BankSelector(reader.getSchemaFactory().getSchema("HitBasedTrkg::HBHits"));
           selector_na.add("sector==1&&trkID>0");
           selector_na.getIndexSet(event, indexSet);
           System.out.println(" LENGTH = " + indexSet.count());
           DCTrack track_na = new DCTrack();
           for(int i = 0; i < indexSet.count(); i++){
               int superlayer = selector_hb.getBank().getInt("superlayer", indexSet.getIndex(i));
               int layer = selector_hb.getBank().getInt("layer", indexSet.getIndex(i));
               int wire = selector_hb.getBank().getInt("wire", indexSet.getIndex(i));
               int trueLayer = (superlayer-1)*6+(layer-1);
               track_na.setWire(trueLayer,wire-1);
           }
           canvas2D.getTrackViews().get(2).setTrack(track_na);
           
           track_na.show();
           DCTrack trackMY = new DCTrack();
           
           DCSectorGroup group = new DCSectorGroup();
           group.updateIterator(dcBank);
           group.group(dcBank);
           group.clean();
           group.createMap(dcBank, trackMY);
           canvas2D.getTrackViews().get(3).setTrack(trackMY);
           
           canvas2D.repaint();
           List<DCTrack> trackCandidates = group.getCombinations(dcBank);
           for(int i = 0; i < trackCandidates.size();i++){
               int difference = track_na.difference(trackCandidates.get(i));
               int coincidence = track_na.coincidence(trackCandidates.get(i));
               //System.out.printf("%3d : diff = %5d, coin = %5d\n",i,difference,coincidence);
           }
           
           
           int index = track_na.getBestMatch(trackCandidates, 0);
           System.out.println("SYSTEM DIFFERENCE ==========");
           if(index>=0){
               int difference  = track_na.difference(trackCandidates.get(index));
               int coincidence = track_na.coincidence(trackCandidates.get(index));
               int index_second = track_na.getBestMatch(trackCandidates, difference+1);
               int difference_second  = track_na.difference(trackCandidates.get(index_second));
               int coincidence_second = track_na.coincidence(trackCandidates.get(index_second));
               System.out.printf("%3d : diff = %5d, coin = %5d\n",index,difference,coincidence);
               System.out.printf("%3d : diff = %5d, coin = %5d\n",index_second,difference_second,coincidence_second);
               double[] features = trackCandidates.get(index).getFeatures();
               for(int k = 0; k < features.length; k++){
                   System.out.printf("%4d : %6.4f\n",k,features[k]);
               }
           }
           List<DCTrack> trackCandidatesChecked = group.getCombinationsChecked(dcBank);
           System.out.println("TRACK COMBINATIONS = " + trackCandidates.size());
           System.out.println("TRACK COMBINATIONS CHECKED = " + trackCandidatesChecked.size());
           
           List<DCTrack> orderedTracks = selectBest(trackCandidatesChecked,track_na);
           for(int i = 0; i < 4; i++){
               if(orderedTracks.size()>i){
                   canvas2D.getTrackViews().get(i+4).setTrack(orderedTracks.get(i));
               }
           }
           DatasetUtils   utils = new DatasetUtils();
           for(int i = 0 ; i < orderedTracks.size(); i++){
               utils.addRow(orderedTracks.get(i).getLayerFeatures(), 0.0);
           }
           Dataset<Row> dataFrame = utils.createDataset();
           fitter.decide(dataFrame);
           //canvas.setTrack(track);
           //canvas.repaint();
       }
    }
    
    
    public List<DCTrack>  selectBest(List<DCTrack> candidates, DCTrack realTrack){
        List<DCTrack> selected = new ArrayList<DCTrack>();
        int index = realTrack.getBestMatch(candidates, 0);
        if(index>=0){
            selected.add(candidates.get(index));
            while(index>=0){
                int difference = realTrack.difference(candidates.get(index));
                index = realTrack.getBestMatch(candidates, difference+1);
                if(index>=0)
                    selected.add(candidates.get(index));
            }
        }
        return selected;
    }
    public void openFile(String file){
        reader.open(file);
    }
    
    @Override
    public void actionPerformed(ActionEvent e) {
        if(e.getActionCommand().compareTo(">")==0){
            next();
        }
    }
    
    public static void main(String[] args){
        JFrame frame = new JFrame();
        SparkTrackViewer viewer = new SparkTrackViewer();
        frame.resize(1400, 900);
        viewer.openFile("/Users/gavalian/Work/Software/project-7a.0.0/cooked_005038.01102.hipo");
        frame.add(viewer);
        frame.setVisible(true);
    }
   
}
