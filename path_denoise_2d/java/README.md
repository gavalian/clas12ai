# CLAS12 2D Track Denoising - Java Port
## Prediction from file
`predict -p PREDICTION_DATA_FILE --model-config KERAS_MODEL_CONFIG --model-weights KERAS_MODEL_WEIGHTS -r OUTPUT_DIR`

## Prediction API
Predictions can be performed programmatically by calling the `predict` function for a model.
Example:
```java
DenoisingAutoEncoder model = new DenoisingAutoEncoder();
model.loadKerasModel(modelConfigPath, modelWeightsPath);

List<INDArray> features = getFeaturesFromSomewhere();

// Params: feature list, padding x, padding y, threshold
List<INDArray> predictions = model.predict(features, 2, 0, 0.5);
```
