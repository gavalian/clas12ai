package org.crtcjlab.denoise2d.models;

public class DenoisingAutoEncoder extends AbstractCnnDenoisingAutoEncoder {

    public static AbstractCnnDenoisingAutoEncoder getAutoEncoderByName(String modelName) {

        AbstractCnnDenoisingAutoEncoder denoisingAutoEncoder = null;

        switch (modelName) {
            case "0":
                denoisingAutoEncoder = new DenoisingAutoEncoder0();
                break;
            case "0a":
                denoisingAutoEncoder = new DenoisingAutoEncoder0a();
                break;
            case "0b":
                denoisingAutoEncoder = new DenoisingAutoEncoder0b();
                break;
            case "0c":
                denoisingAutoEncoder = new DenoisingAutoEncoder0c();
                break;
            case "0d":
                denoisingAutoEncoder = new DenoisingAutoEncoder0d();
                break;
            case "0e":
                denoisingAutoEncoder = new DenoisingAutoEncoder0e();
                break;
            case "0f":
                denoisingAutoEncoder = new DenoisingAutoEncoder0f();
                break;
            case "0g":
                denoisingAutoEncoder = new DenoisingAutoEncoder0g();
                break;
            default:
                System.out.println("Unexpected model name found. Exiting");
                System.exit(1);
        }

        return denoisingAutoEncoder;
    }

    @Override
    public void buildModel() {
        System.out.println("Must choose a named autoencoder. Exiting");
        System.exit(1);
    }
}
