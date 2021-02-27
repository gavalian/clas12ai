package org.crtc_jlab.path.denoise2d.models;

public class DenoisingAutoEncoder extends AbstractCnnDenoisingAutoEncoder{

	public static AbstractCnnDenoisingAutoEncoder getAutoEncoderByName(String modelName) {
		
		AbstractCnnDenoisingAutoEncoder denoising_ae = null;
		
		if(modelName.equals("0"))
        {
        	denoising_ae = new DenoisingAutoEncoder0();
        }
		else if(modelName.equals("0a"))
        {
        	denoising_ae = new DenoisingAutoEncoder0a();
        }
		else if(modelName.equals("0b"))
        {
        	denoising_ae = new DenoisingAutoEncoder0b();
        }
		else if(modelName.equals("0c"))
        {
        	denoising_ae = new DenoisingAutoEncoder0c();
        }
		else if(modelName.equals("0d"))
        {
        	denoising_ae = new DenoisingAutoEncoder0d();
        }
		else if(modelName.equals("0e"))
        {
        	denoising_ae = new DenoisingAutoEncoder0e();
        }
		else if(modelName.equals("0f"))
        {
        	denoising_ae = new DenoisingAutoEncoder0f();
        }
		else
		{
			System.out.println("Unexpected model name found. Exiting");
			System.exit(1);
		}
		
		return denoising_ae;
		
	}

	@Override
	public void buildModel() {
		System.out.println("Must choose a named autoencoder. Exiting");
		System.exit(1);
	}
}
