import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import sys

if len(sys.argv) != 2:
    print("Please specify file to load and sample id")

filename = sys.argv[1]

print("Loading dataset...")
imgs_sparse, labels = load_svmlight_file(filename)
print("Done")
clean_track = []  
noisy_track = []

for i in range(0, len(labels)):
    if labels[i] == 0:
        noisy_track.append(imgs_sparse[i])
    else:
        clean_track.append(imgs_sparse[i])


sample_id = int(input("Please enter a sample id to display in range 0 - "+str(len(labels)/2)+ " or -1 to exit\n"))
while sample_id != -1:
    img_clean = clean_track[sample_id].todense().reshape((-1,36,112))
    img_noisy = noisy_track[sample_id].todense().reshape((-1,36,112))
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Noisy Input")
    ax2.set_title("Clean Label")
    ax1.imshow(img_noisy)
    ax2.imshow(img_clean)

    plt.show()
    sample_id = int(input("Please enter a sample id to display in range 0 - "+str(len(labels)/2)+ " or -1 to exit\n"))
