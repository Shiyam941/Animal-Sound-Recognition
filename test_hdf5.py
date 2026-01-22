import h5py

with h5py.File("dataset/animal_sample.h5", "r") as f:
    print("Features shape:", f["features"].shape)
    print("Labels:", f["labels"][:])
