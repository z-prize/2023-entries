Download the data files from:
https://drive.google.com/file/d/1AD4vpY5n_NyuLqZfFSO9y799vSLCM8Tu

Compile the two programs:
```
g++ SimulatorMain.cpp -o s -lgmp
g++ MSM.cpp -o m -lgmp
```

Run:
```
./s <size>
./m <size>
```

Size can range from 1 to 262144.
