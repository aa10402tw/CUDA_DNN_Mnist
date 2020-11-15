Compile and execute
```
nvcc mat.cu mat_math.cu act.cu layer_fc.cu main.cu -o main
main.exe
```

Main Program
```
int main() {
    // Load Data for minist
    loaddata();

    // Peek the dataset
    for (int i=0; i<5; i++)
        printImage(train_set[i]);

    // Define Layer architceture for fully-connected layer with activation function
    std::vector<Layer_fc*> net;
    net.push_back(new Layer_fc(28*28, 512, (ActivationFunction*) new Act_Sigmoid()));
    net.push_back(new Layer_fc(512,   256, (ActivationFunction*) new Act_Sigmoid()));
    net.push_back(new Layer_fc(256,    10, (ActivationFunction*) new Act_NONE()));

    // Define the hyper parameters and log interval (-1 for no log)
    int batch_size = 1;
    float lr = 0.01;
    int log_intervel = 1000 / batch_size;
    int epochs = 1;

    // Training the network
    for (int i=0; i<epochs; i++)
        train(net, batch_size, lr, log_intervel);

    // Test the network
    test(net);
}
```

Result <br/>
Train acc: 96.82% <br/>
Test acc: 96.57% <br/>