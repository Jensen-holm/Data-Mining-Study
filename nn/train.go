package nn

func (nn *NN) Train() {
	nn.InitWnB()
	nn.TrainTestSplit()

	// iterate n times where n = nn.Epochs
	// use backprop algorithm on each iteration
	// to fit the model to the data
	for i := 0; i < nn.Epochs; i++ {
	}

}
