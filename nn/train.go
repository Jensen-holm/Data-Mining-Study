package nn

func (nn *NN) Train() {
	nn.InitWnB()
	nn.TrainTestSplit()
	nn.Backprop()
}
